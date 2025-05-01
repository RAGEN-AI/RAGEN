"""
CriticSearch  ↔  RAGEN  环境适配器
=================================
Action 语法（模型输出的纯字符串，区分大小写）：
    SEARCH[query1|query2|...]           → 搜索并返回聚合结果
    BROWSE[https://xxx]                 → 爬取单页正文
    TAKING_NOTES[some text ...]         → 将文本写入 agent.memo，返回 "NOTED"
    START_WRITING[section_title]        → 生成段落并调用 ReportBench 评估；episode 结束

除 START_WRITING 外 reward=0；在 START_WRITING 时以 ReportVerifier
算出的 accuracy ∈[0,1] 作为 reward，并返回 done=True。
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, Tuple, Any

from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.deep_research.config import CriticSearchEnvConfig

# —— CriticSearch 依赖 —— （确保已 pip install -e .）
from criticsearch.base_agent import BaseAgent
from criticsearch.reportbench.report_benchmark import ReportBenchmark
from criticsearch.reportbench.verifier import ReportVerifier
from criticsearch.tools.search_adapter.search_aggregator import SearchAggregator
from criticsearch.utils import extract_citations


# ---------------- Session 封装 ---------------- #
class CriticSearchSession:
    """
    把 CriticSearch 的工具封装成简单 API，供环境 step 调用。
    - search() / browse() / taking_notes() 会把结果写入 agent 状态；
    - start_writing() 生成段落并返回 (content, accuracy)
    """
    def __init__(self, cfg: CriticSearchEnvConfig):
        self.cfg = cfg
        self.agent = BaseAgent()
        self.agent.receive_task(cfg.user_prompt)
        # search/browse 工具
        self.search_agg: SearchAggregator = self.agent.search_aggregator
        self.verifier = ReportVerifier(self.agent)
        # 预加载 ReportBench 数据（供最后评估）
        pkg = "criticsearch.reportbench.wiki_data"
        with (
            __import__("importlib.resources").resources.files(pkg).
            joinpath(cfg.benchmark_file)
        ) as jf:
            self.benchmark = ReportBenchmark(str(jf))
        self.memo: set[str] = set()
        self.section_gt = None  # 在 start_writing 前由外部填充
        self.detailed_web_results: str = ""  # 连续 BROWSE 累积

    # --------- Action API --------- #
    async def search(self, queries: list[str]) -> str:
        res = await self.search_agg.search(queries)
        self.agent.taking_notes(res)
        return res

    async def browse(self, url: str) -> str:
        res = await self.agent.content_scraper.scrape(urls=url)
        self.agent.taking_notes(res)
        self.detailed_web_results += "\n\n" + res
        return res

    def taking_notes(self, note: str) -> str:
        # 直接调用 agent 的接口；实际环境不返回任何网页数据
        _ = self.agent.taking_notes(note)
        return "NOTED"

    def start_writing(self, section_title: str) -> Tuple[str, float]:
        # 这里简化调用：使用你在 pipeline 里的 guided_generation_thought 模板
        draft = self.agent.chat_with_template(
            "guided_generation_thought.txt",
            {
                "task": self.cfg.user_prompt,
                "context": " ".join(self.memo) or "No previous context.",
                "guidline": section_title,
                "search_result": self.detailed_web_results,
                "memo": self.memo,
            },
        )
        # Extract "answer" 部分（你 utils 里已有函数，但为防依赖，这里直接返回全文）
        content = draft
        # 从 GT 中抽取当前 section 的 facts（简单匹配标题）
        if self.section_gt is None:
            # 只生成一次滑窗
            item = self.benchmark.generate_benchmark_item(max_window_tokens=200)[0]
            self.section_gt = item["extracted_facts"]
        accuracy = self.verifier.verify_section(content, self.section_gt)
        return content, accuracy


# -------------- RAGEN 环境 -------------- #
class DeepResearchEnv(BaseLanguageBasedEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: CriticSearchEnvConfig | None = None):
        self.cfg = config or CriticSearchEnvConfig()
        self._session: CriticSearchSession | None = None
        self._trajectory = []
        super().__init__()

    # Gym API: reset ---------------------------------------------------------
    def reset(self, seed=None, **kwargs) -> str:
        self._session = CriticSearchSession(self.cfg)
        self._trajectory.clear()
        return f"INSTRUCTION: {self.cfg.user_prompt}"

    # Gym API: step ----------------------------------------------------------
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        if self._session is None:
            raise RuntimeError("Env.reset() 尚未调用")
        m = re.match(r"(\w+)\[(.*)\]$", action.strip(), flags=re.S)
        if not m:
            return "INVALID ACTION FORMAT", self.cfg.invalid_penalty, False, {}

        act_type, payload = m.group(1).upper(), m.group(2)
        obs, reward, done = "", 0.0, False

        try:
            if act_type == "SEARCH":
                queries = [q.strip() for q in payload.split("|") if q.strip()]
                obs = asyncio.run(self._session.search(queries))

            elif act_type == "BROWSE":
                obs = asyncio.run(self._session.browse(payload.strip()))

            elif act_type == "TAKING_NOTES":
                obs = self._session.taking_notes(payload)

            elif act_type == "START_WRITING":
                content, accuracy = self._session.start_writing(payload.strip())
                obs = f"SECTION DONE. accuracy={accuracy:.4f}"
                reward, done = accuracy, True

            else:
                obs, reward = "UNKNOWN ACTION", self.cfg.invalid_penalty

        except Exception as exc:  # 工具调用错误 -> 小负奖励
            obs = f"ERROR: {exc}"
            reward = self.cfg.invalid_penalty

        # 保存 step 记录，方便 render/debug
        self._trajectory.append(
            {"type": act_type, "payload": payload, "obs": obs[:200], "reward": reward}
        )
        return obs[: self.cfg.max_tokens], reward, done, {}

    # 可选：打印整条轨迹
    def render(self, mode="human"):
        if mode == "human":
            for i, t in enumerate(self._trajectory, 1):
                print(f"{i:02d} | {t}")
        else:
            super().render(mode=mode)