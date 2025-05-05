# ragen/env/deep_research/env.py

from __future__ import annotations
import json, uuid
from pathlib import Path
from typing import Dict, Any, Tuple

from jinja2 import Template
from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.deep_research.config import CriticSearchEnvConfig

from criticsearch.base_agent import BaseAgent
from criticsearch.tools.tool_registry import ToolRegistry
from criticsearch.tools.note_manager import set_session, taking_notes, retrieve_notes
from criticsearch.utils import extract_tag_content


class DeepResearchEnv(BaseLanguageBasedEnv):
    metadata = {"render.modes": ["human"]}

    # ---------- init ----------
    def __init__(self, config: CriticSearchEnvConfig | None = None):
        super().__init__()
        self.cfg = config or CriticSearchEnvConfig()
        self.agent: BaseAgent | None = None
        self.registry: ToolRegistry | None = None
        self.history: list[Dict[str, str]] = []
        self._traj: list[Dict[str, Any]] = []
        self._step_count: int = 0

    # ---------- reset ----------
    def reset(self, seed=None, **kw) -> str:
        self.agent = BaseAgent()
        self.registry = self.agent.tool_registry
        self._step_count = 0

        # 为笔记工具新建 session
        set_session(str(uuid.uuid4()))

        schemas = []
        # 注册 4 个工具
        for fn in [
            self.agent.search_aggregator.search,
            self.agent.content_scraper.scrape,
            taking_notes,
            retrieve_notes,
        ]:
            schemas.extend(self.registry.get_or_create_tool_schema(fn))

        # 渲染 system prompt
        tpl = Path(self.agent.prompts_dir) / "tool_use.txt"
        system_prompt = Template(tpl.read_text(encoding="utf-8")).render(
            AVAILABLE_TOOLS=json.dumps(schemas, ensure_ascii=False),
        )

        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.cfg.user_prompt},
        ]
        self._traj.clear()

        return system_prompt + "\n\n" + self.cfg.user_prompt

    # ---------- step ----------
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        self._step_count += 1
        self.history.append({"role": "assistant", "content": action})

        # 超步数直接终止
        if self._step_count > self.cfg.max_steps:
            msg = "<error>max_steps_exceeded</error>"
            self.history.append({"role": "user", "content": msg})
            return msg, self.cfg.invalid_penalty, True, {}

        tool_xml = extract_tag_content(action, "tool_use")
        if not tool_xml:
            # 最终回答
            obs, reward, done = action, 1.0, True
        else:
            # -------- 解析工具调用 --------
            tool_name = extract_tag_content(tool_xml, "name")
            arg_str = extract_tag_content(tool_xml, "arguments") or "{}"
            try:
                args = json.loads(arg_str)
            except json.JSONDecodeError:
                error_xml = (
                    f"<tool_use_result><name>{tool_name}</name>"
                    f"<error>arguments_not_json</error></tool_use_result>"
                )
                self.history.append({"role": "user", "content": error_xml})
                return error_xml, self.cfg.invalid_penalty, False, {}

            # -------- 执行工具 --------
            try:
                result = self.registry.invoke_tool(tool_name, args)
                result_xml = (
                    f"<tool_use_result><name>{tool_name}</name>"
                    f"<result>{json.dumps(result, ensure_ascii=False)}</result>"
                    f"</tool_use_result>"
                )
                self.history.append({"role": "user", "content": result_xml})
                obs, reward, done = result, 0.0, False
            except Exception as exc:
                error_xml = (
                    f"<tool_use_result><name>{tool_name}</name>"
                    f"<error>{str(exc)}</error></tool_use_result>"
                )
                self.history.append({"role": "user", "content": error_xml})
                return error_xml, self.cfg.invalid_penalty, False, {}

        # 记录轨迹
        self._traj.append({"a": action, "r": reward})
        # 统一转字符串并截断
        obs_str = json.dumps(obs, ensure_ascii=False) if isinstance(obs, (dict, list)) else str(obs)
        return obs_str[: self.cfg.max_tokens], reward, done, {}

    # ---------- render ----------
    def render(self, mode="human"):
        if mode == "human":
            for i, t in enumerate(self._traj, 1):
                print(f"{i:02d}| r={t['r']} | {t['a']}")
        else:
            super().render(mode)