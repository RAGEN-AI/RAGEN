# ragen/env/deep_research/env.py

from __future__ import annotations
from itertools import cycle
import json, uuid
from pathlib import Path
from typing import Dict, Any, Tuple
import logging  # 新增导入

from jinja2 import Template
from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.deep_research.config import DeepResearchEnvConfig

from criticsearch.base_agent import BaseAgent
from criticsearch.tools.tool_registry import ToolRegistry
from criticsearch.tools.note_manager import set_session, taking_notes, retrieve_notes
from criticsearch.utils import extract_tag_content
from criticsearch.reportbench.instruction_generator import InstructionGenerator
from criticsearch.reportbench.verifier import ReportVerifier


class DeepResearchEnv(BaseLanguageBasedEnv):
    metadata = {"render.modes": ["human"]}

    # ---------- init ----------
    def __init__(self, config = DeepResearchEnvConfig()):
        super().__init__()
        self.cfg = config or DeepResearchEnvConfig()
        self.config = self.cfg

        # ---------- 提前配置日志记录器 ----------
        self.save_path = Path("/map-vepfs/qinyu/CodeSpace/RAGEN")  # 确保 save_path 是 Path 对象
        self.save_path.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
        # 防止重复添加 handler
        if not self.logger.handlers:
            log_file_path = self.save_path / "env.log"
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.propagate = False  # 防止将日志消息传递给根记录器的处理程序

        self.agent = BaseAgent()
        self.registry = ToolRegistry()
        self.history: list[Dict[str, str]] = []
        self._traj: list[Dict[str, Any]] = []
        self._step_count: int = 0

        # ---------- ① 生成样本列表 ----------
        self.instruction_generator = InstructionGenerator()
        self.section_level_samples: list[dict] = (
            self.instruction_generator.get_all_section_level_instructions()
        )
        self._sample_iter = cycle(self.section_level_samples)
        self.current_facts: list[dict] | None = None  

        self.render_cache = None

    # ---------- reset ----------
    def reset(self, seed=None, **kw) -> str:
        """
        环境的 initial observation. 模型在每条轨迹（episode）开始时看到的内容
        """
        # 为笔记工具新建 session
        set_session(str(uuid.uuid4()))
        # 取一个新样本
        sample = next(self._sample_iter)
        user_prompt = "User Query: " + sample["section_full_prompt"]
        self.current_facts = sample["extracted_facts"]  # 存起来给 step 用

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
            {"role": "user", "content": user_prompt},
        ]

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # print(f"--- Full Prompt ---")
        # print(full_prompt)
        # print(type(full_prompt))
        # if type(full_prompt) != str:
        #     assert "full prompt is not a str"

        self.render_cache = full_prompt

        return full_prompt

    # ---------- step ----------
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        
        # action = '<tool_use>\n  <name>search</name>\n  <arguments>{\"query\": [\"最新俄乌战争新闻\", \"Russia Ukraine war news latest\"]}</arguments>\n</tool_use>'

        self.logger.info(f"--- Model Action ---")
        self.logger.info(action)
        print(f"--- Model Action ---")
        print(action)
        self._step_count += 1
        self.history.append({"role": "assistant", "content": action})

        # 超步数直接终止
        if self._step_count > self.cfg.max_steps:
            msg = "<error>max_steps_exceeded</error>"
            self.render_cache = msg
            self.history.append({"role": "user", "content": msg})
            self.logger.warning(f"Max steps exceeded. Action: {action}, Msg: {msg}")
            return msg, self.cfg.invalid_penalty, True, {}

        tool_xml = extract_tag_content(action, "tool_use")
        
        if not tool_xml:
            # 获取answer tag内的模型最终回答
            answer_content = extract_tag_content(action, "answer")
            if not answer_content:
                self.logger.warning("No tool use or answer tag detected.")
                # 如果既不是工具调用，也没有检测到 answer tag，则给予格式惩罚
                msg = "<error>format_error_no_tool_or_answer</error>"
                self.history.append({"role": "user", "content": msg})
                return msg, self.cfg.format_penalty, True, {}
            else:
                # 检测到 answer tag，按原逻辑处理
                verifier = ReportVerifier(self.agent)
                acc = verifier.verify_section(action, self.current_facts)
                self.render_cache = action
                self.logger.info(f"--- Model Section Answer ---")
                self.logger.info(action)
                self.logger.info(f"--- Accuracy ---")
                self.logger.info(str(acc))
                # 最终回答
                obs, reward, done = action, acc, True 
        else:
            # -------- 解析工具调用 --------
            self.logger.info(f"--- Detected Model Tool Use ---")
            self.logger.info(tool_xml)
            tool_name = extract_tag_content(tool_xml, "name")
            arg_str = extract_tag_content(tool_xml, "arguments") or "{}"
            try:
                args = json.loads(arg_str)
            except json.JSONDecodeError:
                error_xml = (
                    f"<tool_use_result><name>{tool_name}</name>"
                    f"<error>arguments_not_json</error></tool_use_result>"
                )
                self.render_cache = error_xml
                self.history.append({"role": "user", "content": error_xml})
                self.logger.error(f"Failed to parse tool arguments: {arg_str}. Error XML: {error_xml}")
                return error_xml, self.cfg.invalid_penalty, False, {}

            # -------- 执行工具 --------
            try:
                result = self.registry.invoke_tool(tool_name, args)
                result_xml = (
                    f"<tool_use_result><name>{tool_name}</name>"
                    f"<result>{json.dumps(result, ensure_ascii=False)}</result>"
                    f"</tool_use_result>"
                )
                self.render_cache = result_xml
                self.logger.info(f"--- Tool Use Result ---")
                self.logger.info(result_xml)

                self.history.append({"role": "user", "content": result_xml})
                obs, reward, done = result, 0.0, False
            except Exception as exc:
                error_xml = (
                    f"<tool_use_result><name>{tool_name}</name>"
                    f"<error>{str(exc)}</error></tool_use_result>"
                )
                self.history.append({"role": "user", "content": error_xml})
                self.logger.error(f"Error invoking tool {tool_name} with args {args}. Exception: {exc}. Error XML: {error_xml}")
                self.render_cache = error_xml
                return error_xml, self.cfg.invalid_penalty, False, {}

        # 记录轨迹
        self._traj.append({"a": action, "r": reward})
        with open(self.save_path / "error.log", "a+", encoding="utf-8") as f:
            f.write(f"{self._traj}\n")

        # 记录对话历史
        with open(self.save_path / "history.log", "a+", encoding="utf-8") as f:
            f.write(f"{self.history}\n")

        # 统一转字符串并截断
        obs_str = json.dumps(obs, ensure_ascii=False) if isinstance(obs, (dict, list)) else str(obs)
        return obs_str[: self.cfg.max_tokens], reward, done, {}

    # ---------- render ----------
    def render(self, mode="human") -> str:
        """
        打印并返回轨迹（action, reward）信息的文本表示。
        mode="human" 时会在控制台打印，其他模式只返回字符串。
        """
        # if mode == "human":
        #     # print(self.render_cache)
        #     print("hi")
        return self.render_cache