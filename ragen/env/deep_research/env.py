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
        self.episode_id: str | None = None # 新增 Episode ID
        self.current_observation_for_action: str | None = None # 新增，用于存储导致当前action的观察

        # ---------- 提前配置日志记录器 ----------
        self.save_path = Path("log")  # 确保 save_path 是 Path 对象
        self.save_path.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        
        # 修改 logger 名称，使其包含实例 ID 以更好地区分并行环境（如果需要更细致的区分）
        # 但对于写入同一个文件，通常 logger 名称本身不直接影响文件内容，而是 formatter
        logger_name = f"{__name__}.{self.__class__.__name__}.{str(uuid.uuid4())[:8]}" # 为每个env实例提供一个略微独特的logger名
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            log_file_path = self.save_path / "env.log"
            # 使用 'a' 模式确保所有实例都追加到同一个 env.log 文件
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            # 更新 formatter 以包含 episode_id 和 step_count (通过 LogAdapter 或手动添加)
            # 这里我们将在日志消息中手动添加这些信息
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.propagate = False

        self.agent = BaseAgent()
        self.registry = ToolRegistry()
        self.history: list[Dict[str, str]] = [] # 暂时保留 self.history 内部列表，以防基类或其他部分使用
        # self._traj 也不再需要，因为我们直接写入 trajectory.jsonl
        self._step_count: int = 0

        self.instruction_generator = InstructionGenerator()
        self.section_level_samples: list[dict] = (
            self.instruction_generator.get_all_section_level_instructions()
        )
        self._sample_iter = cycle(self.section_level_samples)
        self.current_facts: list[dict] | None = None  

        self.render_cache = None

    def _log_info(self, message: str):
        self.logger.info(f"[Epi: {self.episode_id} | Step: {self._step_count}] {message}")

    def _log_warning(self, message: str):
        self.logger.warning(f"[Epi: {self.episode_id} | Step: {self._step_count}] {message}")

    def _log_error(self, message: str):
        self.logger.error(f"[Epi: {self.episode_id} | Step: {self._step_count}] {message}")

    # ---------- reset ----------
    def reset(self, seed=None, **kw) -> str:
        self.episode_id = str(uuid.uuid4()) # 为新 episode 生成 ID
        self._step_count = 0 # 重置步数计数器
        self.history = [] # 重置内部历史

        set_session(str(uuid.uuid4()))
        sample = next(self._sample_iter)
        user_prompt = "User Query: " + sample["section_full_prompt"]
        self.current_facts = sample["extracted_facts"]

        schemas = []
        for fn in [
            self.agent.search_aggregator.search,
            self.agent.content_scraper.scrape,
            taking_notes,
            retrieve_notes,
        ]:
            schemas.extend(self.registry.get_or_create_tool_schema(fn))

        tpl = Path(self.agent.prompts_dir) / "tool_use_short.txt"
        system_prompt_content = Template(tpl.read_text(encoding="utf-8")).render(
            AVAILABLE_TOOLS=json.dumps(schemas, ensure_ascii=False),
        )

        self.history.append({"role": "system", "content": system_prompt_content})
        self.history.append({"role": "user", "content": user_prompt})
        
        full_prompt = f"{system_prompt_content}\n\n{user_prompt}"
        self.current_observation_for_action = full_prompt # s_0

        self._log_info(f"--- Environment Reset ---")
        log_system_prompt_chars = 20
        if len(system_prompt_content) > 2 * log_system_prompt_chars:
            truncated_system_prompt = (
                f"{system_prompt_content[:log_system_prompt_chars]}..."
                f"{system_prompt_content[-log_system_prompt_chars:]}"
            )
            self._log_info(f"System Prompt (truncated): {truncated_system_prompt}")
        else:
            self._log_info(f"System Prompt: {system_prompt_content}")
        self._log_info(f"User Prompt (Initial Query): {user_prompt}")
        # self._log_info(f"Initial Observation for Model (s_0): {self.current_observation_for_action}")

        self.render_cache = full_prompt
        return full_prompt

    # ---------- step ----------
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        self._step_count += 1
        remarks = "" # 用于 trajectory.jsonl 的备注

        self._log_info(f"--- Model Action (a_{self._step_count-1}) ---")
        self._log_info(action)

        self.history.append({"role": "assistant", "content": action})

        obs: Any = ""
        reward: float = 0.0
        done: bool = False
        info: Dict = {} 

        if self._step_count > self.cfg.max_steps:
            msg = "<error>max_steps_exceeded</error>"
            remarks = "Max steps exceeded"
            self.render_cache = msg
            self.history.append({"role": "user", "content": msg})
            # 不再写入 history.log
            self._log_warning(f"Max steps exceeded. Action: {action}, Msg: {msg}")
            obs, reward, done, info = msg, self.cfg.invalid_penalty, True, {}
        else:
            tool_xml = extract_tag_content(action, "tool_use")
            if not tool_xml:
                answer_content = extract_tag_content(action, "answer")
                if not answer_content:
                    msg = "<error>format_error_no_tool_or_answer</error>"
                    remarks = "Format error: No tool use or answer tag detected"
                    self.history.append({"role": "user", "content": msg})
                    # 不再写入 history.log
                    self._log_warning(f"No tool use or answer tag detected. Action: {action}")
                    obs, reward, done, info = msg, self.cfg.format_penalty, True, {}
                else:
                    remarks = "Final answer provided by model"
                    verifier = ReportVerifier(self.agent)
                    acc = verifier.verify_section(action, self.current_facts)
                    self.render_cache = action
                    self._log_info(f"--- Model Section Answer ---")
                    self._log_info(action)
                    self._log_info(f"--- Accuracy ---")
                    self._log_info(str(acc))
                    obs, reward, done, info = action, acc, True, {}
            else:
                self._log_info(f"--- Detected Model Tool Use ---")
                self._log_info(tool_xml)
                tool_name = extract_tag_content(tool_xml, "name")
                arg_str = extract_tag_content(tool_xml, "arguments") or "{}"
                remarks = f"Tool call attempt: {tool_name}"
                try:
                    args = json.loads(arg_str)
                except json.JSONDecodeError:
                    error_xml = (
                        f"<tool_use_result><name>{tool_name}</name>"
                        f"<error>arguments_not_json</error></tool_use_result>"
                    )
                    remarks = f"Tool call error: arguments_not_json for {tool_name}"
                    self.render_cache = error_xml
                    self.history.append({"role": "user", "content": error_xml})
                    # 不再写入 history.log
                    self._log_error(f"Failed to parse tool arguments: {arg_str}. Error XML: {error_xml}")
                    obs, reward, done, info = error_xml, self.cfg.invalid_penalty, False, {}
                else:
                    try:
                        result = self.registry.invoke_tool(tool_name, args)
                        result_xml = (
                            f"<tool_use_result><name>{tool_name}</name>"
                            f"<result>{json.dumps(result, ensure_ascii=False)}</result>"
                            f"</tool_use_result>"
                        )
                        print("tool_xml:\n", result_xml)
                        print("end of tool xml")

                        remarks = f"Tool call success: {tool_name}"
                        self.render_cache = result_xml
                        self._log_info(f"--- Tool Use Result ---")
                        self._log_info(result_xml)
                        self.history.append({"role": "user", "content": result_xml})
                        # 不再写入 history.log
                        obs, reward, done, info = result, 1.0, False, {}
                    except Exception as exc:
                        error_xml = (
                            f"<tool_use_result><name>{tool_name}</name>"
                            f"<error>{str(exc)}</error></tool_use_result>"
                        )
                        remarks = f"Tool call exception: {tool_name} - {str(exc)}"
                        self.history.append({"role": "user", "content": error_xml})
                        # 不再写入 history.log
                        self._log_error(f"Error invoking tool {tool_name} with args {args}. Exception: {exc}. Error XML: {error_xml}")
                        self.render_cache = error_xml
                        obs, reward, done, info = error_xml, self.cfg.invalid_penalty, False, {}
        
        obs_str = json.dumps(obs, ensure_ascii=False) if isinstance(obs, (dict, list)) else str(obs)
        final_obs_str = obs_str[: self.cfg.max_tokens]

        self._log_info(f"--- Environment Feedback ---")
        self._log_info(f"Reward (r_{self._step_count}): {reward}")
        self._log_info(f"Done: {done}")
        self._log_info(f"Next Observation (s_{self._step_count}) before truncation: {obs_str}")
        if len(obs_str) > self.cfg.max_tokens:
            self._log_info(f"Next Observation (s_{self._step_count}) (truncated): {final_obs_str}")


        # 记录轨迹到 trajectory.jsonl
        current_traj_entry = {
            "episode_id": self.episode_id,
            "step": self._step_count,
            "observation_for_action": self.current_observation_for_action, # s_{t-1}
            "action_taken": action, # a_{t-1}
            "reward_received": reward, # r_t
            "next_observation": final_obs_str, # s_t
            "done": done,
            "remarks": remarks
        }
        # 使用 "trajectory.jsonl" 并以 "a" 模式打开
        with open(self.save_path / "trajectory.jsonl", "a", encoding="utf-8") as f: 
            f.write(json.dumps(current_traj_entry, ensure_ascii=False) + "\n")

        # 更新下一个步骤的 "当前观察"
        self.current_observation_for_action = final_obs_str
        
        return final_obs_str, reward, done, info

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



"""
Use example:

python train.py --config-name base > train.log 2>&1

"""