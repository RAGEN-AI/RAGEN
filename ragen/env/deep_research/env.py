# ragen/env/deep_research/env.py

"""
CriticSearch  ↔  RAGEN  环境适配器
---------------------------------
动作集合：
    SEARCH[q ...]          → external.search(...)
    BROWSE[url]            → external.browse(...)
    TAKING_NOTES[text ...] → external.take_notes(...)
    START_WRITING[section] → external.start_writing(...); 结束一次轨迹

只有 START_WRITING 步会触发 ReportBench 评估并给 reward，其余 reward 均为 0。
"""

from __future__ import annotations
import re
from typing import Tuple, Dict, Any
from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.deep_research.config import CriticSearchEnvConfig

# 新增：引入Criticsearch向外部暴露的接口 Session 供所有RL框架适配
from criticsearch.session import Session

class DeepResearchEnv(BaseLanguageBasedEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config: CriticSearchEnvConfig | None = None):
        self.cfg = config or CriticSearchEnvConfig()
        self._session: Session | None = None
        self._traj: list[dict[str, Any]] = []
        super().__init__()

    def reset(self, seed=None, **kwargs) -> str:
        self._session = Session(prompt=self.cfg.user_prompt)
        self._traj.clear()
        return f"INSTRUCTION: {self.cfg.user_prompt}"

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        if self._session is None:
            raise RuntimeError("请先调用 reset()")

        m = re.match(r"(\w+)\[(.*)\]$", action.strip(), flags=re.S)
        if not m:
            return "INVALID ACTION FORMAT", self.cfg.invalid_penalty, False, {}

        act_type, payload = m.group(1).upper(), m.group(2).split("|")
        # 注意：payload 对于 BROWSE/SEARCH 可以是列表
        if act_type == "SEARCH":
            obs = self._session.search(payload)
            reward, done = 0.0, False

        elif act_type == "BROWSE":
            obs = self._session.browse(payload)
            reward, done = 0.0, False

        elif act_type == "TAKING_NOTES":
            # payload.join 可根据你传入的格式调整
            text = "|".join(payload)
            obs = self._session.take_notes(text)
            reward, done = 0.0, False

        elif act_type == "START_WRITING":
            # 这里只取第一个作为 section
            section = payload[0]
            draft = self._session.start_writing(section)
            obs = f"SECTION WRITTEN. ReportBench accuracy={self._session.last_score:.4f}"
            reward, done = self._session.last_score, True

        else:
            obs, reward, done = "UNKNOWN ACTION", self.cfg.invalid_penalty, False

        self._traj.append({
            "type": act_type, "payload": payload, "obs": obs,
            "reward": reward, "done": done
        })
        return obs, reward, done, {}

    def render(self, mode="human"):
        if mode == "human":
            for t in self._traj:
                print(t)
        else:
            super().render(mode)