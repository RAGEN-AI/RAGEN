# ragen/env/deep_research/config.py
from dataclasses import dataclass
from ragen.env.base import BaseEnvConfig


@dataclass
class CriticSearchEnvConfig(BaseEnvConfig):
    """
    环境超参（可在 config/envs.yaml 覆写）
    """
    user_prompt: str = (
        "请你从网上搜索黄金最新的新闻并且记一下笔记，然后检索笔记验证。"
    )
    max_tokens: int = 32000          # observation 截断
    invalid_penalty: float = -0.2   # JSON 解析 / 工具调用失败惩罚
    max_steps: int = 30             # 轨迹最大步数，防止死循环