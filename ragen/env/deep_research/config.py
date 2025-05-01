# ragen/env/deep_research/config.py
from dataclasses import dataclass
from ragen.env.base import BaseEnvConfig

@dataclass
class CriticSearchEnvConfig(BaseEnvConfig):
    """
    仅放和业务强相关的超参。其余诸如 batch_size、ppo 超参
    继续写在 RAGEN 自带的 yaml 里即可。
    """
    # 给 CriticSearch 的初始指令模板
    user_prompt: str = (
        "Write a detailed article about the {topic} using structured JSON formatting, "
        "including preface, background, electoral system, timeline, voter registration, "
        "parties and candidates, election results, and aftermath."
    )

    # 非法动作惩罚
    invalid_penalty: float = -0.2

    # observation 序列的截断长度，供 ctx_manager 参考
    max_tokens: int = 2048

    # 是否在每条 trajectory 结束后自动重启外部 Session
    auto_reset_session: bool = True
    benchmark_file: str = "2024_Syrian_opposition_offensives.json"