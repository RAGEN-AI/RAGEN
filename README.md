<h1 align="center">RAGEN: Training Agents by Reinforcing Reasoning</h1>
<h3 align="center"><em>Diagnose agent failure modes. Make your RL training better.</em></h3>

<p align="center"><img src="public/ragen_logo.jpeg" width="300px" alt="RAGEN icon" /></p>

<p align="center">
  <strong>RAGEN</strong> (<b>R</b>easoning <b>AGEN</b>T) is a flexible RL framework for training reasoning <strong>LLM agents</strong> with <strong>multi-turn reinforcement learning</strong>, built on the <strong>StarPO</strong> algorithm.
</p>
<p align="center">
  We develop <strong>diagnostics to understand <i>how</i> agentic reinforcement learning (agent RL) training works</strong>, and how to fix hidden failure modes such as the <strong>Echo Trap</strong> (RAGEN) and <strong>reasoning collapse</strong> (RAGEN-2).
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.06268"><img src="https://img.shields.io/badge/📄_V2_Paper-DC143C?style=for-the-badge&logoColor=white" alt="V2 Paper"></a>
  <a href="https://arxiv.org/abs/2504.20073"><img src="https://img.shields.io/badge/📄_v1_Paper-FF8C00?style=for-the-badge&logoColor=white" alt="v1 Paper"></a>
  <a href="https://ragen-ai.github.io/"><img src="https://img.shields.io/badge/📝_HomePage-FF5722?style=for-the-badge&logoColor=white" alt="Blog"></a>
  <!-- <a href="https://ragen-doc.readthedocs.io/"><img src="https://img.shields.io/badge/📚_Documentation-4285F4?style=for-the-badge&logoColor=white" alt="Documentation"></a> -->
  <a href="https://x.com/wzihanw/status/1915052871474712858"><img src="https://img.shields.io/badge/🔍_Post-34A853?style=for-the-badge&logoColor=white" alt="Post"></a>
  <a href="https://api.wandb.ai/links/zihanwang-ai-northwestern-university/a8er8l7b"><img src="https://img.shields.io/badge/🧪_Experiment_Log-AB47BC?style=for-the-badge&logoColor=white" alt="Experiment Log"></a>
</p>

<p align="center">
  <a href="https://ragen-ai.github.io/">Project Page</a> ·
  <a href="https://ragen-ai.github.io/v2/">RAGEN-2 Page</a> ·
  <a href="https://arxiv.org/abs/2504.20073">RAGEN on arXiv</a> ·
  <a href="https://arxiv.org/abs/2604.06268">RAGEN-2 on arXiv</a> ·
  <a href="https://limanling.github.io/">MLL Lab (Manling Li, Northwestern)</a>
</p>

> **Looking for the V1 README?** Please take a look [here](docs/readme_v1.md).

## News

- **2026.3.12.** We are excited to release <font color="#DC143C">RAGEN-2</font>! We introduce a systematic study of reasoning collapse in agent RL and lightweight interventions for stable training. See the [<font color="#DC143C">v2 paper</font>](https://ragen-ai.github.io/v2).
- **2025.4.20.** RAGEN V1 [paper](https://arxiv.org/abs/2504.20073) published on arXiv.
- **2025.1.27.** Initial RAGEN release. [Post](https://x.com/wzihanw/status/1884092805598826609).


## About

RAGEN is built around **StarPO** (**S**tate-**T**hinking-**A**ctions-**R**eward **P**olicy **O**ptimization), a unified RL framework for training multi-turn, trajectory-level agents with flexible control over reasoning processes, reward assignment mechanisms, and prompt-rollout structures.

**RAGEN is flexible with:**

- **StarPO framework.** Unified optimization for multi-turn agents, supporting both trajectory-level and turn-wise training.
- **10 built-in environments.** Sokoban, FrozenLake, WebShop, DeepCoder, SearchQA, Lean, Bandit, Countdown, MetaMathQA, Sudoku.
- **Gym-compatible interface.** Easy to add custom environments.

**<font color="#DC143C">RAGEN-2</font> additionally introduces:**

- **SNR-Adaptive Filtering (<font color="#DC143C">V2</font>).** Lightweight rollout filtering based on reward variance to mitigate noisy gradient updates.
- **Reasoning collapse diagnostics (<font color="#DC143C">V2</font>).** Mutual information proxy metrics to detect and monitor template collapse during training.


## Algorithm

### StarPO: Reinforcing Reasoning via Trajectory-Level Optimization

<p align="center"><img src="public/starpo_logo.png" width="800px" alt="StarPO Framework" /></p>
<p align="center" style="font-size: 16px; max-width: 800px; margin: 0 auto;">
The StarPO (State-Thinking-Action-Reward Policy Optimization) framework with two interleaved stages: <b>rollout stage</b> and <b>update stage</b>. The LLM generates reasoning-guided actions to interact with the environment, collecting trajectory-level rewards to jointly optimize reasoning and action strategies.
</p>

**MDP Formulation.** Agent-environment interactions are formulated as Markov Decision Processes (MDPs) where states and actions are token sequences, allowing LLMs to reason over environment dynamics. The objective is to maximize expected cumulative rewards across multiple interaction turns.

**Rollout Stage.** Given an initial state, the LLM generates multiple trajectories. At each step, the model produces a reasoning-guided action: `<think>...</think><ans> action </ans>`. The environment returns feedback (reward and next state).

**Update Stage.** StarPO optimizes entire trajectories using importance sampling. It supports:
- **PPO.** Token-level advantage estimation via a value function over trajectories.
- **GRPO.** Normalized reward assigned to the full trajectory.

### <font color="#DC143C">V2</font>: Diagnosing Template Collapse

Entropy alone cannot detect *template collapse*, where reasoning appears diverse within a single input but becomes input-agnostic across inputs. <font color="#DC143C">RAGEN-2</font> decomposes reasoning quality into two axes:
- **Within-input diversity:** Conditional Entropy H(Z|X)
- **Cross-input distinguishability:** Mutual Information I(X;Z)

SNR-Adaptive Filtering uses reward variance as a lightweight proxy to select high-signal prompts each iteration, directly addressing the root cause of template collapse.


## Update Log

**2026.3.12.** <font color="#DC143C">RAGEN-2</font> is released! Check out our [<font color="#DC143C">v2 paper</font>](https://ragen-ai.github.io/v2).

<details>
<summary>Older updates</summary>

**2025.5.8.** Official [Documentation](https://ragen-doc.readthedocs.io/) released. NOTE: this document is now outdated.

**2025.5.2.** A [tracking document](https://docs.google.com/document/d/1bg7obeiKTExuHHBl5uOiSpec5uLDZ2Tgvxy6li5pHX4/edit?usp=sharing) for logging minor codebase updates is released.

**2025.4.20.** RAGEN V1 [paper](https://arxiv.org/abs/2504.20073) published. Codebase restructured: veRL integrated as a submodule; architecture decomposed into three modules — Environment State Manager, Context Manager, and Agent Proxy.

**2025.3.13.** RAGEN codebase refactoring underway. See the [developing branch](https://github.com/ZihanWang314/RAGEN/tree/main-new).

**2025.3.8.** KL term issue in veRL [fixed](https://github.com/volcengine/verl/pull/179/files). Default advantage estimator changed to GAE (PPO) for more stable training.

**2025.1.27.** Initial RAGEN release. [Post](https://x.com/wzihanw/status/1884092805598826609).

</details>


## Getting Started

```bash
git clone https://github.com/mll-lab-nu/RAGEN.git
cd RAGEN
conda create -n ragen python=3.12 -y && conda activate ragen
bash scripts/setup_ragen.sh
```

Use `bash scripts/setup_ragen.sh --with-search` to include the search environment. For WebShop, see [docs/experiment_webshop_release.md](docs/experiment_webshop_release.md).

### The Four Reasoning Regimes

<font color="#DC143C">RAGEN-2</font> diagnoses agent behavior along two axes — **within-input diversity** (Conditional Entropy) and **cross-input distinguishability** (Mutual Information) — yielding four distinct reasoning regimes:

<p align="center"><img src="public/teaser.png" width="800px" alt="Four reasoning regimes: diverse reasoning, template collapse, compressed reasoning, low-entropy collapse" /></p>
<p align="center" style="font-size: 15px; max-width: 800px; margin: 0 auto;">
<b>Left:</b> Input-driven reasoning adapts to the current state; templated reasoning produces nearly identical responses across different inputs. <b>Right:</b> Four reasoning regimes along two axes — conditional entropy H(Z|X) (within-input diversity) and mutual information I(X;Z) (input dependence). Template collapse (high entropy, low MI) is invisible to existing entropy-based metrics.
</p>

**Train (no filter, default):**
```bash
python train.py --config-name _2_sokoban
```

**Train with SNR-Adaptive Filtering (<font color="#DC143C">V2</font>, Top-p):**
```bash
python train.py --config-name _2_sokoban \
  actor_rollout_ref.rollout_filter_strategy=top_p \
  actor_rollout_ref.rollout.rollout_filter_value=0.9
```

**Evaluate:**
```bash
python -m ragen.llm_agent.agent_proxy --config-name _2_sokoban
```

SNR-Adaptive Filtering consistently improves training across algorithms, model scales, and modalities (green = gain from filtering):

<p align="center"><img src="public/main_results.png" width="800px" alt="Main results: filtering vs no filtering" /></p>

See the [Rollout Filtering Guide](docs/guide_rollout_filtering.md) for more filtering strategies (Top-k, linear mode, etc.).


## Future Plans

We are actively developing the next generation of RAGEN infrastructure and diagnostics, targeting a release in **late March 2026**.

**Infrastructure**
- [ ] **Async rollout engine** 
- [ ] **HTTP-based environment interface** 
- [ ] **Layered Env Wrapper** 
- [ ] **Optional environment dependencies** 

**Diagnostics & Training Quality**
- [ ] **Expanded benchmark suite** to stress-test diagnostics across diverse, real-world agent tasks
- [ ] **Extended MI diagnostic dashboard**, including richer WandB visualizations for entropy, MI proxy, and gradient decomposition over training
- [ ] **RL training metrics guide**, including a practitioner's blog on how to read training signals (reward distribution, entropy, MI, gradient norms) and act on them before committing to a full run

**Framework**
- [ ] Update full documentation for <font color="#DC143C">RAGEN-2</font>
- [ ] Multi-modal agent support (building upon [VAGEN](https://github.com/RAGEN-AI/VAGEN))
- [ ] Public leaderboard for benchmark results


## Documentation

- [Full Documentation](https://ragen-doc.readthedocs.io/) *(We will release an updated version soon.)*
- [Evaluation Guide](docs/eval.md) — How to evaluate models and configure output formats
- [Rollout Filtering Guide](docs/guide_rollout_filtering.md)
- [MI Metrics Reference](docs/reference_mutual_information_metrics.md)
- Adding Custom Environments — Gym-compatible interface, see `config/envs.yaml`
- Experiment reproduction: [Main Table](docs/experiment_main_table.md) | [Intervention Sweep](docs/experiment_intervention_sweep.md) | [FrozenLake](docs/experiment_frozen_lake_slipper_sweep.md) | [Sokoban Gradient](docs/experiment_sokoban_gradient_analysis.md) | [Search](docs/experiment_search.md) | [DeepCoder](docs/experiment_deepcoder.md) | [WebShop](docs/experiment_webshop_release.md)


## Awesome Work Powered or Inspired by RAGEN

- [ROLL](https://github.com/alibaba/ROLL): Efficient Scaling Library for RL with LLMs ![GitHub Repo stars](https://img.shields.io/github/stars/alibaba/ROLL?style=social)
- [VAGEN](https://github.com/RAGEN-AI/VAGEN): Training Visual Agents with multi-turn RL ![GitHub Repo stars](https://img.shields.io/github/stars/RAGEN-AI/VAGEN?style=social)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1): Train LLMs to reason and call a search engine with RL ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1?style=social)
- [ZeroSearch](https://github.com/Alibaba-nlp/ZeroSearch): Incentivize LLM search capability without searching ![GitHub Repo stars](https://img.shields.io/github/stars/Alibaba-nlp/ZeroSearch?style=social)
- [Agent-R1](https://github.com/AgentR1/Agent-R1): Training Powerful LLM Agents with End-to-End RL
- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): RL tuning for LLM agents ![GitHub Repo stars](https://img.shields.io/github/stars/OpenManus/OpenManus-RL?style=social)
- [MetaSpatial](https://github.com/PzySeere/MetaSpatial): Reinforcing 3D Spatial Reasoning in VLMs ![GitHub Repo stars](https://img.shields.io/github/stars/PzySeere/MetaSpatial?style=social)
- [s3](https://github.com/pat-jj/s3): Efficient Yet Effective Search Agent Training via RL


## Contributors

[**Zihan Wang**\*](https://zihanwang314.github.io/), [**Kangrui Wang**\*](https://jameskrw.github.io/), [**Qineng Wang**\*](https://qinengwang-aiden.github.io/), [**Pingyue Zhang**\*](https://williamzhangsjtu.github.io/), [**Linjie Li**\*](https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en), [**Zhengyuan Yang**](https://zyang-ur.github.io/), [**Xing Jin**](https://openreview.net/profile?id=~Xing_Jin3), [**Kefan Yu**](https://www.linkedin.com/in/kefan-yu-22723a25b/en/), [**Minh Nhat Nguyen**](https://www.linkedin.com/in/menhguin/?originalSubdomain=sg), [**Licheng Liu**](https://x.com/liulicheng10), [**Eli Gottlieb**](https://www.linkedin.com/in/eli-gottlieb1/), [**Yiping Lu**](https://2prime.github.io), [**Kyunghyun Cho**](https://kyunghyuncho.me/), [**Jiajun Wu**](https://jiajunwu.com/), [**Li Fei-Fei**](https://profiles.stanford.edu/fei-fei-li), [**Lijuan Wang**](https://www.microsoft.com/en-us/research/people/lijuanw/), [**Yejin Choi**](https://homes.cs.washington.edu/~yejin/), [**Manling Li**](https://limanling.github.io/)

\*Equal Contribution.


## Acknowledgements

We thank the [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1) team for early conceptual inspirations. We are grateful to the [veRL](https://github.com/volcengine/verl) team for infrastructure support. We thank the [TinyZero](https://github.com/Jiayi-Pan/TinyZero) team for discoveries that informed our initial exploration. We appreciate insightful discussions with Han Liu, Xinyu Xing, Li Erran Li, John Schulman, Akari Asai, Eiso Kant, Lu Lu, Runxin Xu, Huajian Xin, Zijun Liu, Weiyi Liu, Weimin Wu, Yibo Wen, Jiarui Liu, Lorenzo Xiao, Ishan Mukherjee, Anabella Isaro, Haosen Sun, How-Yeh Wan, Lester Xue, Matthew Khoriaty, Haoxiang Sun, Jiajun Liu.

For <font color="#DC143C">RAGEN-2</font>, we additionally thank Yuxiang Lin and Kyunghyun Cho for their support.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mll-lab-nu/ragen&type=Date)](https://www.star-history.com/#mll-lab-nu/ragen&Date)


## Citation

```bibtex
@misc{ragen2,
      title={RAGEN-2: Reasoning Collapse in Agentic RL}, 
      author={Zihan Wang and Chi Gui and Xing Jin and Qineng Wang and Licheng Liu and Kangrui Wang and Shiqi Chen and Linjie Li and Zhengyuan Yang and Pingyue Zhang and Yiping Lu and Jiajun Wu and Li Fei-Fei and Lijuan Wang and Yejin Choi and Manling Li},
      year={2026},
      eprint={2604.06268},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.06268}, 
}
```

```bibtex
@misc{ragen,
      title={RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning},
      author={Zihan Wang and Kangrui Wang and Qineng Wang and Pingyue Zhang and Linjie Li and Zhengyuan Yang and Xing Jin and Kefan Yu and Minh Nhat Nguyen and Licheng Liu and Eli Gottlieb and Yiping Lu and Kyunghyun Cho and Jiajun Wu and Li Fei-Fei and Lijuan Wang and Yejin Choi and Manling Li},
      year={2025},
      eprint={2504.20073},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.20073},
}
```
