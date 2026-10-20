# Search Environment (HotpotQA + Dense Retrieval)

A multi-turn search environment for training LLM agents on multi-hop question answering. The agent interacts with a retrieval server to search Wikipedia and answer questions from HotpotQA.

## Overview

The agent receives a question and can take two types of actions:
- `search[query]` — retrieve relevant Wikipedia passages via dense retrieval (E5 + FAISS)
- `finish[answer]` — submit a final answer

Reward is computed using F1 / exact match against the HotpotQA ground truth.

## Components

| Component | Description |
|-----------|-------------|
| `env.py` | Gym environment: parses actions, calls retrieval server, computes reward |
| `config.py` | Dataclass config (`SearchEnvConfig`) |
| `reward.py` | F1 / EM reward computation |
| `retrieval_client.py` | HTTP client for the retrieval server |
| `parallel_client.py` | Optional anonymous Parallel Search MCP client |
| `scripts/retrieval/server.py` | Flask server: E5 encoder + FAISS index over Wikipedia |

## Optional live-web retrieval

For exploration or small evaluations, [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
provides free web search in Fast mode and page fetching, without an account or API
key. Install the search extra in your RAGEN environment:

```bash
pip install -e '.[search]'
```

Set `retrieval_backend="parallel"` when constructing the environment:

```python
from ragen.env.search import SearchEnv, SearchEnvConfig

env = SearchEnv(SearchEnvConfig(
    train_path="data/search/val.parquet",
    retrieval_backend="parallel",
))
print(env.reset(seed=0))
print(env.step("search[capital of France]")[0])
env.close()
```

The same field is available in the environment manager's `env_config`; with the
existing `_9_search` configuration, the Hydra override is
`+custom_envs.SearchQA.env_config.retrieval_backend=parallel`.

Prepare the question/answer parquet data as below; the Wikipedia index and local
retrieval server are unnecessary for this backend. The agent can use
`search[query]`, then `fetch[https://example.com/page]` to read a result URL, then
`finish[answer]`. Both retrieval actions count toward `max_steps` and return
excerpts bounded by `max_total_chars`; search also honors `max_search_results`.
`retrieval_timeout` bounds each MCP connection, handshake and tool call together.

This is a different retrieval regime: the live web changes over time and can
contain benchmark answers. Use the default `local` backend and fixed Wikipedia
corpus to reproduce the SearchQA experiments. The anonymous service is intended
for light use and has rate limits; it is not a replacement for the dedicated
retrieval server used for high-concurrency training. Service errors appear in
the observation, with no automatic switch to another backend.

Only the explicit `parallel` selection sends queries and fetch URLs to
`https://search.parallel.ai/mcp`. It does not read API keys from the environment.
Each `reset()` creates an opaque episode ID reused by all search and fetch calls
until the next reset. Existing local server settings stay available, and
`mock_mode=True` takes precedence over either backend and makes no network calls.

## Setup

### 1. Prepare data

```bash
# Download HotpotQA → data/search/{train,val}.parquet
python scripts/prepare_search_data.py

# Download Wikipedia corpus + FAISS index (~74GB) → search_data/prebuilt_indices/
python scripts/download_search_index.py
```

### 2. Start the retrieval server

The retrieval server provides dense retrieval over ~21M Wikipedia passages using E5-base-v2 embeddings and a FAISS Flat index.

```bash
python scripts/retrieval/server.py \
    --data_dir ./search_data/prebuilt_indices \
    --port 8000 --host 127.0.0.1 \
    --device cuda:0 --gpu_memory_limit_mb 6144
```

Loading the 61GB FAISS index takes 2-5 minutes. Verify with:

```bash
curl http://127.0.0.1:8000/health
```

**Important: GPU deployment recommendation**

We recommend running the retrieval server on a **dedicated GPU** separate from training. In our experiments, placing the E5 server on the same GPU as training (e.g., GPU 0) caused CUDA OOM errors — vLLM rollout and training both compete for GPU memory, squeezing out the retrieval server process.

Run the server on a GPU not used by training (e.g., `--device cuda:7`, train on GPUs 0-6). During rollout, hundreds of environments issue concurrent retrieval requests (e.g., 256 env groups can produce 1000+ requests). Running on CPU cannot keep up with this concurrency and causes timeouts. A dedicated GPU with `threading.Lock` serialization handles this load reliably.

### 3. Run training

```bash
# PPO, no filtering (baseline)
bash scripts/runs/run_search_benchmark.sh \
    --algos PPO \
    --gpus 0,1,2,3,4,5,6,7 --gpus-per-exp 8

# PPO, top_k filtering (keep top 25% by reward variance)
bash scripts/runs/run_search_benchmark.sh \
    --algos PPO --filter-strategy top_k --filter-value 0.25 \
    --gpus 0,1,2,3,4,5,6,7 --gpus-per-exp 8

# PPO, top_p filtering (keep 90% by reward variance)
bash scripts/runs/run_search_benchmark.sh \
    --algos PPO --filter-strategy top_p --filter-value 0.9 \
    --gpus 0,1,2,3,4,5,6,7 --gpus-per-exp 8
```

Key training parameters (pass via `run_search_benchmark.sh` flags):

| Flag | Description | Default |
|------|-------------|---------|
| `--algos` | Algorithm: PPO or GRPO | PPO |
| `--filter-strategy` | Rollout filter strategy: `top_p`, `top_k`, etc. | `top_p` |
| `--filter-value` | Filter value (1.0 = no filtering) | `1.0` |
| `--gpus` | Comma-separated GPU IDs | auto-detect |
| `--gpus-per-exp` | GPUs per experiment | 1 |
| `--gpu-memory-utilization` | vLLM KV cache memory fraction | 0.6 |
| `--micro-batch` | Micro batch size per GPU | config default |
| `--mini-batch` | PPO mini batch size | config default |
| `--save-freq` | Checkpoint save frequency | -1 (disabled) |
| `--steps` | Total training steps | 200 |
| `--retrieval-port` | Retrieval server port | 8000 |

## Config

The search environment config is at `config/_9_search.yaml`. Key settings:

```yaml
micro_batch_size_per_gpu: 4
ppo_mini_batch_size: 32

agent_proxy:
  max_turn: 5              # up to 5 search rounds
  max_actions_per_turn: 1  # one action per response

actor_rollout_ref:
  rollout:
    max_model_len: 5000
    max_num_batched_tokens: 5000

es_manager:
  train:
    env_groups: 16
    group_size: 8           # 16 × 8 = 128 samples per batch
```

Note: when using `top_k` filtering with a small value (e.g., 0.25), the effective batch size after filtering is `env_groups × group_size × filter_value`. Ensure `ppo_mini_batch_size` does not exceed this value, or training will fail with an assertion error.

## Acknowledgment

The search environment is adapted from the [RLLM](https://github.com/rllm-org/rllm) project. We thank the RLLM authors for their open-source contributions.
