# Scripts README

## Release Environment Setup

Use the setup flow below for the validated RAGEN environment.

This setup has been validated on `H100`, `H200`, and `B200`, and supports:
- `bandit`
- `sokoban`
- `frozenlake`
- `metamathqa`
- `countdown`
- `deepcoder`

## Requirements

- `CUDA >= 12.8`

### 1. Clone the repository

```bash
git clone https://github.com/mll-lab-nu/RAGEN.git
cd RAGEN
```

### 2. Create and activate the conda environment

```bash
conda create -n ragen python=3.12 -y
conda activate ragen
```

### 3. Run the environment setup script

```bash
bash scripts/setup_ragen.sh
```

If you want to install the `search` environment, use the following command:

```bash
bash scripts/setup_ragen.sh --with-search
```

This release setup does not install `webshop`. If you need `webshop`, use its separate setup flow instead of `setup_ragen.sh`.

If you want to run WebShop experiments, see [docs/experiment_webshop_release.md](../docs/experiment_webshop_release.md).
