# Manual Scripts to Setup Environment
```bash
conda create -n ragen python=3.9 uv -y
conda activate ragen


git clone git@github.com:ZihanWang314/ragen.git
cd ragen

uv pip install -e .
uv pip install torch==2.4.0

# Optional: install flash-attn using a matched release from https://github.com/Dao-AILab/flash-attention/releases/
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

uv pip install -r requirements.txt
```
