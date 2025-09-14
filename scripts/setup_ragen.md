# Manual Scripts to Setup Environment
```bash
conda create -n ragen python=3.12 -y
conda activate ragen


git clone git@github.com:ZihanWang314/ragen.git
cd ragen

pip install -e .
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Optional: to install flash-attn, you may need to install cuda-toolkit first if you don't have
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX # /opt/conda/envs/zero
pip3 install flash-attn --no-build-isolation

pip install -r requirements.txt

git submodule init
git submodule update
cd verl
pip install -e .
cd ..

```
