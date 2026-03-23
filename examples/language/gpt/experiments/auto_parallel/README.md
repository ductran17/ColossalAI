# Auto-Parallelism with GPT2

## Requirements

Before you can launch training, you need to install the following requirements.

### Install PyTorch

```bash
#pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### Install Colossal-AI (0.5.0)

```bash
pip install -e /home/ductm27/ColossalAI/
```

### Install transformers

```bash
pip install transformers==4.51.3
```

### Install pulp and coin-or-cbc

```bash
pip install pulp
conda install -c conda-forge coin-or-cbc
```

## Dataset

For simplicity, the input data is randomly generated here.

## Training

```bash
#Run the auto parallel resnet example with 4 GPUs with a dummy dataset.
colossalai run --nproc_per_node 4 auto_parallel_with_gpt.py

torchrun --nproc_per_node=2 auto_parallel_with_gpt2_large.py --steps 100 --seq_len 512 --per_device_bs 2
```
