# Nanochat Speedrun (Modal)

Run Andrej Karpathy's nanochat speedrun on Modal GPUs with persistent storage for datasets, checkpoints, and logs.

## What this does
- Clones the nanochat repo inside a Modal container
- Checks out a specific branch/tag/commit
- Creates persistent `data/` and `logs/` symlinks backed by a Modal volume
- Uses 8x Nvidia H100 GPUs for maximum throughput
- **Saves checkpoints every 100 training steps** for fine-grained resumability
- Automatically resumes if a checkpoint exists
- Provides a 10-step smoke test for quick validation
- Uses NVIDIA CUDA + cuDNN base image for full GPU library support

## Prerequisites
- A Modal account and configured CLI (`modal setup`)
- Access to H100 GPUs (or modify `GPU_CONFIG` to use available GPUs like `A100:8`)

## Install Modal CLI
If `modal` is not on your PATH, install it locally:

```bash
python3 -m pip install --user modal
