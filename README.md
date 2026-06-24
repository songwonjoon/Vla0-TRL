# vla0-trl: Minimal VLA-0 Reimplementation with TRL

unofficial source code analysis and reimplementation of [VLA-0](https://github.com/NVlabs/vla0), built using [Vla0-TRL](https://github.com/MilkClouds/vla0-trl) and [TRL](https://github.com/huggingface/trl)'s SFTTrainer.

## Installation

<!-- TODO: upgrade lerobot -->

We recommend using [`uv`](https://docs.astral.sh/uv/) for managing dependencies.

```bash
uv venv --python 3.11
uv pip install -e .
# LeRobot dependency
GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/huggingface/lerobot.git@f39652707caed42a7cd5ab36066da5663b9565eb

# For evaluation
uv pip install -e ".[eval]"

# Do not forget activating your venv
source .venv/bin/activate
```

## Usage

### Train

```bash
# Single GPU
python scripts/train.py --config configs/vla0.yaml

# Multi-GPU
accelerate launch --num_processes=8 scripts/train.py --config configs/vla0.yaml
```

### Eval

```bash
python scripts/eval.py \
    --model_path ./runs/vla0/checkpoint-xxx \
    --task_suite libero_spatial \
    --action_horizon 8 \
    --ensemble_prediction 8 \
    --torch_compile \
    --skip_evaluated \
    --shard_id 0 --num_shards 10
```

| Argument | Description |
|----------|-------------|
| `--task_suite` | Task suite: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| `--action_horizon` | Execute N actions before re-querying model (default: 1) |
| `--ensemble_prediction` | Average N overlapping action chunks (default: 1 = off) |
| `--torch_compile` | Enable torch.compile for faster inference |
| `--skip_evaluated` | Skip episodes with existing result videos |
| `--shard_id`, `--num_shards` | Parallelize: run shard M of N (e.g., 0/10, 1/10, ...) |
| `--log_dir` | Output directory (default: auto-generated with timestamp) |

Note: When running multiple shards in parallel, specify `--log_dir` explicitly to ensure all shards write to the same directory.

### SLURM

For SLURM users, see [`scripts/train.sbatch`](scripts/train.sbatch) and [`scripts/eval.sbatch`](scripts/eval.sbatch). The `eval.sbatch` demonstrates batch evaluation with round-robin shard distribution across multiple GPUs.

## Configuration

See [`configs/vla0.yaml`](configs/vla0.yaml). Key parameters:

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 4e-5 (5e-6 × 8 GPUs) |
| `num_train_epochs` | 32 |
| `per_device_train_batch_size` | 8 |
| `horizon` | 8 |

Training 80k steps takes ~18h on 8×H100. Batch eval with [`eval.sbatch`](scripts/eval.sbatch) takes ~4h with 50 episode per task. I expect the computational cost of training and evaluation can be drastically reduced, though the solution remains an open question.

## Project Structure

```
├── configs/vla0.yaml       # Training config
├── scripts/
│   ├── train.py            # Training entry
│   └── eval.py             # Evaluation entry
└── src/
    ├── rv_train/           # Dataset, collator, model
    └── rv_eval/            # LIBERO evaluator
```

## Limitations (inherited from VLA-0)

- **LIBERO only** — other environments not ported
- **Qwen2.5-VL only** — other backbones not supported

## Known Issues

### Ensemble Prediction is Non-Functional (inherited from original)

Both the original VLA-0 (`libs/RoboVerse/roboverse/evals/libero/eval.py`) and this refactored implementation have a bug where `--ensemble_prediction` has **no effect** when `action_horizon >= horizon`. The ensemble logic trims previous chunks by `action_horizon` each step (`old_chunk = old_chunk[action_horizon:]`), which produces an empty array when `action_horizon == horizon`. With default settings (`horizon=8`, `action_horizon=8`), ensemble is completely disabled regardless of `--ensemble_prediction` value.

## Attribution

This is a derivative work of [VLA-0](https://github.com/NVlabs/vla0) by NVIDIA.

Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Citation

If you use this code, please cite both this repository and the original VLA-0 paper:

```bibtex
@misc{vla0-trl,
  author = {Suhwan Choi},
  title = {vla0-trl: Minimal VLA-0 Reimplementation with TRL},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/MilkClouds/vla0-trl},
  doi = {10.5281/ZENODO.18712424}
}

@article{goyal2025vla0,
  title={VLA-0: Building State-of-the-Art VLAs with Zero Modification},
  author={Goyal, Ankit and Hadfield, Hugo and Yang, Xuning and Blukis, Valts and Ramos, Fabio},
  journal={arXiv preprint arXiv:2510.13054},
  year={2025}
}
```
