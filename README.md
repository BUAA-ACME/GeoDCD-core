# GeoDCD-core: Geometric Dynamic Causal Discovery

A clean, minimal implementation of the GeoDCD model for hierarchical causal discovery in large-scale time series.

**Key features:**
- Learnable geometric pooling for hierarchical node clustering
- Sparse causal graph discovery via basis decomposition
- Multi-scale consistency across hierarchy levels
- Dynamic causal strength estimation via Transformer encoding
- Cross-level information bridging

## Installation

```bash
pip install -r requirements.txt
```

For distributed training (optional):
```bash
pip install accelerate
# Configure:
accelerate config
```

## Data Generation

Generate synthetic Lorenz96 or ClusterLorenz datasets:

```bash
# Single Lorenz96 ring (32 nodes)
python generate_data.py --dataset lorenz96 --p 32 --T 2000 --num_replicas 5

# 4 independent Lorenz96 rings, 8 nodes each
python generate_data.py --dataset cluster_lorenz --p 32 --num_groups 4 --T 2000
```

This creates `.npy` files in `data/synthetic/{dataset_name}/`.

## Training

Train the model on synthetic data:

```bash
# Single GPU
python train.py --dataset lorenz96 --data_path ./data/synthetic \
    --hierarchy 16 8 --epochs 100

# Multi-GPU with accelerate
accelerate launch train.py --dataset lorenz96 --data_path ./data/synthetic \
    --hierarchy 16 8 --epochs 100
```

Training progress prints epoch loss and causal discovery metrics (F1, AUROC) when ground truth is available. The trained model and metrics JSON are saved to `./results/`.

## Inference

Evaluate a trained model:

```bash
python inference.py --model_path ./results/lorenz96/.../model.pth \
    --dataset lorenz96 --data_path ./data/synthetic
```

Outputs metrics (AUROC, F1, Precision, Recall, SHD) to `metrics.json`.

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | lorenz96 | Dataset name |
| `--data_path` | data/synthetic | Data directory |
| `--replica_id` | 0 | Dataset replica index |
| `--hierarchy` | [] | Number of clusters per level (e.g., `16 8`) |
| `--d_model` | 64 | Model dimension |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate |

## Citation

If you use GeoDCD in your research, please cite the original paper.

```bibtex
@article{geodcd,
  title={GeoDCD: Geometric Dynamic Causal Discovery},
  author={...},
  journal={...},
  year={2025}
}
```
