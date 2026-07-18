"""
GeoDCD inference script.
Loads a trained model and evaluates on a dataset. Computes causal discovery
metrics (AUROC, F1, etc.) against ground truth if available.
Usage:
    accelerate launch inference.py --model_path ./results/model.pth \\
        --dataset lorenz96 --data_path ./data/synthetic
"""

import os
import argparse
import datetime
import json
import torch
import numpy as np
from accelerate import Accelerator
from tqdm.auto import tqdm

from model import GeoDCD
from dataloader import get_data_context
from metrics import count_accuracy


def main(args):
    accelerator = Accelerator()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(
        args.output_dir, args.dataset, timestamp, "inference"
    )

    accelerator.print(f"Output Dir: {args.output_dir}")
    accelerator.print(f"Loading model from: {args.model_path}")

    train_loader, _, meta = get_data_context(args)

    if args.N is None:
        sample_data = train_loader.dataset[0]
        if isinstance(sample_data, (list, tuple)):
            args.N = sample_data[0].shape[0]
        else:
            args.N = sample_data.shape[0]
        accelerator.print(f"Auto-detected N={args.N} from dataset")

    # Infer hierarchy from the checkpoint if not provided, so the inference
    # architecture matches training (the README example omits --hierarchy).
    if not args.hierarchy:
        try:
            ckpt = torch.load(args.model_path, map_location='cpu', weights_only=True)
            layer_idxs = set()
            pooler_sizes = {}
            for k, v in ckpt.items():
                parts = k.split('.')
                if len(parts) >= 2 and parts[0] == 'layers' and parts[1].isdigit():
                    layer_idxs.add(int(parts[1]))
                if len(parts) >= 3 and parts[0] == 'poolers' and parts[1].isdigit() \
                        and parts[2] == 'centroids':
                    pooler_sizes[int(parts[1])] = v.shape[0]
            num_levels = len(layer_idxs)
            hierarchy = [pooler_sizes[i] for i in range(num_levels - 1)]
            args.hierarchy = hierarchy
            accelerator.print(f"Inferred hierarchy from checkpoint: {hierarchy}")
        except Exception as e:
            accelerator.print(f"Could not infer hierarchy ({e}); using empty hierarchy.")

    model = GeoDCD(
        N=args.N,
        coords=meta['coords'],
        hierarchy=args.hierarchy,
        d_model=args.d_model,
        num_bases=args.num_bases,
    )

    try:
        state_dict = torch.load(args.model_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        accelerator.print("Weights loaded successfully.")
        if len(unexpected) > 0:
            accelerator.print(f"Ignored unexpected keys: {len(unexpected)}")
        if len(missing) > 0:
            accelerator.print(f"Missing keys: {len(missing)}")
    except Exception as e:
        accelerator.print(f"Failed to load weights: {e}")
        return

    model.to(accelerator.device)
    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    # Warm-up forward pass
    with torch.no_grad():
        dummy_x = torch.zeros(1, args.N, args.window_size).to(accelerator.device)
        model(dummy_x)

    # Extract estimated causal graph (finest level)
    est_fine = model.layers[0].graph.get_soft_graph().detach().cpu().numpy()
    accelerator.print(f"Estimated graph shape: {est_fine.shape}")

    # Compute metrics against ground truth
    gt_fine = meta.get('gt_fine')
    if gt_fine is not None:
        if gt_fine.ndim == 3:
            if gt_fine.shape[0] == gt_fine.shape[1]:
                gt_fine = np.max(gt_fine, axis=-1)
            else:
                gt_fine = np.max(gt_fine, axis=0)

        metrics = count_accuracy(gt_fine, est_fine)
        accelerator.print("\nInference Metrics:")
        accelerator.print(json.dumps(metrics, indent=2))

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        accelerator.print(f"Metrics saved to {metrics_path}")
    else:
        accelerator.print("No ground truth available; metrics skipped.")
        np.save(os.path.join(args.output_dir, "est_graph.npy"), est_fine)

    accelerator.print(f"Inference finished. Results in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoDCD Inference")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model.pth")
    parser.add_argument("--dataset", type=str, default="lorenz96")
    parser.add_argument("--data_path", type=str, default="data/synthetic")
    parser.add_argument("--replica_id", type=int, default=0)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hierarchy", type=int, nargs='+', default=[])
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_bases", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4 if os.name != 'nt' else 0,
                        help="DataLoader workers (default: 4 on Linux/macOS, 0 on Windows)")
    parser.add_argument("--output_dir", type=str, default="./results")

    args = parser.parse_args()
    main(args)
