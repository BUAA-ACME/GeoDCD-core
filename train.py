"""
GeoDCD training script.
Trains the hierarchical causal discovery model on synthetic data.
Usage:
    accelerate launch train.py --dataset lorenz96 --data_path ./data/synthetic \\
        --hierarchy 16 8 --epochs 100
"""

import os
import argparse
import datetime
import json
import torch
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from model import GeoDCD
from dataloader import get_data_context
from metrics import count_accuracy


def train_one_epoch(model, loader, optimizer, accelerator, args):
    model.train()
    total_loss = 0.0

    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        optimizer.zero_grad()

        results = model(x)

        batch_loss = 0.0
        for res in results:
            x_pred, x_target = res['x_pred'], res['x_target']
            batch_loss += F.mse_loss(x_pred, x_target[..., 1:])

        unwrapped = accelerator.unwrap_model(model)
        l1_loss = args.lambda_l1 * unwrapped.get_structural_l1_loss()
        entropy_loss = args.lambda_entropy * unwrapped.get_entropy_loss()
        consistency_loss = args.lambda_consistency * unwrapped.get_consistency_loss()

        final_loss = batch_loss + l1_loss + entropy_loss + consistency_loss
        accelerator.backward(final_loss)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += final_loss.item()

    return total_loss / len(loader)


def main(args):
    accelerator = Accelerator()
    set_seed(args.seed)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(
        args.output_dir, args.dataset,
        f"{timestamp}-{args.dataset}-{args.replica_id}-GeoDCD"
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.print(f"Experiment: {args.dataset} | Output: {args.output_dir}")

    train_loader, _, meta = get_data_context(args)

    if args.N is None:
        sample_data = train_loader.dataset[0]
        if isinstance(sample_data, (list, tuple)):
            args.N = sample_data[0].shape[0]
        else:
            args.N = sample_data.shape[0]
        accelerator.print(f"Auto-detected N={args.N} from dataset")

    model = GeoDCD(
        N=args.N,
        coords=meta['coords'],
        hierarchy=args.hierarchy,
        d_model=args.d_model,
        num_bases=args.num_bases,
        penalty_factor=args.penalty_factor,
        shift_scale=args.shift_scale,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    progress_bar = tqdm(
        range(args.epochs),
        disable=not accelerator.is_local_main_process,
        desc="Training"
    )

    for ep in progress_bar:
        loss = train_one_epoch(model, train_loader, optimizer, accelerator, args)
        scheduler.step()

        if accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            log = {"loss": f"{loss:.6f}", "lr": f"{current_lr:.2e}"}

            gt_fine = meta.get('gt_fine')
            if (gt_fine is not None) and (gt_fine.ndim == 2):
                unwrapped = accelerator.unwrap_model(model)
                est_fine = (
                    unwrapped.layers[0]
                    .graph.get_soft_graph().detach().cpu().numpy()
                )
                metrics = count_accuracy(gt_fine, est_fine)
                log.update({k: f"{v:.4f}" for k, v in metrics.items()})
                postfix = (
                    f"L={loss:.4f}, "
                    f"F1={metrics['F1']:.3f}, "
                    f"AUC={metrics['AUROC']:.3f}, "
                    f"LR={current_lr:.1e}"
                )
            else:
                postfix = f"L={loss:.4f}, LR={current_lr:.1e}"

            progress_bar.set_postfix_str(postfix)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        # Save model weights
        save_path = os.path.join(args.output_dir, "model.pth")
        torch.save(unwrapped_model.state_dict(), save_path)
        accelerator.print(f"Model saved to {save_path}")

        # Final evaluation
        gt_fine = meta.get('gt_fine')
        if gt_fine is not None:
            est_fine = unwrapped_model.layers[0].graph.get_soft_graph().detach().cpu().numpy()
            if gt_fine.ndim == 3:
                if gt_fine.shape[0] == gt_fine.shape[1]:
                    gt_fine_2d = np.max(gt_fine, axis=-1)
                else:
                    gt_fine_2d = np.max(gt_fine, axis=0)
            else:
                gt_fine_2d = gt_fine

            metrics = count_accuracy(gt_fine_2d, est_fine)
            accelerator.print("\nFinal Metrics:")
            accelerator.print(json.dumps(metrics, indent=2))

            metrics_path = os.path.join(args.output_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            accelerator.print(f"Metrics saved to {metrics_path}")

        accelerator.print(f"Done. Results in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoDCD Training")

    # Data
    parser.add_argument("--dataset", type=str, default="lorenz96")
    parser.add_argument("--data_path", type=str, default="data/synthetic")
    parser.add_argument("--replica_id", type=int, default=0)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)

    # Model
    parser.add_argument("--hierarchy", type=int, nargs='+', default=[])
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_bases", type=int, default=4)
    parser.add_argument("--penalty_factor", type=float, default=5.0)
    parser.add_argument("--shift_scale", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4 if os.name != 'nt' else 0,
                        help="DataLoader workers (default: 4 on Linux/macOS, 0 on Windows)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_l1", type=float, default=1e-4)
    parser.add_argument("--lambda_entropy", type=float, default=0.01)
    parser.add_argument("--lambda_consistency", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--output_dir", type=str, default="./results")

    args = parser.parse_args()
    main(args)
