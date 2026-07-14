"""
Synthetic data generators for GeoDCD.
Supports: Lorenz96 (single ring) and ClusterLorenz (multiple independent rings).
"""

import os
import argparse
import numpy as np
from scipy.integrate import odeint


def save_data(data, ground_truth, coords, base_path, dataset_name, replica_id):
    """Save generated data, ground truth, and coordinates as .npy files."""
    data_dir = os.path.join(base_path, dataset_name)
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, f'data_{replica_id}.npy'), data)
    np.save(os.path.join(data_dir, f'gt_{replica_id}.npy'), ground_truth)
    np.save(os.path.join(data_dir, f'coords_{replica_id}.npy'), coords)
    print(f"Saved replica {replica_id} to {data_dir}")


def generate_lorenz96_system(p, T, F=10.0, seed=0, delta_t=0.1, burn_in=1000,
                              noise_scale=0.1, center=(0, 0), radius=5.0):
    """
    Generate data from the Lorenz96 ODE system.
    Returns (T, p) time series, (p, p) ring-topology GT, and (p, 2) coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    def lorenz96_deriv(x, t):
        x_plus_1 = np.roll(x, -1)
        x_minus_1 = np.roll(x, 1)
        x_minus_2 = np.roll(x, 2)
        return (x_plus_1 - x_minus_2) * x_minus_1 - x + F

    x0 = np.random.normal(scale=0.01, size=p) + F
    total_steps = T + burn_in
    t = np.linspace(0, total_steps * delta_t, total_steps)
    X_full = odeint(lorenz96_deriv, x0, t)
    X = X_full[burn_in:, :]

    if noise_scale > 0:
        X += np.random.normal(scale=noise_scale, size=X.shape)

    # Ground truth: ring topology
    # x_i depends on x_{i-1}, x_{i-2}, x_{i+1}
    gt = np.zeros((p, p), dtype=int)
    for i in range(p):
        gt[(i - 1) % p, i] = 1
        gt[(i - 2) % p, i] = 1
        gt[(i + 1) % p, i] = 1
        gt[i, i] = 1

    # Coordinates on a ring centered at `center`
    angles = np.linspace(0, 2 * np.pi, p, endpoint=False)
    coords = np.stack([
        radius * np.cos(angles) + center[0],
        radius * np.sin(angles) + center[1]
    ], axis=1)
    coords += np.random.normal(0, 0.1, coords.shape)

    return X, gt, coords


def generate_cluster_lorenz(p, T, seed, num_groups=4):
    """
    Generate multiple independent Lorenz96 rings arranged in a grid.
    Returns block-diagonal ground truth (one ring per group, no cross-ring edges).
    """
    if p % num_groups != 0:
        raise ValueError(f"p ({p}) must be divisible by num_groups ({num_groups})")

    nodes_per_group = p // num_groups

    all_data = []
    all_gt = np.zeros((p, p), dtype=int)
    all_coords = []

    grid_size = int(np.ceil(np.sqrt(num_groups)))
    spacing = 15.0

    for g in range(num_groups):
        row = g // grid_size
        col = g % grid_size
        center = (col * spacing, row * spacing)
        group_seed = seed + g * 100

        X, gt, coords = generate_lorenz96_system(
            p=nodes_per_group, T=T, seed=group_seed,
            center=center, radius=3.0
        )

        all_data.append(X)
        all_coords.append(coords)

        start = g * nodes_per_group
        end = (g + 1) * nodes_per_group
        all_gt[start:end, start:end] = gt

    final_data = np.concatenate(all_data, axis=1)
    final_coords = np.concatenate(all_coords, axis=0)
    return final_data, all_gt, final_coords


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for GeoDCD.")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['lorenz96', 'cluster_lorenz'],
                        help='Dataset type to generate.')
    parser.add_argument('--num_replicas', type=int, default=5,
                        help='Number of independent replicas.')
    parser.add_argument('--output_path', type=str, default='data/synthetic',
                        help='Base output directory.')
    parser.add_argument('--T', type=int, default=1000, help='Time steps.')
    parser.add_argument('--p', type=int, default=32,
                        help='Total number of variables (nodes).')
    parser.add_argument('--num_groups', type=int, default=4,
                        help='Number of groups for cluster_lorenz.')
    args = parser.parse_args()

    for i in range(args.num_replicas):
        seed = 42 + i * 100
        print(f"Generating {args.dataset}, Replica {i+1}/{args.num_replicas} (seed={seed})")

        if args.dataset == 'lorenz96':
            data, gt, coords = generate_lorenz96_system(p=args.p, T=args.T, seed=seed)
        elif args.dataset == 'cluster_lorenz':
            data, gt, coords = generate_cluster_lorenz(
                p=args.p, T=args.T, seed=seed, num_groups=args.num_groups
            )
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        save_data(data, gt, coords, args.output_path, args.dataset, i)


if __name__ == '__main__':
    main()
