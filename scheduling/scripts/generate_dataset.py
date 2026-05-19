import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ..utils import task_registry
from ..algo.ppo.runner import Runner
from .. import envs  # noqa: F401 — triggers task registration

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def get_args():
    parser = argparse.ArgumentParser(description='Generate FJSP+AGV dataset (.pt)')
    parser.add_argument('--task',      type=str, default='fjsp_agv')
    parser.add_argument('--split',     type=str, default='val',
                        choices=['val', 'test'],
                        help='Which split to generate: val or test')
    parser.add_argument('--num_cases', type=int, default=100,
                        help='Number of instances to generate')
    parser.add_argument('--name',      type=str, default=None,
                        help='File name (without .pt). '
                                'Defaults to {task}_{num_cases}')
    return parser.parse_args()


def main():
    args = get_args()

    env, _       = task_registry.make_env(args.task)
    _, train_cfg = task_registry.get_cfgs(args.task)
    runner       = Runner(env, train_cfg, log_dir=None)

    dataset = []
    for _ in range(args.num_cases):
        pt, agv_num = runner._generate_instance()
        dataset.append({'PT': pt, 'agv_num': agv_num})

    name     = args.name or f"{args.task}_{args.num_cases}"
    out_dir  = os.path.join(DATA_ROOT, args.split)
    out_path = os.path.join(out_dir, f"{name}.pt")

    os.makedirs(out_dir, exist_ok=True)
    torch.save(dataset, out_path)
    print(f"Saved {len(dataset)} cases -> {out_path}")


if __name__ == '__main__':
    main()
