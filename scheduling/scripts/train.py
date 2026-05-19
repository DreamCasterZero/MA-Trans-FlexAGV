import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ..utils import task_registry
from .. import envs  # noqa: F401 — triggers task registration


def get_args():
    parser = argparse.ArgumentParser(description='Train FJSP+AGV scheduling agent')
    parser.add_argument('--task',      type=str,  default='fjsp_agv')
    parser.add_argument('--resume',    action='store_true',
                        help='Resume training from a checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path (used with --resume)')
    parser.add_argument('--val_set',   type=str,  default=None,
                        help='Path to validation .pt file (list of {PT, agv_num} dicts)')
    parser.add_argument('--log_root',       type=str,  default='default',
                        help='Root directory for logs (pass "none" to disable logging)')
    parser.add_argument('--total_episodes', type=int,  default=None,
                        help='Override total training episodes from config')
    parser.add_argument('--log_interval',   type=int,  default=None,
                        help='Override log print interval (episodes)')
    return parser.parse_args()


def main():
    args = get_args()

    env, _       = task_registry.make_env(args.task)
    _, train_cfg = task_registry.get_cfgs(args.task)

    if args.resume:
        train_cfg.runner.resume = True
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint_path = args.checkpoint
    if args.total_episodes is not None:
        train_cfg.runner.total_episodes = args.total_episodes
    if args.log_interval is not None:
        train_cfg.runner.log_interval = args.log_interval

    log_root = None if args.log_root == 'none' else args.log_root
    runner, _ = task_registry.make_alg_runner(env, name=args.task,
                                               train_cfg=train_cfg,
                                               log_root=log_root)

    val_dataset = None
    if args.val_set:
        val_dataset = torch.load(args.val_set)
        print(f"Loaded validation set: {len(val_dataset)} cases from {args.val_set}")

    runner.run(val_dataset)


if __name__ == '__main__':
    main()
