import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ..utils import task_registry
from .. import envs  # noqa: F401 — triggers task registration


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate FJSP+AGV scheduling agent')
    parser.add_argument('--task',       type=str, default='fjsp_agv')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--test_set',   type=str, default=None,
                        help='Path to test .pt file (list of {PT, agv_num} dicts). '
                             'If omitted, random cases are generated.')
    parser.add_argument('--num_cases',  type=int, default=100,
                        help='Number of random cases to generate when --test_set is not given')
    parser.add_argument('--save_csv',   type=str, default=None,
                        help='Optional path to save per-case makespan results as CSV')
    return parser.parse_args()


def main():
    args = get_args()

    env, _       = task_registry.make_env(args.task)
    _, train_cfg = task_registry.get_cfgs(args.task)

    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(env, name=args.task,
                                               train_cfg=train_cfg,
                                               log_root=None)
    runner.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    if args.test_set:
        dataset = torch.load(args.test_set)
        print(f"Loaded test set: {len(dataset)} cases from {args.test_set}")
    else:
        raw = [runner._generate_instance() for _ in range(args.num_cases)]
        dataset = [{'PT': pt, 'agv_num': agv_num} for pt, agv_num in raw]
        print(f"Generated {len(dataset)} random cases")

    makespans = []
    for i, case in enumerate(dataset):
        state, pad_mask, job_mask, agv_mask = env.reset(case['PT'], case['agv_num'])
        done = False
        while not done:
            job_act, _, _ = runner.job_agent.choose_action(
                state, pad_mask, job_mask, greedy=True)
            mid_state, pad_mask, agv_mask, mach_act = env.job_step(job_act)
            agv_act, _, _ = runner.agv_agent.choose_action(
                mid_state, pad_mask, agv_mask, greedy=True)
            state, pad_mask, job_mask, agv_mask, _, done, _ = \
                env.step(job_act, mach_act, agv_act)
        makespans.append(env.fjsp.makespan)

    arr = np.array(makespans, dtype=np.float32)
    print(f"\nResults ({len(arr)} cases):")
    print(f"  Mean : {arr.mean():.2f}")
    print(f"  Std  : {arr.std():.2f}")
    print(f"  Min  : {arr.min():.2f}")
    print(f"  Max  : {arr.max():.2f}")

    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['case', 'makespan'])
            for i, ms in enumerate(makespans):
                writer.writerow([i, ms])
        print(f"Saved results to {args.save_csv}")


if __name__ == '__main__':
    main()
