import os
from datetime import datetime
from typing import Tuple

from .. import SCHEDULING_DIR
from .helpers import set_seed

class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs     = {}
        self.train_cfgs   = {}

    def register(self, name: str, task_class, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name]     = env_cfg
        self.train_cfgs[name]   = train_cfg

    def get_task_class(self, name: str):
        return self.task_classes[name]

    def get_cfgs(self, name: str) -> Tuple:
        return self.env_cfgs[name], self.train_cfgs[name]

    def make_env(self, name: str, env_cfg=None):
        if name not in self.task_classes:
            raise ValueError(f"Task '{name}' is not registered")
        task_class = self.get_task_class(name)
        if env_cfg is None:
            env_cfg, train_cfg = self.get_cfgs(name)
            set_seed(train_cfg.seed)
        return task_class(env_cfg), env_cfg

    def make_alg_runner(self, env, name=None, train_cfg=None, log_root="default"):
        from ..algo.ppo.runner import Runner  # lazy import，runner 建好后生效

        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be provided")
            _, train_cfg = self.get_cfgs(name)

        if log_root == "default":
            log_root = os.path.join(
                SCHEDULING_DIR, "logs", train_cfg.runner.experiment_name
            )

        log_dir = (
            os.path.join(log_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
            if log_root is not None else None
        )

        runner = Runner(env, train_cfg, log_dir)

        if train_cfg.runner.resume and train_cfg.runner.checkpoint_path:
            runner.load(train_cfg.runner.checkpoint_path)

        return runner, train_cfg


task_registry = TaskRegistry()
