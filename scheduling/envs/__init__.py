from .custom.scheduling_env import SchedulingEnv
from .custom.scheduling_config import SchedulingConfig, SchedulingConfigAlgo
from ..utils.task_registry import task_registry

task_registry.register(
    "fjsp_agv",
    SchedulingEnv,
    SchedulingConfig(),
    SchedulingConfigAlgo(),
)
