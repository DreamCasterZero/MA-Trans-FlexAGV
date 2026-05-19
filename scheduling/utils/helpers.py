import os
import random
import torch
import numpy as np


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        val = getattr(obj, key)
        if isinstance(val, list):
            result[key] = [class_to_dict(item) for item in val]
        else:
            result[key] = class_to_dict(val)
    return result


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = sorted(os.listdir(root))
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except Exception:
        raise ValueError(f"No runs found in: {root}")

    load_run = last_run if load_run == -1 else os.path.join(root, load_run)

    if checkpoint == -1:
        models = sorted(f for f in os.listdir(load_run) if "model" in f)
        model  = models[-1]
    else:
        model = f"model_{checkpoint}.pt"

    return os.path.join(load_run, model)
