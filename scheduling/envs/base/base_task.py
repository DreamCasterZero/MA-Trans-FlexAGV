
class BaseTask:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.env.device if hasattr(cfg, "env") else "cpu"

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_observations(self):
        raise NotImplementedError

    def is_done(self):
        raise NotImplementedError
