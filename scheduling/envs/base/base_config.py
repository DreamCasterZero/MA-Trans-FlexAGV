import inspect

class BaseConfig:
    def __init__(self) -> None:
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        for key in dir(obj):
            if key.startswith("__"):
                continue

            var = getattr(obj, key)

            if inspect.isclass(var):
                instance = var()
                setattr(obj, key, instance)
                BaseConfig.init_member_classes(instance)