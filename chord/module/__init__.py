modules = {}

def register(name):
    def decorator(cls):
        modules[name] = cls
        return cls
    return decorator


def make(name, config):
    model = modules[name](config)
    return model


from . import (
    light,
    stable_diffusion,
    chord,
)