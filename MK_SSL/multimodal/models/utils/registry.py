from typing import Type, Callable, Union, Optional

_METHOD_REGISTRY = {}

def register_method(
    name: str,
    model_cls: Type,
    loss_fn: Callable,
    transformation: Callable,
    logs: Union[str, Callable[[object, Optional[object]], str]] = None,
):
    _METHOD_REGISTRY[name.lower()] = {
        "model": model_cls,
        "loss": loss_fn,
        "transformation": transformation,
        "logs": logs,
    }

def get_method(name: str):
    key = name.lower()
    if key not in _METHOD_REGISTRY:
        raise ValueError(f"Method {name} not registered.")
    return _METHOD_REGISTRY[key]
