from typing import Type, Callable, Union, Optional

_METHOD_REGISTRY = {}

def register_method(
    name: str,
    model_cls: Type,
    loss: Type,
    transformation: Callable,
    logs: Union[str, Callable[[object, Optional[object]], str]] = None,
    default_params: Optional[dict] = None,

):
    _METHOD_REGISTRY[name.lower()] = {
        "model": model_cls,
        "loss": loss,
        "transformation": transformation,
        "logs": logs,
        "default_params": default_params or {},

    }

def get_method(name: str):
    key = name.lower()
    if key not in _METHOD_REGISTRY:
        raise ValueError(f"Method {name} not registered.")
    return _METHOD_REGISTRY[key]
