from typing import Type, Callable, Union, Optional

_METHOD_REGISTRY = {}

def register_method(
    name: str,
    model_cls: Type,
    logs: Union[str, Callable[[object], str]] = None,
):
    _METHOD_REGISTRY[name.lower()] = {
        "model": model_cls,
        "logs": logs,
    }

def get_method(name: str):
    key = name.lower()
    if key not in _METHOD_REGISTRY:
        raise ValueError(f"Method {name} not registered.")
    return _METHOD_REGISTRY[key]
