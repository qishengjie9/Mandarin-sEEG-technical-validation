from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, category: str) -> None:
        self.category = category
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        key = name.lower()

        def decorator(item: Callable[..., Any]):
            if key in self._items:
                raise KeyError(f"duplicate {self.category} registration: {key}")
            self._items[key] = item
            return item

        return decorator

    def build(self, config: dict | str, **extra: Any):
        if isinstance(config, str):
            name, kwargs = config, {}
        else:
            name = str(config.get("name", ""))
            kwargs = {key: value for key, value in config.items() if key != "name"}
        key = name.lower()
        if key not in self._items:
            raise KeyError(f"unknown {self.category} {name!r}; available={sorted(self._items)}")
        kwargs.update(extra)
        return self._items[key](**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


MODELS = Registry("model")
TEMPORAL_ENCODERS = Registry("temporal encoder")
SPATIAL_ENCODERS = Registry("spatial encoder")
CHANNEL_POOLERS = Registry("channel pooler")
CONTEXT_ENCODERS = Registry("context encoder")


def build_model(config: dict, task_sizes: dict[str, int], n_domains: int):
    # Import triggers component/model registration while keeping callers independent
    # of the concrete architecture module.
    from . import modular_models  # noqa: F401

    return MODELS.build(config, task_sizes=task_sizes, n_domains=n_domains)

