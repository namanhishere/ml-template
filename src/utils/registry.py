from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn_or_cls: Callable[..., Any]) -> Callable[..., Any]:
            self._items[name] = fn_or_cls
            return fn_or_cls

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            raise KeyError(
                f"{self.name} '{name}' not found. Available: {list(self._items.keys())}"
            )
        return self._items[name]

    def instantiate(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(name)(*args, **kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def list(self) -> list[str]:
        return list(self._items.keys())


DATASETS = Registry("dataset")
MODELS = Registry("model")
LOSSES = Registry("loss")
METRICS = Registry("metric")
EXPERIMENTS = Registry("experiment")
CALLBACKS = Registry("callback")
PREDICTORS = Registry("predictor")
FINETUNE_STRATEGIES = Registry("finetune_strategy")
