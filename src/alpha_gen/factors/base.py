"""Factor interface and registry.

A factor maps :class:`MarketData` to a wide DataFrame of raw scores with the same index/columns.
Convention: **higher score = higher expected forward return**. Scores at ``t`` must only use data
labelled ``<= t`` (the test-suite enforces this for every registered factor).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, TypeVar

import pandas as pd

from alpha_gen.data.market import MarketData

_REGISTRY: dict[str, type[Factor]] = {}

F = TypeVar("F", bound="type[Factor]")


@dataclass(frozen=True)
class Factor(ABC):
    """Base class. Subclasses are frozen dataclasses whose fields are the factor's parameters."""

    name: ClassVar[str] = ""
    requires: ClassVar[tuple[str, ...]] = ("close",)
    description: ClassVar[str] = ""

    @abstractmethod
    def compute(self, data: MarketData) -> pd.DataFrame:
        """Return raw scores aligned to ``data.close``."""

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def label(self) -> str:
        args = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({args})"

    def __call__(self, data: MarketData) -> pd.DataFrame:
        for field in self.requires:
            data.require(field)
        return self.compute(data).reindex(index=data.index, columns=data.symbols)


def register(cls: F) -> F:
    """Class decorator adding a factor to the global registry under ``cls.name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name`")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"Factor name {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def available_factors() -> dict[str, type[Factor]]:
    """All registered factor classes by name."""
    return dict(sorted(_REGISTRY.items()))


def get_factor(name: str, **params: Any) -> Factor:
    """Instantiate a registered factor, e.g. ``get_factor("momentum", lookback=72)``."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown factor {name!r}; available: {sorted(_REGISTRY)}") from None
    return cls(**params)


def _check_positive(**values: int) -> None:
    for key, value in values.items():
        if value < 1:
            raise ValueError(f"{key} must be >= 1, got {value}")
