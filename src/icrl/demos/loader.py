"""Base class for demonstration loaders."""
from __future__ import annotations

from abc import ABC, abstractmethod

from icrl.core.types import DemoDataset


class BaseDemoLoader(ABC):
    @abstractmethod
    def load(self) -> DemoDataset:
        ...
