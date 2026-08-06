# ABOUTME: BaseDemoLoader abstract class: a single load() -> DemoDataset contract for all demo loaders.
# ABOUTME: Subclassed by GridWorldDemoLoader and STWebAgentDemoLoader.
"""Base class for demonstration loaders."""
from __future__ import annotations

from abc import ABC, abstractmethod

from icrl.core.types import DemoDataset


class BaseDemoLoader(ABC):
    @abstractmethod
    def load(self) -> DemoDataset:
        ...
