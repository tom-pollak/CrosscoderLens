"""
CrossCoder Lens - A library for training and analyzing CrossCoders.
"""

from .model import CrossCoder, CrossCoderConfig
from .store.cached_activation_store import (
    CacheActivationsRunnerConfig,
    CachedActivationsStore,
)
from .trainer import CrossCoderTrainer
from .visualization import CrossCoderVisualizer

__version__ = "0.1.0"
__all__ = [
    "CrossCoder",
    "CrossCoderConfig",
    "CrossCoderTrainer",
    "CachedActivationsStore",
    "CacheActivationsRunnerConfig",
    "CrossCoderVisualizer",
]
