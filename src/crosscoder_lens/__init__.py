"""
CrossCoder Lens - A library for training and analyzing CrossCoders.
"""

from .model import CrossCoder, CrossCoderConfig
from .trainer import CrossCoderTrainer, CrossCoderTrainerConfig
from .visualization import CrossCoderVisualizer

__version__ = "0.1.0"
__all__ = [
    "CrossCoder",
    "CrossCoderConfig",
    "CrossCoderTrainer",
    "CrossCoderTrainerConfig",
    "CrossCoderVisualizer",
]
