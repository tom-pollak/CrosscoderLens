from pathlib import Path

import torch as t
from sae_lens.store.base_store import BaseStore
from sae_lens.training.sae_trainer import SAETrainer

from .model import CrossCoder


class CrossCoderTrainer(SAETrainer):
    """Trainer for CrossCoder model"""

    def __init__(self, sae: CrossCoder, activation_store: BaseStore, **kwargs):
        # Create stores for each model checkpoint
        super().__init__(
            model=None,  # type: ignore
            sae=sae,
            activation_store=activation_store,  # type: ignore
            **kwargs,
        )

        self.trainer_eval_config.compute_l2_norms = False
        self.trainer_eval_config.compute_sparsity_metrics = False
        self.trainer_eval_config.compute_variance_metrics = False
        self.trainer_eval_config.compute_kl = False
        self.trainer_eval_config.compute_ce_loss = False
