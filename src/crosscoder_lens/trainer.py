import warnings
from dataclasses import dataclass
from typing import Any

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.store.base_store import BaseStore
from sae_lens.training.sae_trainer import SAETrainer

from .model import CrossCoder


@dataclass(kw_only=True)
class CrossCoderTrainerConfig(LanguageModelSAERunnerConfig):
    n_models: int  # Number of models to compare

    def __post_init__(self):
        super().__post_init__()
        if not self.use_cached_activations:
            warnings.warn(
                "Using cached activations for CrossCoder training.", stacklevel=2
            )
            self.use_cached_activations = True
        assert self.architecture == "standard"

    def get_training_sae_cfg_dict(self) -> dict[str, Any]:
        return {**super().get_training_sae_cfg_dict(), "n_models": self.n_models}


class CrossCoderTrainer(SAETrainer):
    """Trainer for CrossCoder model"""

    def __init__(
        self,
        sae: CrossCoder,
        activation_store: BaseStore,
        save_checkpoint_fn,  # type: ignore
        cfg: CrossCoderTrainerConfig,
    ):
        # Create stores for each model checkpoint
        super().__init__(
            model=None,  # type: ignore
            sae=sae,
            activation_store=activation_store,  # type: ignore
            save_checkpoint_fn=save_checkpoint_fn,
            cfg=cfg,
        )

        # Disable model-specific evals
        self.trainer_eval_config.compute_l2_norms = False
        self.trainer_eval_config.compute_sparsity_metrics = False
        self.trainer_eval_config.compute_variance_metrics = False
        self.trainer_eval_config.compute_kl = False
        self.trainer_eval_config.compute_ce_loss = False
