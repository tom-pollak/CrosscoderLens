from typing import Any, cast

import wandb
from sae_lens.sae_training_runner import SAETrainingRunner
from sae_lens.store.base_store import BaseStore
from sae_lens.store.cached_activation_store import CachedActivationsStore
from sae_lens.training.sae_trainer import SAETrainer
from tqdm import tqdm

from .config import CrossCoderTrainerConfig, TrainingCrosscoderConfig
from .model import CrossCoder


class CrossCoderTrainer(SAETrainer):
    """Trainer for CrossCoder model"""

    sae: CrossCoder

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

    def fit(self) -> CrossCoder:
        pbar = tqdm(total=self.cfg.total_training_tokens, desc="Training SAE")

        self._estimate_norm_scaling_factor_if_needed()

        # Train loop
        while self.n_training_tokens < self.cfg.total_training_tokens:
            # Do a training step.
            layer_acts = self.activation_store.next_batch().to(
                self.sae.device
            )  # literally the only difference is not indexing in [:, 0, :]
            self.n_training_tokens += layer_acts.shape[0]  # and this line

            step_output = self._train_step(sae=self.sae, sae_in=layer_acts)

            if self.cfg.log_to_wandb:
                self._log_train_step(step_output)
                self._run_and_log_evals()

            self._checkpoint_if_needed()
            self.n_training_steps += 1
            self._update_pbar(step_output, pbar)

            ### If n_training_tokens > sae_group.cfg.training_tokens, then we should switch to fine-tuning (if we haven't already)
            self._begin_finetuning_if_needed()

        # fold the estimated norm scaling factor into the sae weights
        if self.activation_store.estimated_norm_scaling_factor is not None:
            self.sae.fold_activation_norm_scaling_factor(
                self.activation_store.estimated_norm_scaling_factor
            )

        # save final sae group to checkpoints folder
        self.save_checkpoint(
            trainer=self,
            checkpoint_name=f"final_{self.n_training_tokens}",
            wandb_aliases=["final_model"],
        )

        pbar.close()
        return self.sae


class CrossCoderTrainingRunner(SAETrainingRunner):
    sae: CrossCoder
    activations_store: CachedActivationsStore
    cfg: CrossCoderTrainerConfig

    def __init__(
        self,
        cfg: CrossCoderTrainerConfig,
        activation_store: CachedActivationsStore,
    ):
        self.cfg = cfg
        sae_cfg = TrainingCrosscoderConfig.from_dict(
            self.cfg.get_training_crosscoder_cfg_dict()
        )
        self.sae = CrossCoder(sae_cfg)
        self.activations_store = activation_store

    def run(self):
        """
        Run the training of the SAE.
        """
        print("Running training")
        print(self.activations_store)
        print(self.sae)

        if self.cfg.log_to_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                config=cast(Any, self.cfg),
                name=self.cfg.run_name,
                id=self.cfg.wandb_id,
            )

        trainer = CrossCoderTrainer(
            sae=self.sae,
            activation_store=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg,
        )

        self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.log_to_wandb:
            wandb.finish()

        return sae
