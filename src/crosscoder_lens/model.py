from dataclasses import asdict, dataclass

import einops
import torch as t
from jaxtyping import Float
from sae_lens.training.training_sae import TrainingSAE, TrainingSAEConfig

from .trainer import CrossCoderTrainerConfig


@dataclass(kw_only=True)
class CrossCoderConfig(TrainingSAEConfig):
    n_models: int  # Number of models to compare

    def __post_init__(self):
        assert (
            self.architecture == "standard"
        ), "CrossCoder only supports standard architecture"

    @classmethod
    def from_sae_runner_config(cls, cfg: CrossCoderTrainerConfig) -> "CrossCoderConfig":
        return cls(n_models=cfg.n_models, **asdict(super().from_sae_runner_config(cfg)))


class CrossCoder(TrainingSAE):
    """
    CrossCoder model that learns shared features across different model checkpoints.
    Inherits most functionality from TrainingSAE, only modifies decoder to handle multiple models.
    """

    def __init__(self, config: CrossCoderConfig):
        super().__init__(config)
        self.encode = self.encode_crosscoder

    def encode_crosscoder(
        self, x: Float[t.Tensor, "batch model d_in"]
    ) -> Float[t.Tensor, "batch model d_sae"]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
            )
            + self.b_enc
        )
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts

    def decode(
        self, feature_acts: Float[t.Tensor, "batch d_sae"]
    ) -> Float[t.Tensor, "batch model d_in"]:
        sae_out = self.hook_sae_recons(
            einops.einsum(
                self.apply_finetuning_scaling_factor(feature_acts),
                self.W_dec,
                "batch d_sae, d_sae model d_in -> batch model d_in",
            )
            + self.b_dec
        )

        # handle run time activation normalization if needed
        # will fail if you call this twice without calling encode in between.
        sae_out = self.run_time_activation_norm_fn_out(sae_out)

        # handle hook z reshaping if needed.
        sae_out = self.reshape_fn_out(sae_out, self.d_head)  # type: ignore

        return sae_out
