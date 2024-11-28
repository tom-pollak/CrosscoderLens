from dataclasses import asdict, dataclass

import einops
import torch as t
from jaxtyping import Float
from sae_lens.training.training_sae import TrainingSAE

from .config import TrainingCrosscoderConfig


class CrossCoder(TrainingSAE):
    """
    CrossCoder model that learns shared features across different model checkpoints.
    Inherits most functionality from TrainingSAE, only modifies decoder to handle multiple models.

    Could I use head_idx to handle multiple models?
    """

    cfg: TrainingCrosscoderConfig

    def __init__(self, config: TrainingCrosscoderConfig):
        super().__init__(config)
        self.encode = self.encode_crosscoder
        self.encode_with_hidden_pre_fn = self.encode_with_hidden_pre_crosscoder

        # d_in, d_sae -> n_models, d_in, d_sae
        self.W_enc.data = self.W_enc.data.unsqueeze(0).expand(self.cfg.n_models, -1, -1).clone()
        # d_sae, d_in -> d_sae, n_models, d_in
        self.W_dec.data = self.W_dec.data.unsqueeze(1).expand(-1, self.cfg.n_models, -1).clone()
        # d_in -> n_models, d_in
        self.b_dec.data = self.b_dec.data.unsqueeze(0).expand(self.cfg.n_models, -1).clone()

    def encode_crosscoder(
        self, x: Float[t.Tensor, "batch model d_in"]
    ) -> Float[t.Tensor, "batch model d_sae"]:
        feature_acts, _ = self.encode_with_hidden_pre_fn(x)
        return feature_acts

    def encode_with_hidden_pre_crosscoder(
        self, x: Float[t.Tensor, "batch model d_in"]
    ) -> tuple[
        Float[t.Tensor, "batch model d_sae"], Float[t.Tensor, "batch model d_sae"]
    ]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
            )
            + self.b_enc
        )
        hidden_pre_noised = hidden_pre + (
            t.randn_like(hidden_pre) * self.cfg.noise_scale * self.training
        )
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre_noised))
        return feature_acts, hidden_pre_noised


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
