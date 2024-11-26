from dataclasses import dataclass

import einops
import torch as t
from jaxtyping import Float
from sae_lens.training.training_sae import TrainingSAE, TrainingSAEConfig


@dataclass(kw_only=True)
class CrossCoderConfig(TrainingSAEConfig):
    n_models: int  # Number of models to compare


class CrossCoder(TrainingSAE):
    """
    CrossCoder model that learns shared features across different model checkpoints.
    Inherits most functionality from TrainingSAE, only modifies decoder to handle multiple models.
    """

    def __init__(self, config: CrossCoderConfig):
        super().__init__(config)
        self.config = config

        # Replace single decoder with multi-model decoder
        del self.W_dec  # Remove parent's decoder
        self.W_dec = t.nn.Parameter(
            t.randn(
                config.n_models,
                config.d_sae,
                config.d_in,
                device=config.device,
                dtype=t.float32,
            )
            * config.init_scale
        )

    def decode(
        self, sae_out: Float[t.Tensor, "... d_sae"], model_idx: int | None = None
    ) -> Float[t.Tensor, "... d_in"]:
        """Decode from feature space back to activation space"""
        if model_idx is not None:
            return sae_out @ self.W_dec[model_idx]

        # Decode for all models
        # sae_out: [batch, d_sae]
        # W_dec: [n_models, d_sae, d_in]
        # returns: [n_models, batch, d_in]
        return einops.einsum(
            sae_out, self.W_dec, "... d_sae, n d_sae d_in -> n ... d_in"
        )

    def forward(
        self, x: Float[t.Tensor, "... d_in"], model_idx: int | None = None
    ) -> Float[t.Tensor, "... d_in"]:
        """Full forward pass: encode then decode"""
        sae_out = self.encode(x)
        return self.decode(sae_out, model_idx)

    def encode_with_hidden_pre(
        self, x: Float[t.Tensor, "... d_in"]
    ) -> tuple[Float[t.Tensor, "... d_sae"], Float[t.Tensor, "... d_sae"]]:
        """Get both encoder output and pre-activation values"""
        sae_in = self.process_sae_in(x)
        hidden_pre = sae_in @ self.W_enc + self.b_enc
        sae_out = self.nonlinearity(hidden_pre)
        return sae_out, hidden_pre
