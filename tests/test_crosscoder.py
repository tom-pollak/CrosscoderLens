from pathlib import Path
from typing import Any

import pytest
import torch as t
from sae_lens.config import LanguageModelSAERunnerConfig

from crosscoder_lens.model import CrossCoder, CrossCoderConfig


def build_sae_runner_cfg(dump_dir: Path, **kwargs: Any) -> LanguageModelSAERunnerConfig:
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    mock_config_dict = {
        "use_cached_activations": True,
        "cached_activations_path": str(dump_dir / "activations"),
        "hook_name": "blocks.0.hook_mlp_out",
        "hook_layer": 0,
        "d_in": 64,
        "l1_coefficient": 2e-3,
        "lp_norm": 1,
        "lr": 2e-4,
        "training_tokens": 1_000_000,
        "train_batch_size_tokens": 4,
        "feature_sampling_window": 50,
        "dead_feature_threshold": 1e-7,
        "dead_feature_window": 1000,
        "store_batch_size_prompts": 4,
        "log_to_wandb": False,
        "wandb_project": "test_project",
        "wandb_entity": "test_entity",
        "wandb_log_frequency": 10,
        "device": "cpu",
        "seed": 24,
        "checkpoint_path": str(dump_dir / "checkpoints"),
        "dtype": "float32",
        "prepend_bos": True,
    }
    mock_config_dict.update(kwargs)
    mock_config = LanguageModelSAERunnerConfig(**mock_config_dict)
    return mock_config


@pytest.fixture
def sae_runner_cfg(tmp_path: Path):
    cfg = build_sae_runner_cfg(
        tmp_path,
        d_in=64,
        d_sae=128,
        hook_layer=0,
    )
    return cfg


@pytest.fixture
def model(sae_runner_cfg):
    return CrossCoder(sae_runner_cfg)


def test_crosscoder_init(model, config):
    """Test CrossCoder initialization"""
    assert model.W_enc.shape == (config.d_in, config.d_sae)
    assert model.b_enc.shape == (config.d_sae,)
    assert model.W_dec.shape == (config.n_models, config.d_sae, config.d_in)


def test_crosscoder_encode(model, config):
    """Test encoding step"""
    batch_size = 2
    x = t.randn(batch_size, config.d_in)

    # Test encode
    z = model.encode(x)
    assert z.shape == (batch_size, config.d_sae)

    # Test encode_with_hidden_pre
    z, hidden = model.encode_with_hidden_pre(x)
    assert z.shape == (batch_size, config.d_sae)
    assert hidden.shape == (batch_size, config.d_sae)


def test_crosscoder_decode(model, config):
    """Test decoding step"""
    batch_size = 2
    z = t.randn(batch_size, config.d_sae)

    # Test decode for specific model
    recon = model.decode(z, model_idx=0)
    assert recon.shape == (batch_size, config.d_in)

    # Test decode for all models
    recon = model.decode(z)
    assert recon.shape == (config.n_models, batch_size, config.d_in)


def test_crosscoder_forward(model, config):
    """Test full forward pass"""
    batch_size = 2
    x = t.randn(batch_size, config.d_in)

    # Test forward for specific model
    recon = model(x, model_idx=0)
    assert recon.shape == (batch_size, config.d_in)

    # Test forward for all models
    recon = model(x)
    assert recon.shape == (config.n_models, batch_size, config.d_in)


def test_crosscoder_loss(model, config):
    """Test loss computation"""
    batch_size = 2
    x = t.randn(batch_size, config.d_in)

    # Get reconstructions for all models
    recons = model(x)

    # Compute MSE loss
    mse_loss = t.mean((recons - x.unsqueeze(0)) ** 2)
    assert mse_loss.ndim == 0  # Scalar

    # Compute L1 loss
    z = model.encode(x)
    l1_loss = config.l1_coeff * t.mean(t.abs(z))
    assert l1_loss.ndim == 0  # Scalar

    # Total loss
    loss = mse_loss + l1_loss
    assert not t.isnan(loss)
    assert not t.isinf(loss)
