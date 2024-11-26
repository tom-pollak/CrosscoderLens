import pytest
import torch as t
from crosscoder_lens.model import CrossCoder, CrossCoderConfig


@pytest.fixture
def config():
    return CrossCoderConfig(d_in=8, d_sae=4, n_models=3, l1_coeff=0.1, device="cpu")


@pytest.fixture
def model(config):
    return CrossCoder(config)


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
