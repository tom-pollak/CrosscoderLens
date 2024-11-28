import torch as t

from .helpers import model


def test_crosscoder_init(model):
    """Test CrossCoder initialization"""
    cfg = model.cfg
    assert model.W_enc.shape == (cfg.n_models, cfg.d_in, cfg.d_sae)
    assert model.b_enc.shape == (cfg.d_sae,)
    assert model.W_dec.shape == (cfg.d_sae, cfg.n_models, cfg.d_in)
    assert model.b_dec.shape == (cfg.n_models, cfg.d_in)


def test_crosscoder_encode(model):
    """Test encoding step"""
    batch_size = 2
    cfg = model.cfg
    x = t.randn(batch_size, cfg.n_models, cfg.d_in)

    # Test encode
    z = model.encode(x)
    assert z.shape == (batch_size, cfg.d_sae)

    # Test encode_with_hidden_pre
    z, hidden = model.encode_with_hidden_pre_fn(x)
    assert z.shape == (batch_size, cfg.d_sae)
    assert hidden.shape == (batch_size, cfg.d_sae)


def test_crosscoder_decode(model):
    """Test decoding step"""
    batch_size = 2
    cfg = model.cfg
    z = t.randn(batch_size, cfg.d_sae)

    # Test decode for all models
    recon = model.decode(z)
    assert recon.shape == (batch_size, cfg.n_models, cfg.d_in)


def test_crosscoder_forward(model):
    """Test full forward pass"""
    batch_size = 2
    cfg = model.cfg
    x = t.randn(batch_size, cfg.n_models, cfg.d_in)

    recon = model(x)
    assert recon.shape == (batch_size, cfg.n_models, cfg.d_in)


def test_crosscoder_loss(model):
    """Test loss computation"""
    batch_size = 2
    cfg = model.cfg
    x = t.randn(batch_size, cfg.n_models, cfg.d_in)

    # Get reconstructions for all models
    recons = model(x)

    # Compute MSE loss
    mse_loss = t.mean((recons - x) ** 2)
    assert mse_loss.ndim == 0  # Scalar

    # Compute L1 loss
    z = model.encode(x)
    l1_loss = cfg.l1_coefficient * t.mean(t.abs(z))
    assert l1_loss.ndim == 0  # Scalar

    # Total loss
    loss = mse_loss + l1_loss
    assert not t.isnan(loss)
    assert not t.isinf(loss)
