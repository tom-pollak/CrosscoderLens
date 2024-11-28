from pathlib import Path

import pytest
import torch as t

from crosscoder_lens.model import CrossCoder, CrossCoderConfig
from crosscoder_lens.trainer import CrossCoderTrainer


@pytest.fixture
def config():
    return CrossCoderConfig(d_in=8, d_sae=4, n_models=3, l1_coeff=0.1, device="cpu")


@pytest.fixture
def model(config):
    return CrossCoder(config)


@pytest.fixture
def mock_activation_dirs(tmp_path):
    """Create mock activation directories with dummy data"""
    n_models = 3
    dirs = []
    for i in range(n_models):
        model_dir = tmp_path / f"model_{i}"
        hook_dir = model_dir / "blocks.4.hook_resid_post"
        hook_dir.mkdir(parents=True)

        # Create dummy activation data
        data = t.randn(10, 8)  # 10 samples, d_in=8
        t.save({"acts": data}, hook_dir / "buffer_0.pt")
        dirs.append(model_dir)

    return dirs


@pytest.fixture
def trainer(model, mock_activation_dirs):
    return CrossCoderTrainer(
        sae=model,
        activation_dirs=mock_activation_dirs,
        hook_name="blocks.4.hook_resid_post",
        batch_size=2,
    )


def test_trainer_init(trainer, model, mock_activation_dirs):
    """Test trainer initialization"""
    assert trainer.sae == model
    assert len(trainer.stores) == len(mock_activation_dirs)
    assert trainer.optimizer is not None


def test_train_step(trainer, config):
    """Test single training step"""
    # Get batch from first store
    batch = next(trainer.stores[0].dl_it)

    # Run training step
    output = trainer._train_step(batch)

    # Check output shapes
    assert output.sae_in.shape[-1] == config.d_in
    assert output.sae_out.shape[-1] == config.d_sae
    assert output.hidden_pre.shape[-1] == config.d_sae

    # Check losses
    assert "loss" in output.losses
    assert "mse_loss" in output.losses
    assert "l1_loss" in output.losses
    assert "sparsity" in output.losses

    # Check loss values
    for loss_name, loss_val in output.losses.items():
        assert isinstance(loss_val, float)
        assert not t.isnan(t.tensor(loss_val))
        assert not t.isinf(t.tensor(loss_val))

        # stores = []
        # for path in activation_dirs:
        #     if not path.exists():
        #         raise ValueError(f"Activation path {path} does not exist")
        #     store = CachedActivationsStore(
        #         activation_save_path=path,
        #         column_names=[hook_name],
        #         batch_size=batch_size
        #     )
        #     stores.append(store)
