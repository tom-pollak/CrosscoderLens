import pytest
import torch as t

from crosscoder_lens.trainer import CrossCoderTrainingRunner

from .helpers import cached_activations_store, crosscoder_trainer_cfg


@pytest.fixture
def train_runner(crosscoder_trainer_cfg, cached_activations_store):
    return CrossCoderTrainingRunner(
        cfg=crosscoder_trainer_cfg,
        activation_store=cached_activations_store,
    )


def test_trainer_run(train_runner):
    """Test trainer run"""
    train_runner.run()
