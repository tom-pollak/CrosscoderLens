import pytest
import torch as t
from pathlib import Path
from crosscoder_lens.model import CrossCoder, CrossCoderConfig


@pytest.fixture(autouse=True)
def setup_torch():
    """Set torch settings for reproducible tests"""
    t.manual_seed(42)
    t.set_grad_enabled(True)
    yield
    t.set_grad_enabled(True)


@pytest.fixture
def device():
    """Use CPU for testing"""
    return "cpu"


@pytest.fixture
def d_in():
    """Input dimension"""
    return 8


@pytest.fixture
def d_sae():
    """SAE dimension"""
    return 4


@pytest.fixture
def n_models():
    """Number of models to compare"""
    return 3


@pytest.fixture
def batch_size():
    """Batch size for testing"""
    return 2
