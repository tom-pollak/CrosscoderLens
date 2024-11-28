from dataclasses import asdict
from pathlib import Path
from typing import Any

import datasets
import pytest
import torch as t
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.store.cache_activations_runner import (
    CacheActivationsRunner,
    CacheActivationsRunnerConfig,
)
from sae_lens.store.cached_activation_store import CachedActivationsStore

from crosscoder_lens.config import CrossCoderTrainerConfig, TrainingCrosscoderConfig
from crosscoder_lens.model import CrossCoder

N_MODELS = 2
SAE_BATCH_SIZE = 16
CONTEXT_SIZE = 16

def build_sae_runner_cfg(dump_dir: Path, **kwargs: Any) -> LanguageModelSAERunnerConfig:
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    mock_config_dict = {
        "use_cached_activations": True,
        "cached_activations_path": str(dump_dir / "activations"),
        "hook_name": "blocks.0.hook_mlp_out",
        "hook_layer": 0,
        "d_in": 512,
        "expansion_factor": 4,
        "l1_coefficient": 2e-3,
        "lp_norm": 1,
        "lr": 2e-4,
        "training_tokens": 10000,
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
        "prepend_bos": False,
        "normalize_sae_decoder": False,
    }
    mock_config_dict.update(kwargs)
    mock_config = LanguageModelSAERunnerConfig(**mock_config_dict)
    return mock_config


def build_crosscoder_training_cfg(
    dump_dir: Path, n_models: int, **kwargs: Any
) -> CrossCoderTrainerConfig:
    sae_runner_cfg = build_sae_runner_cfg(dump_dir, **kwargs)
    return CrossCoderTrainerConfig(n_models=n_models, **asdict(sae_runner_cfg))


def build_crosscoder_cfg(
    dump_dir: Path, n_models: int, **kwargs: Any
) -> TrainingCrosscoderConfig:
    training_cfg = build_crosscoder_training_cfg(dump_dir, n_models, **kwargs)
    return TrainingCrosscoderConfig.from_crosscoder_trainer_config(training_cfg)


@pytest.fixture
def model(tmp_path: Path) -> CrossCoder:
    cfg = build_crosscoder_cfg(tmp_path, N_MODELS)
    return CrossCoder(cfg)


def mk_dataset(
    tmp_path: Path,
) -> datasets.Dataset:
    total_training_tokens = 10000
    batch_size = 256
    d_in = 512
    dtype = "float32"
    device = (
        "cuda"
        if t.cuda.is_available()
        else "mps"
        if t.backends.mps.is_available()
        else "cpu"
    )

    def _create_cfg(hook_name: str):
        return CacheActivationsRunnerConfig(
            new_cached_activations_path=str(tmp_path / "activations" / hook_name),
            dataset_path="chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests",
            model_name="gelu-1l",
            hook_name=hook_name,
            hook_layer=0,
            buffer_size_gb=0.02,  # 20MB
            ### Parameters
            training_tokens=total_training_tokens,
            model_batch_size=batch_size,
            context_size=CONTEXT_SIZE,
            ###
            d_in=d_in,
            shuffle=False,
            prepend_bos=False,
            device=device,
            seed=42,
            dtype=dtype,
        )

    dataset_pre = CacheActivationsRunner(_create_cfg("blocks.0.hook_resid_pre")).run()
    dataset_post = CacheActivationsRunner(_create_cfg("blocks.0.hook_resid_post")).run()
    dataset_pre = dataset_pre.rename_column("blocks.0.hook_resid_pre", "pre")
    dataset_post = dataset_post.rename_column("blocks.0.hook_resid_post", "post")
    return datasets.concatenate_datasets([dataset_pre, dataset_post], axis=1)


@pytest.fixture
def crosscoder_trainer_cfg(tmp_path: Path) -> CrossCoderTrainerConfig:
    return build_crosscoder_training_cfg(
        tmp_path,
        n_models=N_MODELS,
    )


@pytest.fixture
def cached_activations_store(tmp_path: Path) -> CachedActivationsStore:
    dataset = mk_dataset(tmp_path)
    return CachedActivationsStore(
        ds=dataset,
        column_names=["pre", "post"],
        batch_size=SAE_BATCH_SIZE,
        context_size=CONTEXT_SIZE,
    )
