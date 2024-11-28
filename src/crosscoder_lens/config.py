import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any

from sae_lens import LanguageModelSAERunnerConfig, TrainingSAEConfig


@dataclass(kw_only=True)
class CrossCoderTrainerConfig(LanguageModelSAERunnerConfig):
    n_models: int  # Number of models to compare

    def __post_init__(self):
        # d_sae calculated in __post_init__
        if self.expansion_factor is not None and self.d_in is not None:
            self.expansion_factor = None

        super().__post_init__()
        if not self.use_cached_activations:
            warnings.warn(
                "Using cached activations for CrossCoder training.", stacklevel=2
            )
            self.use_cached_activations = True
        assert self.architecture == "standard"

    def get_training_crosscoder_cfg_dict(self) -> dict[str, Any]:
        return {**super().get_training_sae_cfg_dict(), "n_models": self.n_models}


@dataclass(kw_only=True)
class TrainingCrosscoderConfig(TrainingSAEConfig):
    n_models: int  # Number of models to compare

    def __post_init__(self):
        assert (
            self.architecture == "standard"
        ), "CrossCoder only supports standard architecture"
        assert not self.normalize_sae_decoder, "normalize_sae_decoder must be False"

    @classmethod
    def from_crosscoder_trainer_config(
        cls, cfg: CrossCoderTrainerConfig
    ) -> "TrainingCrosscoderConfig":
        # First get the base config as TrainingSAEConfig
        base_config = TrainingSAEConfig.from_sae_runner_config(cfg)
        # Then create CrossCoderConfig with base config fields + n_models
        return cls(n_models=cfg.n_models, **asdict(base_config))

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingCrosscoderConfig":
        # remove any keys that are not in the dataclass
        # since we sometimes enhance the config with the whole LM runner config
        valid_field_names = {field.name for field in fields(cls)}
        valid_config_dict = {
            key: val for key, val in config_dict.items() if key in valid_field_names
        }

        # ensure seqpos slice is tuple
        # ensure that seqpos slices is a tuple
        # Ensure seqpos_slice is a tuple
        if "seqpos_slice" in valid_config_dict:
            if isinstance(valid_config_dict["seqpos_slice"], list):
                valid_config_dict["seqpos_slice"] = tuple(
                    valid_config_dict["seqpos_slice"]
                )
            elif not isinstance(valid_config_dict["seqpos_slice"], tuple):
                valid_config_dict["seqpos_slice"] = (valid_config_dict["seqpos_slice"],)

        return cls(**valid_config_dict)
