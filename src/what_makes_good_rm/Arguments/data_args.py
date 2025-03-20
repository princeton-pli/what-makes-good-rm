from dataclasses import dataclass, field
from typing import Dict, Any

from transformers import GenerationConfig


@dataclass
class DatasetReLabelArguments:
    initial_dataset_path: str
    reward_model_path: str
    language_model_path: str = field(default=None)
    filter_equals: bool = field(default=False,
                                metadata={
                                    "help": (
                                        "Whether to filter out rows where score_chosen and score_rejected are the same"
                                    )
                                })
    generation_config: GenerationConfig = field(default=None)
    tokenizer_for_length_filtering: str = field(default=None)
    max_prompt_length: int = field(default=-1)
    max_response_length: int = field(default=-1)
    rm_batch_size: int = field(default=8)
    cache_dir: str = field(default=None)
    train_split: str = field(default="train_prefs")
    test_split: str = field(default="test_prefs")
    gpu_id: int = field(default=0)
    frac_train_for_rm: float = field(default=0.8)
    train_split_seed: int = field(default=-1)
    push_to_hub: bool = field(default=False)
    output_dir: str = field(default=None)
    private: bool = field(default=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        args = cls(**config_dict)
        return args

@dataclass
class DatasetArguments:
    seed: int = field(default=-1)
    relabel_config: DatasetReLabelArguments = field(
        default=None,
        metadata={
            "help": (
                "Config for relabelling data"
            )
        },
    )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        args = cls(**config_dict)
        return args
