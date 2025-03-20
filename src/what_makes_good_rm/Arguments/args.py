import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from transformers import TrainingArguments, GenerationConfig
from trl import RLOOConfig, RewardConfig, SFTConfig, PPOConfig


@dataclass
class BaseHelperTraining:

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], exp_dir_prefix: str = "", seed: int = -1, run_id: int = -1):
        """Creates an TrainingArguments subclass instance from a config dictionary."""
        algorithm_name = config_dict.get('algorithm_name')
        algo_args_class = cls.get_algo_args_class(algorithm_name)
        algo_specific_config = config_dict.get('algo_specific', {})

        output_dir_base = config_dict.get('output_dir_base', 'outputs')
        exp_dir_name = f"{exp_dir_prefix}{cls.get_exp_dir_name(config_dict, seed)}"
        timestamp_str = datetime.now(timezone.utc).strftime("%Y_%m_%d-%H_%M_%S")

        output_dir_name = f"{exp_dir_name}_{timestamp_str}_rid_{run_id}" if run_id >= 0 else f"{exp_dir_name}_{timestamp_str}"
        output_dir = os.path.join(output_dir_base, output_dir_name)
        config_dict['output_dir'] = output_dir
        config_dict['logging_dir'] = output_dir

        algo_args_fields = algo_args_class.__dataclass_fields__

        # Overwrite overlapping keys in algo_specific_config with those from config_dict
        for key in algo_args_fields:
            if key in config_dict:
                algo_specific_config[key] = config_dict[key]

        # Instantiate algo_specific arguments
        algo_specific = algo_args_class(**algo_specific_config)

        args_config = {
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__ and k != 'algo_specific' and k != 'output_dir'
        }
        args = cls(output_dir, algo_specific=algo_specific, **args_config)
        return args

    def to_dict(self):
        return {k: str(v) for k, v in asdict(self).items() if not k.startswith('_')}


@dataclass
class CustomSFTTrainingArguments(TrainingArguments, BaseHelperTraining):
    output_dir_base: str = field(
        default="outputs",
        metadata={
            "help": (
                "Output folder"
            )
        },
    )
    dataset_path: str = field(
        default="",
        metadata={
            "help": (
                "Dataset to use for SFT trainings"
            )
        },
    )
    load_dataset_from_file: bool = field(
        default=False,
        metadata={
            "help": (
                "Load dataset from a disk file"
            )
        },
    )
    pretrained_model_path: str = field(
        default="",
        metadata={
            "help": (
                "Path to the pretrained model to be trained for SFT"
            )
        },
    )
    algorithm_name: str = field(
        default="sft",
        metadata={
            "help": (
                "Name of algo to use"
            ),
            "choices": ['sft']
        }
    )
    algo_specific: Optional[TrainingArguments] = field(
        default=None,
        metadata={
            "help": "Algorithm-specific arguments"
        }
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust remote code or not"
            )
        },
    )
    dataset_train_split: str = field(
        default="",
        metadata={
            "help": (
                "Name of the training split in the dataset"
            )
        },
    )
    dataset_test_split: str = field(
        default="",
        metadata={
            "help": (
                "Name of the test split in the dataset"
            )
        },
    )
    dataset_text_field: str = field(
        default="chosen",
        metadata={
            "help": (
                "Name of field to take data for SFT. Not used if AlpacaFarm is the dataset"
            )
        },
    )
    num_train_samples: int = field(
        default=-1,
        metadata={
            "help": (
                "Number of training samples to use (< 0 means all)"
            )
        },
    )
    save_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save the model after training"
            )
        },
    )

    @staticmethod
    def get_exp_dir_name(config_dict: dict, seed: int = -1):
        return f"sft_seed_{seed}"

    @staticmethod
    def get_algo_args_class(algorithm_name: str):
        """Returns the appropriate AlgoArgs subclass based on algorithm_name."""
        return SFTConfig


@dataclass
class CustomPolicyGradientTrainingArguments(TrainingArguments, BaseHelperTraining):
    output_dir_base: str = field(
        default="outputs",
        metadata={
            "help": (
                "Output folder"
            )
        },
    )
    dataset_path: str = field(
        default="",
        metadata={
            "help": (
                "Dataset to use for RLHF trainings"
            )
        },
    )
    load_dataset_from_file: bool = field(
        default=False,
        metadata={
            "help": (
                "Load dataset from a disk file"
            )
        },
    )
    num_train_samples: int = field(
        default=-1,
        metadata={
            "help": (
                "Number of training samples to use (< 0 means all)"
            )
        },
    )
    data_selection_seed: int = field(
        default=-1,
        metadata={
            "help": (
                "Random seed for data selection (< 0 means no seed is used)"
            )
        },
    )
    dataset_train_split: str = field(
        default="",
        metadata={
            "help": (
                "Name of the training split in the dataset"
            )
        },
    )
    dataset_test_split: str = field(
        default="",
        metadata={
            "help": (
                "Name of the test split in the dataset"
            )
        },
    )
    dataset_num_proc: int = field(
        default=None,
        metadata={
            "help": (
                "Number of processes to use for processing the dataset."
            )
        },
    )
    algorithm_name: str = field(
        default="",
        metadata={
            "help": (
                "Name of algo to use"
            ),
            "choices": ['RLOO', 'PPO', 'GRPO']
        },
    )
    reward_model_path: str = field(
        default="",
        metadata={
            "help": (
                "Name of the reward model to use, either on HF or local"
            )
        },
    )
    use_reward_model_tokenizer: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use the reward model's tokenizer instead of sharing language model and reward model tokenizers"
            )
        },
    )
    path_to_precomputed_rewards_for_normalization: str = field(
        default="",
        metadata={
            "help": (
                "Path to precomputed rewards, to be used for normalizing the rweards produced by the reward model"
            )
        }
    )
    language_model_path: str = field(
        default="",
        metadata={
            "help": (
                "Name of the sft model to use, either on HF or local"
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust remote code or not"
            )
        },
    )
    algo_specific: Optional[TrainingArguments] = field(
        default=None,
        metadata={
            "help": "Algorithm-specific arguments"
        }
    )
    eval_final_policy_at_end: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to run evaluation of the learned policy right after the end of training"
            )
        }
    )
    delete_language_model_checkpoint_after_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to delete the language model checkpoints after their evaluation. "
                "Use with CAUTION! Intended mostly for automatic evaluation scripts "
                "of multiple runs in order to save disk space taken by large checkpoints."
            )
        },
    )

    @staticmethod
    def get_exp_dir_name(config_dict: dict, seed: int = -1):
        rm_name_parts = config_dict["reward_model_path"].split("_")
        frac_first_indices = [i for i, name_part in enumerate(rm_name_parts) if "frac-first" == name_part]
        if len(frac_first_indices) > 0:
            frac_first = rm_name_parts[frac_first_indices[0] + 1]
            frac_first_str = f"_frac-first_{frac_first}"
        else:
            frac_first_str = ""

        frac_scaled_str = ""
        if config_dict['algo_specific'].get('scale_reward_for_frac_prompts', 0) > 0:
            frac_scaled_str = (f"_frac_scaled_{config_dict['algo_specific']['scale_reward_for_frac_prompts']}_"
                               f"scale_{config_dict['algo_specific']['reward_scale_factor']}")

        initial_str = f"{config_dict['algorithm_name']}_seed_{seed}_"
        return initial_str + (f"epochs_{config_dict['num_train_epochs']}"
                              f"{frac_first_str}{frac_scaled_str}")

    @staticmethod
    def get_algo_args_class(algorithm_name: str):
        """Returns the appropriate AlgoArgs subclass based on algorithm_name."""
        if algorithm_name == 'RLOO':
            return CustomRLOOConfig
        elif algorithm_name == 'PPO':
            return CustomPPOConfig
        elif algorithm_name == 'GRPO':
            return CustomGRPOConfig
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")


@dataclass
class CustomRLOOConfig(RLOOConfig):
    early_stopping: bool = field(default=False, metadata={"help": "Whether to early stop"})
    target_kl: float = field(default=0.1, metadata={"help": "KL target for early stopping"})
    per_prompt_normalize_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to normalize rewards per prompt by dividing by reward standard deviation estimate"}
    )
    scale_reward_for_frac_prompts: float = field(
        default=0.0,
        metadata={"help": "Fraction of prompts to scale their reward by a constant"}
    )
    scale_reward_prompt_selection_seed: int = field(
        default=-1,
        metadata={"help": "Random seed for selecting prompts to scale their reward (if < 0 no seed is used)"}
    )
    reward_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Factor to scale the rewards of the fraction of prompts specified by 'scale_reward_for_frac_prompts'"}
    )
    whiten_rewards_without_kl: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the rewards without including the KL term, as opposed to with the KL term as done in the default "
                          "TRL implementation. This flag is only relevant if 'whiten_rewards' is True. "
                          "Otherwise, will not whiten rewards across the batch."}
    )


@dataclass
class CustomPPOConfig(PPOConfig):
    scale_reward_for_frac_prompts: float = field(
        default=0.0,
        metadata={"help": "Fraction of prompts to scale their reward by a constant"}
    )
    scale_reward_prompt_selection_seed: int = field(
        default=-1,
        metadata={"help": "Random seed for selecting prompts to scale their reward (if < 0 no seed is used)"}
    )
    reward_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Factor to scale the rewards of the fraction of prompts specified by 'scale_reward_for_frac_prompts'"}
    )
    whiten_rewards_without_kl: bool = field(
        default=False,
        metadata={"help": "Whether to whiten the rewards without including the KL term, as opposed to with the KL term as done in the default "
                          "TRL implementation. This flag is only relevant if 'whiten_rewards' is True. "
                          "Otherwise, will not whiten rewards across the batch."}
    )


@dataclass
class CustomGRPOConfig(RLOOConfig):
    scale_reward_for_frac_prompts: float = field(
        default=0.0,
        metadata={"help": "Fraction of prompts to scale their reward by a constant"}
    )
    scale_reward_prompt_selection_seed: int = field(
        default=-1,
        metadata={"help": "Random seed for selecting prompts to scale their reward (if < 0 no seed is used)"}
    )
    reward_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Factor to scale the rewards of the fraction of prompts specified by 'scale_reward_for_frac_prompts'"}
    )


@dataclass
class CustomRMTrainingArguments(TrainingArguments, BaseHelperTraining):
    output_dir_base: str = field(
        default="outputs",
        metadata={
            "help": (
                "Output folder"
            )
        },
    )
    dataset_path: str = field(
        default="",
        metadata={
            "help": (
                "Path to the preference dataset"
            )
        },
    )
    second_dataset_path: str = field(
        default="",
        metadata={
            "help": (
                "Path to a second preference dataset"
            )
        },
    )
    load_dataset_from_file: bool = field(
        default=False,
        metadata={
            "help": (
                "Load dataset from a disk file"
            )
        },
    )
    frac_samples_from_first_dataset: float = field(
        default=1.0,
        metadata={
            "help": (
                "Fraction of samples to take from the first dataset"
            )
        },
    )
    num_train_samples: int = field(
        default=-1,
        metadata={
            "help": (
                "Number of training samples to use (< 0 means all)"
            )
        },
    )
    data_selection_seed: int = field(
        default=-1,
        metadata={
            "help": (
                "Random seed for data selection (< 0 means no seed is used)"
            )
        },
    )
    pretrained_model_path: str = field(
        default="",
        metadata={
            "help": (
                "Path to the pretrained model to be trained for reward modelling"
            )
        },
    )
    save_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save the model after training"
            )
        },
    )
    algo_specific: Optional[TrainingArguments] = field(
        default=None,
        metadata={
            "help": "Algorithm-specific arguments"
        }
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust remote code or not"
            )
        },
    )

    @staticmethod
    def get_exp_dir_name(config_dict: dict, seed: int = -1):
        initial_str = f"rm_seed_{seed}_"
        return initial_str + (f"epochs_{config_dict['num_train_epochs']}_"
                              f"frac-first_{config_dict['frac_samples_from_first_dataset']}_"
                              f"dataseed_{config_dict['data_selection_seed']}")

    @staticmethod
    def get_algo_args_class(algorithm_name: str):
        """Returns the appropriate AlgoArgs subclass based on algorithm_name."""
        return RewardConfig


@dataclass
class CustomRMEvalArguments:
    dataset_path: str
    language_model_path: str
    ground_truth_reward_model_path: str
    proxy_reward_models: List[str]
    cache_dir: str = field(
        default=None,
        metadata={
            "help": (
                "Directory to the Hugging Face cache for loading models and datasets"
            )
        },
    )
    output_dir_base: str = field(
        default="outputs",
        metadata={
            "help": (
                "Output folder"
            )
        },
    )
    output_dir: str = field(
        default="outputs",
        metadata={
            "help": (
                "Output folder"
            )
        },
    )
    load_dataset_from_file: bool = field(
        default=False,
        metadata={
            "help": (
                "Load dataset from a disk file"
            )
        },
    )
    generation_config: GenerationConfig = field(default=None)
    save_generated_responses: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save the full generated responses."
            )
        },
    )
    rm_batch_size: int = field(default=8)
    lm_batch_size: int = field(default=8)
    delete_language_model_checkpoint_after_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to delete the language model checkpoint after evaluation. Use with CAUTION! Intended mostly for automatic evaluation scripts "
                "of multiple runs in order to save disk space taken by large checkpoints."
            )
        },
    )
    only_lm_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "When script is used only for evaluating the rewards obtained by a language model, "
                "will not compute ranking accuracies of the reward models or save rewards on the offline responses in the dataset"
            )
        },
    )
    train_split: str = field(default="train_prefs")
    test_split: str = field(default="test_prefs")
    num_train_samples: int = field(
        default=-1,
        metadata={
            "help": (
                "Number of training samples to use (< 0 means all)"
            )
        },
    )
    num_test_samples: int = field(
        default=-1,
        metadata={
            "help": (
                "Number of test samples to use (< 0 means all)"
            )
        },
    )
    data_selection_seed: int = field(
        default=-1,
        metadata={
            "help": (
                "Random seed for data selection (< 0 means no seed is used)"
            )
        },
    )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], exp_dir_prefix: str = "", seed: int = -1, cache_dir: str = None, run_id: int = -1):
        args = cls(**config_dict)
        args.cache_dir = cache_dir

        if "output_dir" not in config_dict:
            initial_output_dir_str = f"eval_seed_{seed}_data_seed_{args.data_selection_seed}"
            if args.only_lm_eval:
                initial_output_dir_str = "lm_only_" + initial_output_dir_str

            timestamp_str = datetime.now(timezone.utc).strftime("%Y_%m_%d-%H_%M_%S")
            output_dir_name = f"{exp_dir_prefix}{initial_output_dir_str}_{timestamp_str}_rid_{run_id}" if run_id >= 0 \
                else f"{exp_dir_prefix}{initial_output_dir_str}_{timestamp_str}"
            args.output_dir = os.path.join(args.output_dir_base, output_dir_name) if args.output_dir_base is not None else ""
        else:
            args.output_dir = config_dict["output_dir"]

        return args


@dataclass
class CustomTrainingArguments:
    seed: int = field(
        default=42,
        metadata={
            "help": (
                "Random seed for reproducibility"
            )
        },
    )
    cache_dir: str = field(
        default=None,
        metadata={
            "help": (
                "Directory to the Hugging Face cache for loading models and datasets"
            )
        },
    )
    do_rm: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to train your custom reward model"
            )
        },
    )
    do_sft: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to do sft before policy gradient algorithm"
            )
        },
    )

    do_pg: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to do rlhf"
            )
        },
    )

    do_rm_eval: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to do reward model evaluation"
            )
        },
    )

    exp_dir_prefix: str = field(
        default="",
        metadata={
            "help": (
                "Prefix for the experiment directory"
            )
        }
    )

    sft_config: CustomSFTTrainingArguments = field(
        default=None,
        metadata={
            "help": (
                "Config for SFT training"
            )
        },
    )

    pg_config: CustomPolicyGradientTrainingArguments = field(
        default=None,
        metadata={
            "help": (
                "Config for RLHF training"
            )
        },
    )
    rm_config: CustomRMTrainingArguments = field(
        default=None,
        metadata={
            "help": (
                "Config for Reward model training"
            )
        },
    )

    rm_eval_config: CustomRMEvalArguments = field(
        default=None,
        metadata={
            "help": (
                "Config for Reward model evaluation"
            )
        },
    )
