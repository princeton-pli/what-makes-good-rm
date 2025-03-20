import gc
import json
from typing import Union

import numpy as np
import torch
from datasets import load_dataset, DatasetDict, IterableDatasetDict, Dataset, IterableDataset
from transformers import AutoTokenizer

from what_makes_good_rm.Data.Utils import CausalLMWrapper
from what_makes_good_rm.Data.registry import REWARD_MODEL_WRAPPER_REGISTRY
from what_makes_good_rm.Utils import DEFAULT_USER_TOKEN, DEFAULT_ASSISTANT_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_PADDING_TOKEN, update_tokenizer, \
    update_model_num_embeddings_and_special_tokens
from what_makes_good_rm.Utils import single_process_logging as logging_utils
from what_makes_good_rm.Utils.strings import DEFAULT_TRAIN_RM_SPLIT_NAME, DEFAULT_TRAIN_RLHF_SPLIT_NAME, DEFAULT_TEST_SPLIT_NAME


class PreferenceDatasetCreator:

    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.device = torch.device(f"cuda:{self.config.gpu_id}" if torch.cuda.is_available() and self.config.gpu_id >= 0 else "cpu")

        if self.config.reward_model_path:
            self.rm_wrapper = self.__get_reward_model_wrapper(self.config.reward_model_path, device=self.device, cache_dir=self.config.cache_dir)

            # If the reward model does not have a chat template, adds a default one (chat template needs to already exist)
            if not self.rm_wrapper.tokenizer.chat_template:
                logging_utils.warning("Reward model does not have a chat template. "
                                      "Adding a default one, which should only be used for debugging purposes.")
                update_tokenizer(tokenizer=self.rm_wrapper.tokenizer, num_added_toks=0, pad_token=DEFAULT_PADDING_TOKEN,
                                 eos_token=DEFAULT_EOS_TOKEN, logger=logging_utils.get_default_logger(), user_token=DEFAULT_USER_TOKEN,
                                 assistant_token=DEFAULT_ASSISTANT_TOKEN)
                self.rm_wrapper.rm.config.pad_token_id = self.rm_wrapper.tokenizer.pad_token_id
                update_model_num_embeddings_and_special_tokens(self.rm_wrapper.rm, self.rm_wrapper.tokenizer)

        self.train_test_splits = [split for split in [self.config.train_split, self.config.test_split] if split is not None]

        if self.config.tokenizer_for_length_filtering:
            self.tokenizer_for_length_filtering = AutoTokenizer.from_pretrained(self.config.tokenizer_for_length_filtering,
                                                                                use_fast=True,
                                                                                trust_remote_code=True,
                                                                                cache_dir=self.config.cache_dir)
        else:
            self.tokenizer_for_length_filtering = self.rm_wrapper.tokenizer

    @staticmethod
    def __get_reward_model_wrapper(reward_model_name: str, **kwargs):
        if reward_model_name in REWARD_MODEL_WRAPPER_REGISTRY:
            rm_wrapper = REWARD_MODEL_WRAPPER_REGISTRY[reward_model_name]()
        else:
            logging_utils.info(f"Reward model wrapper '{reward_model_name}' not found in the registry. Defaulting to general config")
            rm_wrapper = REWARD_MODEL_WRAPPER_REGISTRY["GeneralRM"]()
            kwargs["rm_str"] = reward_model_name
        rm_wrapper.prepare(**kwargs)
        return rm_wrapper

    @staticmethod
    def __get_lm_wrapper(lm_str: str, device, config):
        try:
            lm_wrapper = CausalLMWrapper(lm_str, device=device, config=config)
            lm_wrapper.prepare()
            return lm_wrapper
        except Exception as e:
            logging_utils.error(str(e))
            raise Exception(e)

    def __filter_by_length(self, example):
        prompt_not_too_long = False
        if self.config.max_prompt_length > 0:
            prompt_length = len(self.tokenizer_for_length_filtering.encode(example["prompt"]))
            prompt_not_too_long = prompt_length <= self.config.max_prompt_length

        responses_not_too_long = False
        if self.config.max_response_length > 0:
            chosen_length = len(self.tokenizer_for_length_filtering.encode(example["chosen"][1]["content"]))
            rejected_length = len(self.tokenizer_for_length_filtering.encode(example["rejected"][1]["content"]))
            responses_not_too_long = chosen_length <= self.config.max_response_length and rejected_length <= self.config.max_response_length

        return prompt_not_too_long and responses_not_too_long

    @torch.no_grad()
    def prepare_dataset(self):
        self.dataset = load_dataset(self.config.initial_dataset_path,
                                    cache_dir=self.config.cache_dir)

        logging_utils.info(f"Logging config: {self.config}")

        self.__delete_irrelevant_splits()
        logging_utils.info("Dataset properties before filtering and relabeling.")
        self.log_dataset_properties()

        if self.config.max_prompt_length > 0 or self.config.max_response_length > 0:
            for split in self.train_test_splits:
                self.dataset[split] = self.dataset[split].filter(self.__filter_by_length)

            logging_utils.info(f"Dataset properties after filtering out samples with prompts longer than {self.config.max_prompt_length} tokens, "
                               f"or responses longer than {self.config.max_response_length} tokens")
            self.log_dataset_properties()

        # Generate pairs of responses from given language model
        if self.config.language_model_path:
            self.__replace_existing_responses_with_onpolicy_generations()

        # Compute rewards
        if self.config.reward_model_path:
            self.dataset = self.__updated_dataset_w_computed_rewards(batch_size=self.config.rm_batch_size)

        self.__create_dataset_splits_for_rm_training_and_rlhf()
        self.log_dataset_properties()
        return self.dataset

    def log_dataset_properties(self):
        if self.dataset is None:
            logging_utils.info("Dataset has not yet been initialized")

        num_rows = {k: f"num of rows: {self.dataset[k].num_rows}" for k in self.dataset.keys()}
        score_means = {k: [f"mean of {l}: {np.mean(self.dataset[k][l])}" for l in ['score_chosen', 'score_rejected']] for k in self.dataset.keys()}
        response_lengths = {
            k: [
                f"mean chosen response length: "
                f"{np.mean([len(self.tokenizer_for_length_filtering.encode(x['chosen'][1]['content'])) for x in self.dataset[k]])}"
            ]
            for k in self.dataset.keys()
        }
        response_lengths.update({
            k: [
                f"mean rejected response length: "
                f"{np.mean([len(self.tokenizer_for_length_filtering.encode(x['rejected'][1]['content'])) for x in self.dataset[k]])}"
            ]
            for k in self.dataset.keys()
        })
        logging_utils.info("=" * 110)
        logging_utils.info("Dataset characteristics:\n%s", json.dumps(num_rows, indent=2))
        logging_utils.info("Dataset characteristics regarding score means:\n%s", json.dumps(score_means, indent=2))
        logging_utils.info("Dataset characteristics regarding response lengths:\n%s", json.dumps(response_lengths, indent=2))

        if not self.config.language_model_path and "was_chosen_rejected_swapped" in self.dataset[next(iter(self.dataset.keys()))].column_names:
            gold_rm_acc = {k: [f"Gold RM Accuracy: {1 - np.mean(self.dataset[k]['was_chosen_rejected_swapped'])}"] for k in self.dataset.keys()}
            logging_utils.info("Dataset gold RM accuracy:\n%s", json.dumps(gold_rm_acc, indent=2))

        logging_utils.info("=" * 110 + "\n")

    def __delete_irrelevant_splits(self):
        for split in list(self.dataset.keys()):
            if split not in self.train_test_splits:
                del self.dataset[split]

    def __replace_existing_responses_with_onpolicy_generations(self):
        logging_utils.info(f"Using language model {self.config.language_model_path} for generating new responses.")
        self.lm_wrapper = self.__get_lm_wrapper(self.config.language_model_path,
                                                device=self.device,
                                                config=self.config)
        self.dataset = self.__replace_responses_with_new_generations(batch_size=self.config.rm_batch_size)

        del self.lm_wrapper
        torch.cuda.empty_cache()
        gc.collect()

    def __updated_dataset_w_computed_rewards(self, batch_size: int) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        tokenizer = self.rm_wrapper.tokenizer

        @torch.no_grad()
        def process_batch(batch):
            raw_chosen_texts = batch['chosen']
            raw_rejected_texts = batch['rejected']

            chosen_texts = tokenizer.apply_chat_template(batch['chosen'], tokenize=False)
            rejected_texts = tokenizer.apply_chat_template(batch['rejected'], tokenize=False)

            chosen_inputs = tokenizer(
                chosen_texts,
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_tensors='pt'
            ).to(self.device)
            rejected_inputs = tokenizer(
                rejected_texts,
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_tensors='pt'
            ).to(self.device)

            score_chosen_np = self.rm_wrapper.compute_batch_rewards(chosen_inputs).cpu().numpy()
            score_rejected_np = self.rm_wrapper.compute_batch_rewards(rejected_inputs).cpu().numpy()
            del rejected_inputs, chosen_inputs

            # Swap chosen and rejected where new score_chosen < new score_rejected
            swap_indices = score_chosen_np < score_rejected_np
            if any(swap_indices):
                # Swap texts
                chosen_array = np.array(raw_chosen_texts, dtype=object)
                rejected_array = np.array(raw_rejected_texts, dtype=object)

                chosen_array[swap_indices], rejected_array[swap_indices] = (
                    rejected_array[swap_indices],
                    chosen_array[swap_indices],
                )

                # Swap scores
                score_chosen_np[swap_indices], score_rejected_np[swap_indices] = (
                    score_rejected_np[swap_indices],
                    score_chosen_np[swap_indices],
                )

                batch['chosen'] = chosen_array.tolist()
                batch['rejected'] = rejected_array.tolist()

            batch['score_chosen'] = score_chosen_np
            batch['score_rejected'] = score_rejected_np
            batch['was_chosen_rejected_swapped'] = swap_indices.tolist()

            return batch

        for split in self.train_test_splits:
            logging_utils.info("=" * 110)
            logging_utils.info(f"Relabeling for the following split of {self.config.initial_dataset_path}: {split}")
            logging_utils.info("=" * 110 + "\n")
            self.dataset[split] = self.dataset[split].map(
                process_batch, batched=True, batch_size=batch_size, desc=f"Processing batches in {split}"
            )
            self.dataset[split] = self.dataset[split].remove_columns(["messages"])

        return self.dataset

    @torch.no_grad()
    def __replace_responses_with_new_generations(self, batch_size: int) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        tokenizer = self.lm_wrapper.tokenizer

        # If the language model does not have a chat template, adds a default one (chat template needs to already exist)
        if not tokenizer.chat_template:
            logging_utils.warning("Language model does not have a chat template. "
                                  "Adding a default one, which should only be used for debugging purposes.")
            update_tokenizer(tokenizer=tokenizer, num_added_toks=0, pad_token=DEFAULT_PADDING_TOKEN, eos_token=DEFAULT_EOS_TOKEN,
                             logger=logging_utils.get_default_logger(), user_token=DEFAULT_USER_TOKEN, assistant_token=DEFAULT_ASSISTANT_TOKEN)
            self.lm_wrapper.lm.config.pad_token_id = tokenizer.pad_token_id
            update_model_num_embeddings_and_special_tokens(self.lm_wrapper.lm, self.lm_wrapper.tokenizer)

        def process_batch(batch):
            batch_prompts = [[{"content": x, "role": "user"}] for x in batch['prompt']]
            prompts = tokenizer.apply_chat_template(batch_prompts, add_generation_prompt=True, tokenize=False)
            tokenizer_prompts = tokenizer(
                prompts,
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_tensors='pt'
            ).to(self.device)

            # Arbitrarily assigning to chosen and rejected, as rewards will be recomputed later on
            resp_1 = self.lm_wrapper.compute_batch_generation(tokenizer_prompts)
            batch['chosen'] = [
                p + [{"content": r, "role": "assistant"}]
                for p, r in zip(batch_prompts, resp_1)
            ]

            resp_2 = self.lm_wrapper.compute_batch_generation(tokenizer_prompts)
            batch['rejected'] = [
                p + [{"content": r, "role": "assistant"}]
                for p, r in zip(batch_prompts, resp_2)
            ]

            del tokenizer_prompts
            return batch

        for split in self.train_test_splits:
            logging_utils.info("=" * 110)
            logging_utils.info(f" Generating responses for the following split of {self.config.initial_dataset_path}: {split}\n")
            logging_utils.info("=" * 110 + "\n")
            self.dataset[split] = self.dataset[split].map(
                process_batch, batched=True, batch_size=batch_size, desc=f"Processing batches in {split}"
            )

        return self.dataset

    def __create_dataset_splits_for_rm_training_and_rlhf(self):
        train_split = self.dataset[self.config.train_split]

        train_rm_and_rlhf_splits = train_split.train_test_split(test_size=1 - float(self.config.frac_train_for_rm),
                                                                seed=self.config.train_split_seed)
        train_rm_split, train_rlhf_split = train_rm_and_rlhf_splits["train"], train_rm_and_rlhf_splits["test"]

        self.dataset = DatasetDict({
            DEFAULT_TRAIN_RM_SPLIT_NAME: train_rm_split,
            DEFAULT_TRAIN_RLHF_SPLIT_NAME: train_rlhf_split,
            DEFAULT_TEST_SPLIT_NAME: self.dataset[self.config.test_split]

        })
