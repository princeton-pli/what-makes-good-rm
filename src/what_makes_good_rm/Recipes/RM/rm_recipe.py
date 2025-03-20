import logging
import os

import numpy as np
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer

from what_makes_good_rm.Recipes import BaseRecipe
from what_makes_good_rm.Utils import get_logger, DEFAULT_USER_TOKEN, DEFAULT_ASSISTANT_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_PADDING_TOKEN, update_tokenizer, \
    update_model_num_embeddings_and_special_tokens
from what_makes_good_rm.Utils import is_main_process
from what_makes_good_rm.Utils.strings import DEFAULT_TRAIN_RM_SPLIT_NAME, DEFAULT_TEST_SPLIT_NAME

logger = get_logger(__name__)
if not is_main_process():
    logger.setLevel(logging.WARNING)


class RMRecipe(BaseRecipe):
    def __init__(self, config):
        super().__init__(config)
        self.train_dataset = None
        self.eval_dataset = None
        self.num_added_toks = 0

        logger.info(f"Initialized RMRecipe with config: {self.config}")

    def __prepare_tokenizer(self, **kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.rm_config.pretrained_model_path,
                                                       trust_remote_code=self.config.rm_config.trust_remote_code,
                                                       use_fast=True,
                                                       cache_dir=self.config.cache_dir)

        self.tokenizer, self.num_added_toks = update_tokenizer(self.tokenizer, self.num_added_toks, DEFAULT_PADDING_TOKEN,
                                                               DEFAULT_EOS_TOKEN, logger, DEFAULT_USER_TOKEN, DEFAULT_ASSISTANT_TOKEN)

    def __prepare_dataset(self, **kwargs):
        dataset = self.__load_dataset(self.config.rm_config.dataset_path)
        self.train_dataset, self.eval_dataset = dataset[DEFAULT_TRAIN_RM_SPLIT_NAME], dataset.get(DEFAULT_TEST_SPLIT_NAME)

        if self.config.rm_config.data_selection_seed > 0:
            perm = np.random.RandomState(seed=self.config.rm_config.data_selection_seed).permutation(len(self.train_dataset))
        else:
            perm = np.random.permutation(len(self.train_dataset))

        num_train_samples = min(self.config.rm_config.num_train_samples, len(self.train_dataset))
        if num_train_samples > 0 and num_train_samples < len(self.train_dataset):
            perm = perm[:num_train_samples]

        if not self.config.rm_config.second_dataset_path:
            self.train_dataset = self.train_dataset.select(perm)
        else:
            second_dataset = self.__load_dataset(self.config.rm_config.second_dataset_path)[DEFAULT_TRAIN_RM_SPLIT_NAME]
            num_samples_from_first = int(self.config.rm_config.frac_samples_from_first_dataset * len(perm))
            self.train_dataset = concatenate_datasets([self.train_dataset.select(perm[:num_samples_from_first]),
                                                       second_dataset.select(perm[num_samples_from_first:])])

        self.__format_dataset_inputs()

    def __format_dataset_inputs(self):
        def format_inputs(sample):
            sample['chosen'] = self.tokenizer.apply_chat_template(sample['chosen'], tokenize=False, add_generation_prompt=False)
            sample['rejected'] = self.tokenizer.apply_chat_template(sample['rejected'], tokenize=False, add_generation_prompt=False)
            return sample

        self.train_dataset = self.train_dataset.map(format_inputs)
        self.train_dataset = self.train_dataset.select_columns(['chosen', 'rejected'])
        if self.eval_dataset is not None:
            self.eval_dataset = self.eval_dataset.map(format_inputs)
            self.eval_dataset = self.eval_dataset.select_columns(['chosen', 'rejected'])

    def __load_dataset(self, dataset_path: str):
        if not self.config.rm_config.load_dataset_from_file:
            dataset = load_dataset(dataset_path, trust_remote_code=True, cache_dir=self.config.cache_dir)
        else:
            dataset = load_from_disk(dataset_path)
        return dataset

    def run(self, **kwargs):
        self.__prepare_tokenizer()
        torch_dtype = torch.bfloat16 if self.config.rm_config.bf16 else (torch.float16 if self.config.rm_config.fp16 else torch.float32)

        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.rm_config.pretrained_model_path,
            num_labels=1,
            trust_remote_code=self.config.rm_config.trust_remote_code,
            torch_dtype=torch_dtype,
            cache_dir=self.config.cache_dir
        )

        if self.num_added_toks > 0:
            self.reward_model = update_model_num_embeddings_and_special_tokens(self.reward_model, self.tokenizer)

        if self.tokenizer.model_max_length > 10000:
            self.tokenizer.model_max_length = self.reward_model.config.max_position_embeddings

        self.config.rm_config.algo_specific.max_length = self.tokenizer.model_max_length

        self.__prepare_dataset()

        self.trainer = RewardTrainer(
            args=self.config.rm_config.algo_specific,
            model=self.reward_model,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )

        logger.info(f"Starting reward model training...\n"
                    f"Number of training samples: {len(self.train_dataset)}\n"
                    f"Number of test samples: {len(self.eval_dataset) if self.eval_dataset is not None else 0}")

        self.trainer.train()

        if self.eval_dataset:
            self.trainer.evaluate()

        logger.info(f"Finished training reward model")
        model_already_saved_by_trained = self.config.rm_config.algo_specific.save_strategy != "no"
        if self.config.rm_config.save_model and not model_already_saved_by_trained:
            rm_dir = os.path.join(self.config.rm_config.output_dir, "reward_model")
            self.trainer.save_model(rm_dir)
            logger.info(f"Saved model and tokenizer to '{rm_dir}'")
