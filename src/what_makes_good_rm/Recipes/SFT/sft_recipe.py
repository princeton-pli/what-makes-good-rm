import os

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import logging

from what_makes_good_rm.Recipes import BaseRecipe
from what_makes_good_rm.Utils import get_logger, DEFAULT_USER_TOKEN, DEFAULT_ASSISTANT_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_PADDING_TOKEN, update_tokenizer, \
    update_model_num_embeddings_and_special_tokens, is_main_process

logger = get_logger(__name__)
if not is_main_process():
    logger.setLevel(logging.WARNING)

class SFTRecipe(BaseRecipe):
    def __init__(self, config):
        super().__init__(config)
        self.train_dataset = None
        self.eval_dataset = None
        self.num_added_toks = 0

        logger.info(f"Initialized SFTRecipe with config: {self.config}")

    def __prepare_tokenizer(self, **kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.sft_config.pretrained_model_path,
                                                       trust_remote_code=self.config.sft_config.trust_remote_code,
                                                       use_fast=True,
                                                       cache_dir=self.config.cache_dir)
        self.tokenizer, self.num_added_toks = update_tokenizer(self.tokenizer, self.num_added_toks, DEFAULT_PADDING_TOKEN,
                                                               DEFAULT_EOS_TOKEN, logger, DEFAULT_USER_TOKEN, DEFAULT_ASSISTANT_TOKEN)

    def __prepare_dataset(self, **kwargs):
        if not self.config.sft_config.load_dataset_from_file:
            dataset = load_dataset(self.config.sft_config.dataset_path, trust_remote_code=True, cache_dir=self.config.cache_dir)
        else:
            dataset = load_from_disk(self.config.sft_config.dataset_path)

        self.train_dataset = dataset[self.config.sft_config.dataset_train_split]
        if self.config.sft_config.dataset_test_split:
            self.eval_dataset = dataset[self.config.sft_config.dataset_test_split]
        else:
            train_test_dataset_splits = self.train_dataset.train_test_split(test_size=0.05, seed=self.config.sft_config.seed)
            self.train_dataset, self.eval_dataset = train_test_dataset_splits["train"], train_test_dataset_splits["test"]

        if self.config.sft_config.num_train_samples > 0:
            num_train_samples = min(self.config.sft_config.num_train_samples, len(self.train_dataset))
            chosen_indices = torch.randperm(len(self.train_dataset))[:num_train_samples]
            self.train_dataset = self.train_dataset.select(chosen_indices)

    def run(self, **kwargs):
        """
        See https://huggingface.co/docs/trl/sft_trainer if more specific implementation is needed
        """
        self.__prepare_tokenizer()

        def formatting_prompts_func(batch):
            if "alpaca" in self.config.sft_config.dataset_path:
                chat_prompts = [
                    [{
                        "role": "user",
                        "content": f"{instruction_text}" if not input_text else f"{instruction_text}\n{input_text}"
                    }]
                    for instruction_text, input_text, output_text in zip(batch["instruction"], batch["input"], batch["output"])
                ]
                batch = [
                    text + [{"role": "assistant", "content": out}]
                    for text, out in zip(chat_prompts, batch['output'])
                ]
            else:
                batch = batch[self.config.sft_config.dataset_text_field]

            chat_texts = self.tokenizer.apply_chat_template(batch, tokenize=False)

            if self.config.sft_config.algo_specific.max_seq_length is None:
                return chat_texts

            # Filter out samples that are too long
            return [text for text in chat_texts if len(self.tokenizer.encode(text)) <= self.config.sft_config.algo_specific.max_seq_length]

        torch_dtype = torch.bfloat16 if self.config.sft_config.bf16 else (torch.float16 if self.config.sft_config.fp16 else torch.float32)
        self.config.sft_config.torch_dtype = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.sft_config.pretrained_model_path,
            trust_remote_code=self.config.sft_config.trust_remote_code,
            torch_dtype=torch_dtype,
            cache_dir=self.config.cache_dir
        )

        if self.num_added_toks > 0:
            self.model = update_model_num_embeddings_and_special_tokens(self.model, self.tokenizer)

        if self.tokenizer.model_max_length > 10000:
            self.tokenizer.model_max_length = self.model.config.max_position_embeddings

        if self.config.sft_config.algo_specific.max_seq_length is None:
            self.config.sft_config.algo_specific.max_seq_length = self.tokenizer.model_max_length

        self.__prepare_dataset()

        collator = DataCollatorForCompletionOnlyLM(response_template=f"{DEFAULT_ASSISTANT_TOKEN}",
                                                   instruction_template=f"{DEFAULT_USER_TOKEN}",
                                                   tokenizer=self.tokenizer)

        self.trainer = SFTTrainer(
            model=self.model,
            args=self.config.sft_config.algo_specific,
            data_collator=collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            formatting_func=formatting_prompts_func
        )

        logger.info(f"Starting SFT...\n"
                    f"Number of training samples: {len(self.train_dataset)}\n"
                    f"Number of test samples: {len(self.eval_dataset) if self.eval_dataset is not None else 0}")

        self.trainer.train()

        if self.eval_dataset:
            self.trainer.evaluate()

        logger.info(f"Finished SFT")
        model_already_saved_by_trained = self.config.sft_config.algo_specific.save_strategy != "no"
        if self.config.sft_config.save_model and not model_already_saved_by_trained:
            sft_model_dir = os.path.join(self.config.sft_config.output_dir, "sft_model")
            self.trainer.save_model(sft_model_dir)
            logger.info(f"Saved model and tokenizer at '{sft_model_dir}'")
