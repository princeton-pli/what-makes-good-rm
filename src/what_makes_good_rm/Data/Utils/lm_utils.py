from abc import abstractmethod
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class LMWrapper:

    @abstractmethod
    def prepare(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_batch_generation(self, inputs, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class CausalLMWrapper(LMWrapper):

    def __init__(self, lm_string, device, config, language_model: torch.nn.Module = None, tokenizer=None):
        self.lm_string = lm_string
        self.device = device
        self.config = config
        self.lm = language_model
        self.tokenizer = tokenizer

    def prepare(self):
        if self.lm is None:
            self.lm = AutoModelForCausalLM.from_pretrained(self.lm_string,
                                                           cache_dir=self.config.cache_dir,
                                                           device_map=self.device,
                                                           trust_remote_code=True)
        self.lm.eval()
        self.lm.to(self.device)

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.lm_string,
                                                           cache_dir=self.config.cache_dir,
                                                           use_fast=True,
                                                           trust_remote_code=True)
        self.tokenizer.init_kwargs['padding_side'] = "left"
        self.tokenizer.padding_side = "left"
        self.tokenizer.init_kwargs['truncation_side'] = "right"
        self.tokenizer.truncation_side = "right"

        self.generation_config = GenerationConfig(
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **self.config.generation_config
        )

        if self.tokenizer.model_max_length > 100000:  # Fixing max length in tokenizer
            self.tokenizer.model_max_length = self.lm.config.max_position_embeddings

    @torch.no_grad()
    def compute_batch_generation(self, tokenized_inputs: List[str], **kwargs) -> torch.Tensor:
        outputs = self.lm.generate(**tokenized_inputs, generation_config=self.generation_config)
        return self.tokenizer.batch_decode(outputs[:, tokenized_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
