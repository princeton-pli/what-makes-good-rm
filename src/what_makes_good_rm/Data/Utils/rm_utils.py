from abc import abstractmethod

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, LlamaForSequenceClassification


class RewardModelWrapper:

    @abstractmethod
    def prepare(self, **kwargs):
        """
        Setup the reward model and its related resources (e.g. tokenizer) for inference.\
        @param kwargs: Reward model specific arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_batch_rewards(self, tokenized_inputs, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class ArmoRMWrapper(RewardModelWrapper):

    def prepare(self, device=torch.device("cpu"), cache_dir: str = None, **kwargs):
        self.rm = AutoModelForSequenceClassification.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1",
                                                                     device_map=device,
                                                                     trust_remote_code=True,
                                                                     torch_dtype=torch.bfloat16,
                                                                     cache_dir=cache_dir)
        self.rm.eval()
        self.rm.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1",
                                                       use_fast=True,
                                                       trust_remote_code=True,
                                                       cache_dir=cache_dir)

        if self.tokenizer.model_max_length > 100000:  # Fixing max length in tokenizer
            self.tokenizer.model_max_length = self.rm.config.max_position_embeddings

    @torch.no_grad()
    def compute_batch_rewards(self, tokenized_inputs, **kwargs):
        output = self.rm(**tokenized_inputs)
        return output.score.float()


class LlamaModelRMWrapper(RewardModelWrapper):

    def prepare(self, rm_str=None, device=torch.device("cpu"), cache_dir: str = None, **kwargs):
        self.rm = LlamaForSequenceClassification.from_pretrained(rm_str,
                                                                 num_labels=1,
                                                                 device_map=device,
                                                                 trust_remote_code=True,
                                                                 cache_dir=cache_dir)
        self.rm.eval()
        self.rm.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(rm_str,
                                                       use_fast=True,
                                                       trust_remote_code=True,
                                                       cache_dir=cache_dir)

        if self.tokenizer.model_max_length > 100000:  # Fixing max length in tokenizer
            self.tokenizer.model_max_length = self.rm.config.max_position_embeddings

    @torch.no_grad()
    def compute_batch_rewards(self, tokenized_inputs, **kwargs):
        output = self.rm(**tokenized_inputs)
        return output.logits.float().view(-1)


class GeneralRMWrapper(RewardModelWrapper):

    def prepare(self, rm_str=None, device=torch.device("cpu"), cache_dir: str = None, **kwargs):
        self.rm = AutoModelForSequenceClassification.from_pretrained(rm_str,
                                                                     num_labels=1,
                                                                     device_map=device,
                                                                     trust_remote_code=True,
                                                                     cache_dir=cache_dir)
        self.rm.eval()
        self.rm.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(rm_str,
                                                       use_fast=True,
                                                       trust_remote_code=True,
                                                       cache_dir=cache_dir)

        if self.tokenizer.model_max_length > 100000:  # Fixing max length in tokenizer
            self.tokenizer.model_max_length = self.rm.config.max_position_embeddings

    @torch.no_grad()
    def compute_batch_rewards(self, tokenized_inputs, **kwargs):
        output = self.rm(**tokenized_inputs)
        return output.logits.float().view(-1)
