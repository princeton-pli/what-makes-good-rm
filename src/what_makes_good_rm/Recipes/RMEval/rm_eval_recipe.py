import gc
import os
import shutil
from datetime import datetime, timezone

import numpy as np
import torch
from datasets import load_dataset, Dataset, load_from_disk
from torch.utils.data import DataLoader

import what_makes_good_rm.Utils.single_process_logging as logging_utils
from what_makes_good_rm.Data.Utils import CausalLMWrapper
from what_makes_good_rm.Data.Utils.rm_utils import RewardModelWrapper, LlamaModelRMWrapper
from what_makes_good_rm.Data.registry import REWARD_MODEL_WRAPPER_REGISTRY
from what_makes_good_rm.Recipes import BaseRecipe
from what_makes_good_rm.Recipes.RMEval.dataloaders import PromptDataset
from what_makes_good_rm.Utils import DEFAULT_USER_TOKEN, DEFAULT_ASSISTANT_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_PADDING_TOKEN, update_tokenizer, \
    update_model_num_embeddings_and_special_tokens
from what_makes_good_rm.Utils.strings import PER_PROMPT_RESPONSES, PER_PROMPT_REWARDS, PER_PROMPT_RANKING_ACCURACY, OFFLINE_RESPONSES_NAME, DATASET_KEY_NAME, \
    SPLIT_KEY_NAME, GOLD_RM_KEY_NAME, LM_KEY_NAME, RM_KEY_NAME, PROMPTS, PROMPT_INDICES_KEY_NAME

# Used for manually selecting LlamaForSequenceClassification instead of using AutoModelForSequenceClassification since the latter can fail
# when the ground truth reward model is ArmoRM (it seems that upon loading the ArmoRM model it modifies some configuration that creates a bug when loading other Llama-based reward models afterwards)
LLAMA_RMS = ["Ray2333/GRM-Llama3.2-3B-rewardmodel-ft", "allenai/llama-3-tulu-2-8b-uf-mean-rm"]


class RMEvalRecipe(BaseRecipe):

    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.lm_name = self.__extract_name_from_path(self.config.rm_eval_config.language_model_path)
        self.lm_file_name = self.__extract_file_name_from_path(self.config.rm_eval_config.language_model_path)
        self.gold_rm_name = self.__extract_name_from_path(self.config.rm_eval_config.ground_truth_reward_model_path)
        self.gold_rm_file_name = self.__extract_file_name_from_path(self.config.rm_eval_config.ground_truth_reward_model_path)
        self.lm_generation_train_path = os.path.join(self.config.rm_eval_config.output_dir,
                                                     f"{PER_PROMPT_RESPONSES}_{self.lm_name}_train.pt")
        self.prompt_train_path = os.path.join(self.config.rm_eval_config.output_dir,
                                              f"{PROMPTS}_{self.lm_name}_train.pt")
        self.lm_generation_test_path = os.path.join(self.config.rm_eval_config.output_dir,
                                                    f"{PER_PROMPT_RESPONSES}_{self.lm_name}_test.pt")
        self.prompt_test_path = os.path.join(self.config.rm_eval_config.output_dir,
                                             f"{PROMPTS}_{self.lm_name}_test.pt")
        self.logger = logging_utils.create_logger(console_logging=True, file_logging=True, log_dir=self.config.rm_eval_config.output_dir,
                                                  log_file_name_prefix="rm_eval")

        os.makedirs(self.config.rm_eval_config.output_dir, exist_ok=True)

    def __extract_name_from_path(self, path: str):
        return "_".join(path.split("/")[-3:]).replace(".", "-")

    def __extract_file_name_from_path(self, path: str):
        """The name used in file paths is shorter to avoid file names exceeding allowed length."""
        return "_".join(path.split("/")[-2:]).replace(".", "-")

    def __create_dict_for_saving(self, split: str, lm_name: str, rm_name: str, prompt_indices, values, values_key: str):
        return {
            DATASET_KEY_NAME: self.config.rm_eval_config.dataset_path,
            SPLIT_KEY_NAME: split,
            GOLD_RM_KEY_NAME: self.gold_rm_name,
            LM_KEY_NAME: lm_name,
            RM_KEY_NAME: rm_name,
            PROMPT_INDICES_KEY_NAME: prompt_indices,
            values_key: values
        }

    def __get_lm_wrapper(self, lm_str: str, device, config):
        try:
            lm_wrapper = CausalLMWrapper(lm_str, device=device, config=config)
            lm_wrapper.prepare()

            # If the language model does not have a chat template, adds a default one for debugging purposes (chat template needs to already exist)
            if not lm_wrapper.tokenizer.chat_template:
                self.logger.warning("Language model does not have a chat template. "
                                    "Adding a default one, which should only be used for debugging purposes.")
                update_tokenizer(tokenizer=lm_wrapper.tokenizer, num_added_toks=0, pad_token=DEFAULT_PADDING_TOKEN,
                                 eos_token=DEFAULT_EOS_TOKEN, logger=self.logger, user_token=DEFAULT_USER_TOKEN,
                                 assistant_token=DEFAULT_ASSISTANT_TOKEN)
                lm_wrapper.lm.config.pad_token_id = lm_wrapper.tokenizer.pad_token_id
                update_model_num_embeddings_and_special_tokens(lm_wrapper.lm, lm_wrapper.tokenizer)
            elif not lm_wrapper.tokenizer.pad_token_id:
                lm_wrapper.tokenizer.pad_token_id = lm_wrapper.tokenizer.eos_token_id
                lm_wrapper.lm.config.pad_token_id = lm_wrapper.tokenizer.pad_token_id

            return lm_wrapper
        except Exception:
            self.logger.exception("Exception while trying to load language model.")
            raise

    def __get_reward_model_wrapper(self, reward_model_name: str, is_llama_rm: bool = False, **kwargs):
        if reward_model_name in REWARD_MODEL_WRAPPER_REGISTRY:
            rm_wrapper = REWARD_MODEL_WRAPPER_REGISTRY[reward_model_name]()
        elif is_llama_rm:
            rm_wrapper = LlamaModelRMWrapper()
            kwargs["rm_str"] = reward_model_name
            kwargs["is_llama_rm"] = is_llama_rm
        else:
            self.logger.info(f"Reward model wrapper '{reward_model_name}' not found in the registry. Defaulting to general config")
            rm_wrapper = REWARD_MODEL_WRAPPER_REGISTRY["GeneralRM"]()
            kwargs["rm_str"] = reward_model_name

        rm_wrapper.prepare(**kwargs)

        if not rm_wrapper.tokenizer.chat_template:
            self.logger.warning(f"Reward model {reward_model_name} does not have a chat template. "
                                "Adding a default one, which should only be used for debugging purposes.")
            update_tokenizer(tokenizer=rm_wrapper.tokenizer, num_added_toks=0, pad_token=DEFAULT_PADDING_TOKEN,
                             eos_token=DEFAULT_EOS_TOKEN, logger=logging_utils.get_default_logger(), user_token=DEFAULT_USER_TOKEN,
                             assistant_token=DEFAULT_ASSISTANT_TOKEN)
            rm_wrapper.rm.config.pad_token_id = rm_wrapper.tokenizer.pad_token_id
            update_model_num_embeddings_and_special_tokens(rm_wrapper.rm, rm_wrapper.tokenizer)
        
        elif not rm_wrapper.rm.config.pad_token_id:
            rm_wrapper.rm.config.pad_token_id = rm_wrapper.tokenizer.pad_token_id

        return rm_wrapper

    def __save_rewards_and_log_metrics(self, per_prompt_rewards: torch.Tensor, prompt_indices: torch.Tensor, lm_name: str, rm_name: str,
                                       split: str = None, lm_file_name: str = "", rm_file_name: str = ""):
        lm_file_name = lm_file_name if lm_file_name else lm_name
        rm_file_name = rm_file_name if rm_file_name else rm_name
        rewards_file_name = f"{PER_PROMPT_REWARDS}_{lm_file_name}_{rm_file_name}_{split}.pt"
        per_prompt_statistics_path = os.path.join(self.config.rm_eval_config.output_dir, rewards_file_name)

        overall_reward_mean = per_prompt_rewards.mean()
        overall_reward_std = per_prompt_rewards.std()
        per_prompt_whitened_rewards = (per_prompt_rewards - overall_reward_mean) / overall_reward_std

        self.logger.info("=" * 180)
        self.logger.info(
            f"Saving per prompt rewards at {per_prompt_statistics_path}:\n"
            f"SPLIT: {split} , "
            f"LANGUAGE MODEL: {lm_name} , "
            f"REWARD MODEL: {rm_name} , "
            f"NUM PROMPTS: {per_prompt_rewards.shape[0]} , "
            f"NUM RESPONSES PER PROMPT: {per_prompt_rewards.shape[1]}"
        )
        torch.save(self.__create_dict_for_saving(split=split, lm_name=lm_name, rm_name=rm_name, prompt_indices=prompt_indices,
                                                 values=per_prompt_rewards, values_key=PER_PROMPT_REWARDS),
                   per_prompt_statistics_path)

        # Calculate mean, std and var of rewards from the aggregated
        self.logger.info("-" * 180)
        self.logger.info(f"Per prompt reward stats aggregated over the dataset:")
        self.__log_stats_for_quantity(per_prompt_rewards.mean(dim=1), "Reward Mean")
        self.__log_stats_for_quantity(per_prompt_rewards.var(dim=1), "Reward Variance")
        self.__log_stats_for_quantity(per_prompt_whitened_rewards.mean(dim=1), "Whitened Reward Mean")
        self.__log_stats_for_quantity(per_prompt_whitened_rewards.var(dim=1), "Whitened Reward Variance")
        self.logger.info("=" * 180)

    def __log_stats_for_quantity(self, values: torch.Tensor, quantity_name: str):
        self.logger.info(
            f"{quantity_name}: "
            f"mean {values.mean()} , "
            f"min {values.min()} , "
            f"25th percentile {torch.quantile(values, q=0.25)} , "
            f"median {values.median()} , "
            f"75th percentile {torch.quantile(values, q=0.75)} , "
            f"max {values.max()}"
        )

    def __compute_num_batches(self, dataset_len: int, batch_size: int):
        num_batches = dataset_len // batch_size
        if dataset_len % batch_size != 0:
            num_batches += 1

        return num_batches

    def __create_dataset_with_generated_responses(self, lm_wrapper: CausalLMWrapper, responses_out_file: str, prompts_out_file: str,
                                                  dataset, split: str, prompt_indices: torch.Tensor) -> Dataset:
        tokenizer = lm_wrapper.tokenizer
        num_return_sequences = self.config.rm_eval_config.generation_config["num_return_sequences"]

        buffer = []

        prompts_dataset = PromptDataset(dataset)
        prompts_dataloader = DataLoader(
            prompts_dataset,
            batch_size=self.config.rm_eval_config.lm_batch_size,
            shuffle=False,
            collate_fn=lambda batch: {
                "prompts": [item["prompt"] for item in batch],
                "prompt_ids": [item["prompt_id"] for item in batch]
            }
        )

        num_batches = self.__compute_num_batches(len(prompts_dataset), self.config.rm_eval_config.lm_batch_size)
        # Generate responses for each prompt
        for i, batch in enumerate(prompts_dataloader):
            if i % 10 == 0:
                self.logger.info(f"Generating responses for batch {i + 1} / {num_batches}")

            batch_prompts = batch["prompts"]
            batch_prompts_ids = batch["prompt_ids"]

            batch_prompts_chat_templated = tokenizer.apply_chat_template(
                [
                    [{"content": prompt, "role": "user"}] for prompt in batch_prompts
                ],
                add_generation_prompt=True,
                tokenize=False
            )
            inputs = tokenizer(batch_prompts_chat_templated, return_tensors="pt", padding=True,
                               truncation=False, add_special_tokens=False).to(self.device)

            # Generate multiple completions per prompt
            decoded_generations = lm_wrapper.compute_batch_generation(inputs)

            # Group outputs into structured format
            for i, (prompt, prompt_id) in enumerate(zip(batch_prompts, batch_prompts_ids)):
                prompt_generations = [
                    {
                        "prompt": prompt,
                        "prompt_id": prompt_id,
                        "response": decoded_generations[j]
                    } for j in range(i * num_return_sequences, (i + 1) * num_return_sequences)
                ]
                buffer.extend(prompt_generations)

        if self.config.rm_eval_config.save_generated_responses:
            prompts = [buffer[0]["prompt"]]
            per_prompt_responses = []
            curr_prompt_id = buffer[0]["prompt_id"]
            curr_prompt_responses = []
            for example in buffer:
                if example["prompt_id"] != curr_prompt_id:
                    per_prompt_responses.append(curr_prompt_responses)
                    curr_prompt_id = example["prompt_id"]
                    curr_prompt_responses = []
                    prompts.append(example["prompt"])
                curr_prompt_responses.append(example["response"])
            per_prompt_responses.append(curr_prompt_responses)

            self.logger.info("=" * 180)
            self.logger.info(f"Saving per prompt responses using {self.lm_name} at {responses_out_file}")
            self.logger.info("=" * 180)
            torch.save(self.__create_dict_for_saving(split=split, lm_name=self.lm_name, rm_name="", prompt_indices=prompt_indices,
                                                     values=per_prompt_responses, values_key=PER_PROMPT_RESPONSES), responses_out_file)
            self.logger.info("=" * 180)
            self.logger.info(f"Saving prompts at {prompts_out_file}")
            self.logger.info("=" * 180)
            torch.save(self.__create_dict_for_saving(split=split, lm_name="", rm_name="", prompt_indices=prompt_indices,
                                                     values=prompts, values_key=PROMPTS), prompts_out_file)

        return Dataset.from_list(buffer)

    def __compute_per_prompt_rewards(self, rm_wrapper: RewardModelWrapper, dataset):
        tokenizer = rm_wrapper.tokenizer

        data_loader = DataLoader(
            dataset,
            batch_size=self.config.rm_eval_config.rm_batch_size,
            shuffle=False,
            collate_fn=lambda x: x
        )

        num_batches = self.__compute_num_batches(len(dataset), self.config.rm_eval_config.rm_batch_size)

        rewards = []
        for i, batch in enumerate(data_loader):
            if i % 10 == 0:
                self.logger.info(f"Computing rewards for batch {i + 1} / {num_batches}")

            prompts = [entry["prompt"] for entry in batch]
            responses = [entry["response"] for entry in batch]
            formatted_inputs = tokenizer.apply_chat_template(
                [
                    [{"content": prompt, "role": "user"}, {"content": response, "role": "assistant"}]
                    for prompt, response in zip(prompts, responses)
                ],
                tokenize=False
            )
            formatted_inputs = [text for text in formatted_inputs]

            tokenized_inputs = tokenizer(
                formatted_inputs,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False
            ).to(self.device)

            batch_rewards = rm_wrapper.compute_batch_rewards(tokenized_inputs=tokenized_inputs).cpu().tolist()
            rewards.extend(batch_rewards)

        # Group rewards per prompt
        per_prompt_rewards = []
        curr_prompt_id = dataset[0]["prompt_id"]
        curr_prompt_rewards = []
        for example, reward in zip(dataset, rewards):
            if example["prompt_id"] != curr_prompt_id:
                curr_prompt_id = example["prompt_id"]
                per_prompt_rewards.append(curr_prompt_rewards)
                curr_prompt_rewards = []
            curr_prompt_rewards.append(reward)
        per_prompt_rewards.append(curr_prompt_rewards)

        return torch.tensor(per_prompt_rewards)

    def __compute_per_prompt_ranking_accuracy(self, per_prompt_rm_rewards: torch.Tensor, per_prompt_gold_rewards: torch.Tensor):
        idx1, idx2 = torch.triu_indices(per_prompt_rm_rewards.shape[1], per_prompt_rm_rewards.shape[1],
                                        offset=1, device=per_prompt_rm_rewards.device)

        rm_rewards_diff = per_prompt_rm_rewards[:, idx1] - per_prompt_rm_rewards[:, idx2]
        gold_rewards_diff = per_prompt_gold_rewards[:, idx1] - per_prompt_gold_rewards[:, idx2]

        rm_rewards_diff_sign = rm_rewards_diff.sign()
        gold_rewards_diff_sign = gold_rewards_diff.sign()

        agreement_matrix = (rm_rewards_diff_sign == gold_rewards_diff_sign).float()
        return agreement_matrix.mean(dim=1)

    def __subsample_dataset(self, dataset, num_samples: int, seed: int = -1):
        if num_samples >= len(dataset) or num_samples <= 0:
            return dataset

        if seed > 0:
            perm = np.random.RandomState(seed=seed).permutation(len(dataset))
        else:
            perm = np.random.permutation(len(dataset))

        chosen_indices = perm[:num_samples]
        return dataset.select(chosen_indices), torch.tensor(chosen_indices)

    def __compute_and_save_per_prompt_ranking_accuracy(self, per_prompt_rm_rewards: torch.Tensor, per_prompt_gold_rewards: torch.Tensor,
                                                       prompt_indices: torch.Tensor, lm_name: str, rm_name: str, split: str, lm_file_name: str = "",
                                                       rm_file_name: str = ""):
        lm_file_name = lm_file_name if lm_file_name else lm_name
        rm_file_name = rm_file_name if rm_file_name else rm_name
        rank_acc_file_name = f"{PER_PROMPT_RANKING_ACCURACY}_{lm_file_name}_{rm_file_name}_{split}.pt"
        file_path = os.path.join(self.config.rm_eval_config.output_dir, rank_acc_file_name)
        per_prompt_ranking_accuracy = self.__compute_per_prompt_ranking_accuracy(per_prompt_rm_rewards,
                                                                                 per_prompt_gold_rewards)

        self.logger.info("=" * 180)
        self.logger.info(
            f"Saving per prompt ranking accuracy at {file_path}:\n"
            f"SPLIT: {split} , "
            f"LANGUAGE MODEL: {lm_name} , "
            f"REWARD MODEL: {rm_name} , "
            f"NUM PROMPTS: {per_prompt_rm_rewards.shape[0]} , "
            f"NUM RESPONSES PER PROMPT: {per_prompt_rm_rewards.shape[1]}"
        )
        torch.save(self.__create_dict_for_saving(split=split, lm_name=lm_name, rm_name=rm_name, prompt_indices=prompt_indices,
                                                 values=per_prompt_ranking_accuracy, values_key=PER_PROMPT_RANKING_ACCURACY),
                   file_path)
        self.logger.info("-" * 180)
        self.__log_stats_for_quantity(per_prompt_ranking_accuracy, "Ranking Accuracy")
        self.logger.info("=" * 180)

    def __prepare_train_and_test_datasets(self):
        if not self.config.rm_eval_config.load_dataset_from_file:
            preference_dataset = load_dataset(self.config.rm_eval_config.dataset_path,
                                              trust_remote_code=True,
                                              cache_dir=self.config.cache_dir)
        else:
            preference_dataset = load_from_disk(self.config.rm_eval_config.dataset_path)

        random_seed = self.config.rm_eval_config.data_selection_seed
        if self.config.rm_eval_config.train_split:
            self.train_dataset, self.train_prompt_indices = self.__subsample_dataset(preference_dataset[self.config.rm_eval_config.train_split],
                                                                                     num_samples=self.config.rm_eval_config.num_train_samples,
                                                                                     seed=random_seed)
        if self.config.rm_eval_config.test_split:
            self.test_dataset, self.test_prompt_indices = self.__subsample_dataset(preference_dataset[self.config.rm_eval_config.test_split],
                                                                                   num_samples=self.config.rm_eval_config.num_test_samples,
                                                                                   seed=random_seed + 1 if random_seed > 0 else -1)

    def __create_online_responses_dataset(self):
        lm_wrapper = self.__get_lm_wrapper(self.config.rm_eval_config.language_model_path,
                                           device=self.device,
                                           config=self.config.rm_eval_config)
        online_train_dataset = None
        if self.config.rm_eval_config.train_split:
            self.logger.info(f"Using LM {self.lm_name}, starting to generate responses for the TRAIN split")
            online_train_dataset = self.__create_dataset_with_generated_responses(lm_wrapper=lm_wrapper,
                                                                                  responses_out_file=self.lm_generation_train_path,
                                                                                  prompts_out_file=self.prompt_train_path,
                                                                                  dataset=self.train_dataset,
                                                                                  split=self.config.rm_eval_config.train_split,
                                                                                  prompt_indices=self.train_prompt_indices)
        online_test_dataset = None
        if self.config.rm_eval_config.test_split:
            self.logger.info(f"Using LM {self.lm_name}, starting to generate responses for the TEST split")
            online_test_dataset = self.__create_dataset_with_generated_responses(lm_wrapper=lm_wrapper,
                                                                                 responses_out_file=self.lm_generation_test_path,
                                                                                 prompts_out_file=self.prompt_test_path,
                                                                                 dataset=self.test_dataset,
                                                                                 split=self.config.rm_eval_config.test_split,
                                                                                 prompt_indices=self.test_prompt_indices)
        # Free device from language model
        del lm_wrapper

        if self.config.rm_eval_config.delete_language_model_checkpoint_after_eval:
            self.logger.warning(f"DELETING language model checkpoint from: {self.config.rm_eval_config.language_model_path}")
            shutil.rmtree(self.config.rm_eval_config.language_model_path, ignore_errors=True)

        torch.cuda.empty_cache()
        gc.collect()
        return online_train_dataset, online_test_dataset

    def __create_prompt_response_dataset_from_offline_responses(self, dataset):
        """
        Receives a Hugging Face dataset that contains columns named 'prompt', 'prompt_id', 'chosen', and 'rejected',
        and returns a dataset with columns 'prompt', 'prompt_id', 'response', where the order of the prompts is maintained and for
        each prompt the chosen and rejected responses appear in consecutive rows (first the chosen and then the rejected)..
        """
        buffer = []
        for example in dataset:
            prompt = example["prompt"]
            prompt_id = example["prompt_id"]
            chosen = example["chosen"][1]["content"]
            rejected = example["rejected"][1]["content"]

            buffer.append({
                "prompt": prompt,
                "prompt_id": prompt_id,
                "response": chosen
            })
            buffer.append({
                "prompt": prompt,
                "prompt_id": prompt_id,
                "response": rejected
            })

        return Dataset.from_list(buffer)

    def __compute_and_save_rewards_and_ranking_accuracies_for_rm(self, online_train_prompt_response_dataset,
                                                                 online_test_prompt_response_dataset,
                                                                 rm_name: str, rm_file_name: str, rm_wrapper: RewardModelWrapper,
                                                                 train_offline_gold_rewards: torch.Tensor, train_online_gold_rewards: torch.Tensor,
                                                                 test_offline_gold_rewards: torch.Tensor, test_online_gold_rewards: torch.Tensor):
        # Compute rewards over the ONLINE responses
        if self.config.rm_eval_config.train_split:
            self.logger.info(f"For RM {rm_name}, starting to compute per prompt rewards for ONLINE TRAIN responses")
            train_online_rm_rewards = self.__compute_per_prompt_rewards(rm_wrapper=rm_wrapper,
                                                                        dataset=online_train_prompt_response_dataset)
            self.__save_rewards_and_log_metrics(per_prompt_rewards=train_online_rm_rewards,
                                                prompt_indices=self.train_prompt_indices,
                                                lm_name=self.lm_name,
                                                rm_name=rm_name,
                                                split=self.config.rm_eval_config.train_split,
                                                lm_file_name=self.lm_file_name,
                                                rm_file_name=rm_file_name)
        if self.config.rm_eval_config.test_split:
            self.logger.info(f"For RM {rm_name}, starting to compute per prompt rewards for ONLINE TEST responses")
            test_online_rm_rewards = self.__compute_per_prompt_rewards(rm_wrapper=rm_wrapper,
                                                                       dataset=online_test_prompt_response_dataset)
            self.__save_rewards_and_log_metrics(per_prompt_rewards=test_online_rm_rewards,
                                                prompt_indices=self.test_prompt_indices,
                                                lm_name=self.lm_name,
                                                rm_name=rm_name,
                                                split=self.config.rm_eval_config.test_split,
                                                lm_file_name=self.lm_file_name,
                                                rm_file_name=rm_file_name)

        # Compute rewards over the OFFLINE responses
        if not self.config.rm_eval_config.only_lm_eval:
            if self.config.rm_eval_config.train_split:
                self.logger.info(f"For RM {rm_name}, starting to compute per prompt rewards for OFFLINE TRAIN responses")
                train_offline_prompt_response_dataset = self.__create_prompt_response_dataset_from_offline_responses(self.train_dataset)
                train_offline_rm_rewards = self.__compute_per_prompt_rewards(rm_wrapper=rm_wrapper,
                                                                             dataset=train_offline_prompt_response_dataset)
                self.__save_rewards_and_log_metrics(per_prompt_rewards=train_offline_rm_rewards,
                                                    prompt_indices=self.train_prompt_indices,
                                                    lm_name=OFFLINE_RESPONSES_NAME,
                                                    rm_name=rm_name,
                                                    split=self.config.rm_eval_config.train_split,
                                                    rm_file_name=rm_file_name)
            if self.config.rm_eval_config.test_split:
                self.logger.info(f"For RM {rm_name}, starting to compute per prompt rewards for OFFLINE TEST responses")
                test_offline_prompt_response_dataset = self.__create_prompt_response_dataset_from_offline_responses(self.test_dataset)
                test_offline_rm_rewards = self.__compute_per_prompt_rewards(rm_wrapper=rm_wrapper,
                                                                            dataset=test_offline_prompt_response_dataset)
                self.__save_rewards_and_log_metrics(per_prompt_rewards=test_offline_rm_rewards,
                                                    prompt_indices=self.test_prompt_indices,
                                                    lm_name=OFFLINE_RESPONSES_NAME,
                                                    rm_name=rm_name,
                                                    split=self.config.rm_eval_config.test_split,
                                                    rm_file_name=rm_file_name)
        del rm_wrapper
        torch.cuda.empty_cache()
        gc.collect()

        # Compute per prompt ranking accuracy over the ONLINE and OFFLINE responses for the reward model
        if self.config.rm_eval_config.train_split:
            self.__compute_and_save_per_prompt_ranking_accuracy(per_prompt_rm_rewards=train_online_rm_rewards,
                                                                per_prompt_gold_rewards=train_online_gold_rewards,
                                                                prompt_indices=self.train_prompt_indices,
                                                                lm_name=self.lm_name,
                                                                rm_name=rm_name,
                                                                split=self.config.rm_eval_config.train_split,
                                                                lm_file_name=self.lm_file_name,
                                                                rm_file_name=rm_file_name)
            if not self.config.rm_eval_config.only_lm_eval:
                self.__compute_and_save_per_prompt_ranking_accuracy(per_prompt_rm_rewards=train_offline_rm_rewards,
                                                                    per_prompt_gold_rewards=train_offline_gold_rewards,
                                                                    prompt_indices=self.train_prompt_indices,
                                                                    lm_name=OFFLINE_RESPONSES_NAME,
                                                                    rm_name=rm_name,
                                                                    split=self.config.rm_eval_config.train_split,
                                                                    rm_file_name=rm_file_name)

        if self.config.rm_eval_config.test_split:
            self.__compute_and_save_per_prompt_ranking_accuracy(per_prompt_rm_rewards=test_online_rm_rewards,
                                                                per_prompt_gold_rewards=test_online_gold_rewards,
                                                                prompt_indices=self.test_prompt_indices,
                                                                lm_name=self.lm_name,
                                                                rm_name=rm_name,
                                                                split=self.config.rm_eval_config.test_split,
                                                                lm_file_name=self.lm_file_name,
                                                                rm_file_name=rm_file_name)
            if not self.config.rm_eval_config.only_lm_eval:
                self.__compute_and_save_per_prompt_ranking_accuracy(per_prompt_rm_rewards=test_offline_rm_rewards,
                                                                    per_prompt_gold_rewards=test_offline_gold_rewards,
                                                                    prompt_indices=self.test_prompt_indices,
                                                                    lm_name=OFFLINE_RESPONSES_NAME,
                                                                    rm_name=rm_name,
                                                                    split=self.config.rm_eval_config.test_split,
                                                                    rm_file_name=rm_file_name)

    @torch.no_grad()
    def run(self):
        start_time = datetime.now(timezone.utc)
        try:
            self.__prepare_train_and_test_datasets()

            train_offline_gold_rewards, train_online_gold_rewards = None, None
            test_offline_gold_rewards, test_online_gold_rewards = None, None

            # Compute per prompt reward metrics for the OFFLINE responses in the dataset (already precomputed in dataset generation)
            # If script is ran for evaluating a language model, then will not compute and save rewards for offline responses
            if not self.config.rm_eval_config.only_lm_eval:
                if self.config.rm_eval_config.train_split:
                    train_offline_gold_rewards = torch.stack([torch.tensor(self.train_dataset[key]) for key in ["score_chosen", "score_rejected"]],
                                                             dim=1)
                    self.__save_rewards_and_log_metrics(per_prompt_rewards=train_offline_gold_rewards,
                                                        prompt_indices=self.train_prompt_indices,
                                                        lm_name=OFFLINE_RESPONSES_NAME,
                                                        rm_name=self.gold_rm_name,
                                                        split=self.config.rm_eval_config.train_split,
                                                        rm_file_name=self.gold_rm_file_name)

                if self.config.rm_eval_config.test_split:
                    test_offline_gold_rewards = torch.stack([torch.tensor(self.test_dataset[key]) for key in ["score_chosen", "score_rejected"]],
                                                            dim=1)
                    self.__save_rewards_and_log_metrics(per_prompt_rewards=test_offline_gold_rewards,
                                                        prompt_indices=self.test_prompt_indices,
                                                        lm_name=OFFLINE_RESPONSES_NAME,
                                                        rm_name=self.gold_rm_name,
                                                        split=self.config.rm_eval_config.test_split,
                                                        rm_file_name=self.gold_rm_file_name)

            # Load language model and generate 'num_return_sequences' responses per prompt
            online_train_prompt_response_dataset, online_test_prompt_response_dataset = self.__create_online_responses_dataset()

            # Load ground truth reward model only after generating responses and removing the language model from the GPU
            rm_wrapper = self.__get_reward_model_wrapper(self.config.rm_eval_config.ground_truth_reward_model_path, device=self.device,
                                                         cache_dir=self.config.cache_dir)

            # Compute ground truth rewards for ONLINE data for train split
            if self.config.rm_eval_config.train_split:
                self.logger.info(f"For RM {self.gold_rm_name}, starting to compute per prompt rewards for ONLINE TRAIN responses")
                train_online_gold_rewards = self.__compute_per_prompt_rewards(rm_wrapper=rm_wrapper,
                                                                              dataset=online_train_prompt_response_dataset)
                self.__save_rewards_and_log_metrics(per_prompt_rewards=train_online_gold_rewards,
                                                    prompt_indices=self.train_prompt_indices,
                                                    lm_name=self.lm_name,
                                                    rm_name=self.gold_rm_name,
                                                    split=self.config.rm_eval_config.train_split,
                                                    lm_file_name=self.lm_file_name,
                                                    rm_file_name=self.gold_rm_file_name)

            # Compute ground truth rewards for ONLINE data for test split
            if self.config.rm_eval_config.test_split:
                self.logger.info(f"For RM {self.gold_rm_name}, starting to compute per prompt rewards for ONLINE TEST responses")
                test_online_gold_rewards = self.__compute_per_prompt_rewards(rm_wrapper=rm_wrapper,
                                                                             dataset=online_test_prompt_response_dataset)
                self.__save_rewards_and_log_metrics(per_prompt_rewards=test_online_gold_rewards,
                                                    prompt_indices=self.test_prompt_indices,
                                                    lm_name=self.lm_name,
                                                    rm_name=self.gold_rm_name,
                                                    split=self.config.rm_eval_config.test_split,
                                                    lm_file_name=self.lm_file_name,
                                                    rm_file_name=self.gold_rm_file_name)

            del rm_wrapper
            torch.cuda.empty_cache()
            gc.collect()

            for rm_str in self.config.rm_eval_config.proxy_reward_models:
                if rm_str == self.config.rm_eval_config.ground_truth_reward_model_path:
                    # If the gold reward model is passed as one of the proxy models as well, no need to evaluate again
                    continue

                rm_wrapper = self.__get_reward_model_wrapper(rm_str, device=self.device, cache_dir=self.config.cache_dir,
                                                             is_llama_rm=rm_str in LLAMA_RMS)
                rm_name = self.__extract_name_from_path(rm_str)
                rm_file_name = self.__extract_file_name_from_path(rm_str)
                self.__compute_and_save_rewards_and_ranking_accuracies_for_rm(online_train_prompt_response_dataset,
                                                                              online_test_prompt_response_dataset,
                                                                              rm_name, rm_file_name, rm_wrapper,
                                                                              train_offline_gold_rewards=train_offline_gold_rewards,
                                                                              train_online_gold_rewards=train_online_gold_rewards,
                                                                              test_offline_gold_rewards=test_offline_gold_rewards,
                                                                              test_online_gold_rewards=test_online_gold_rewards)

        except Exception:
            self.logger.exception("Exception while running evaluation script.")
            raise
        finally:
            end_time = datetime.now(timezone.utc)
            self.logger.info(f"Finished running evaluation script. Time took: {end_time - start_time}")
