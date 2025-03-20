from what_makes_good_rm.Utils import get_chat_template


def update_tokenizer(tokenizer, num_added_toks, pad_token, eos_token, logger, user_token=None, assistant_token=None):
    if not tokenizer.pad_token_id:
        num_added_toks += tokenizer.add_special_tokens({"pad_token": pad_token})
        logger.warning(
            "Adding pad_token. You need to resize your embeddings if you add a new token and add it to your model as well. "
            "This should have been done during sft or rm"
        )

    if not tokenizer.eos_token:
        num_added_toks += tokenizer.add_special_tokens({"eos_token": eos_token})
        logger.warning(
            "No eos_token. You need to resize your embeddings if you add a new token and add it to your model as well. "
            "This should have been done during sft or rm"
        )

    if not tokenizer.chat_template:
        logger.warning("No chat template implemented --- creating a default one")
        num_added_toks += tokenizer.add_special_tokens({"additional_special_tokens": [f"{user_token}", f"{assistant_token}"]})
        tokenizer.chat_template = get_chat_template()

    tokenizer.init_kwargs['padding_side'] = "left"
    tokenizer.padding_side = "left"

    tokenizer.init_kwargs['truncation_side'] = "right"
    tokenizer.truncation_side = "right"

    return tokenizer, num_added_toks


def update_model_num_embeddings_and_special_tokens(model, tokenizer):
    model.resize_token_embeddings(len(tokenizer))

    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return model
