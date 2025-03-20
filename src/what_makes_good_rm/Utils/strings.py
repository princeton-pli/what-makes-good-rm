DEFAULT_USER_TOKEN = "<|user|>"
DEFAULT_ASSISTANT_TOKEN = "<|assistant|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_PADDING_TOKEN = "<|padding|>"

DATASET_KEY_NAME = "dataset"
SPLIT_KEY_NAME = "split"
GOLD_RM_KEY_NAME = "gold_rm_name"
LM_KEY_NAME = "lm_name"
RM_KEY_NAME = "rm_name"
PROMPT_INDICES_KEY_NAME = "prompt_indices"

OFFLINE_RESPONSES_NAME = "offline"
PROMPTS ="prompts"
PER_PROMPT_RESPONSES = "per_prompt_responses"
PER_PROMPT_REWARDS = "per_prompt_rewards"
PER_PROMPT_RANKING_ACCURACY = "per_prompt_ranking_acc"

DEFAULT_TRAIN_RM_SPLIT_NAME = "train_rm_prefs"
DEFAULT_TRAIN_RLHF_SPLIT_NAME = "train_rlhf_prefs"
DEFAULT_TEST_SPLIT_NAME = "test_prefs"


def get_chat_template():
    CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '{DEFAULT_USER_TOKEN}' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ '{DEFAULT_ASSISTANT_TOKEN}' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '{DEFAULT_ASSISTANT_TOKEN}'  + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '{DEFAULT_ASSISTANT_TOKEN}' }}{% endif %}{% endfor %}"
    # CHAT_TEMPLATE = """{%- for message in messages %}{%- if message['role'] == 'user' %}{{ '{DEFAULT_USER_TOKEN}' }}{%- elif message['role'] == 'system' or message['role'] == 'assistant' %}{{ '{DEFAULT_ASSISTANT_TOKEN}' }}{%- endif %}{{ message['content'] + eos_token }}{%- if not loop.last %}\n{%- endif %}{%- endfor %}{%- if add_generation_prompt %}\n{{ '{DEFAULT_ASSISTANT_TOKEN}' }}{%- endif %}"""

    CHAT_TEMPLATE = CHAT_TEMPLATE.replace(
        '{DEFAULT_USER_TOKEN}', DEFAULT_USER_TOKEN
    ).replace(
        '{DEFAULT_ASSISTANT_TOKEN}', DEFAULT_ASSISTANT_TOKEN
    )

    return CHAT_TEMPLATE