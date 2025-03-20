from typing import Dict


def parse_unknown_argparse_args_into_dict(unknown_args: list):
    processed_args = []
    for arg in unknown_args:
        processed_args.extend(arg.split("="))

    unknown_args_dict = {}
    for i in range(0, len(processed_args), 2):
        if i + 1 < len(processed_args) and processed_args[i].startswith('--'):
            key = processed_args[i].lstrip('--')
            value = processed_args[i + 1]
            unknown_args_dict[key] = value

    return unknown_args_dict


def convert_config_dict_to_str_args(config_dict: dict):
    """
    Converts a dictionary of configuration arguments to a string of CLI arguments.
    """
    flat_dict = __flatten_dict(config_dict)

    cli_args = []
    for key, value in flat_dict.items():
        if isinstance(value, list):
            value = ",".join(map(str, value))  # Join list values with commas
        cli_args.append(f"--{key} {value}")

    return " ".join(cli_args)


def __flatten_dict(d, parent_key="", sep="."):
    """Helper function to flatten a nested dictionary with dot notation."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(__flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def update_dict_config_with_dot_notation_override_args(dict_config: dict, override_args: Dict[str, str]):
    """
    For all (key, value) in 'override_args', updates the given 'dict_config' where the key is parsed using dot notation for hierarchy.
    :param dict_config: Dictionary of arguments to be updated.
    :param override_args: A "flat" dictionary containing pairs of string keys in dot notation for hierarchy and values.
    """
    for key, value in override_args.items():
        __update_with_parsed_dot_notation_arg(dict_config, key, value)


def __update_with_parsed_dot_notation_arg(dict_config: dict, key: str, value: str):
    """
    Updates the given arguments dictionary with the complex object field value that is described by the key by dot notation.
    :param dict_config: Dictionary of arguments to be updated.
    :param key: String key of an argument that contains dot notation.
    :param value: Value of argument.
    """
    key_parts = key.split(".")
    current_dict = dict_config
    for i, key_part in enumerate(key_parts):
        if i == len(key_parts) - 1:
            curr_value = current_dict.get(key_part)
            current_dict[key_part] = __parse_value_into_type(value, type(curr_value), curr_value)
            return

        if key_part not in current_dict:
            current_dict[key_part] = {}
        current_dict = current_dict[key_part]


def __parse_value_into_type(value: str, value_type: type, old_value):
    if value_type == bool:
        if value == "True" or value == "true":
            return True
        return False
    elif value_type == list:
        list_elements = value.split(",")
        if old_value is None:
            return list_elements

        list_element_type = type(old_value[0])
        if list_element_type == list:
            raise ValueError(f"Nested lists are not supported for overriding configuration arguments.")
        return [__parse_value_into_type(element, list_element_type, None) for element in list_elements]
    elif value_type == type(None):
        return str(value)
    else:
        return value_type(value)
