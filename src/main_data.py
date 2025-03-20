import argparse
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

from what_makes_good_rm.Arguments import DatasetArguments, DatasetReLabelArguments
from what_makes_good_rm.Arguments.arg_utils import update_dict_config_with_dot_notation_override_args, parse_unknown_argparse_args_into_dict
from what_makes_good_rm.Data import PreferenceDatasetCreator
from what_makes_good_rm.Utils import single_process_logging as logging_utils


def __set_initial_random_seed(random_seed: int):
    if random_seed > 0:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration")

    args, unknown_args = parser.parse_known_args()
    config_path = args.config
    overrides_dict = parse_unknown_argparse_args_into_dict(unknown_args)
    with open(os.path.abspath(config_path), 'r') as file:
        dict_config = yaml.safe_load(file)

    update_dict_config_with_dot_notation_override_args(dict_config, overrides_dict)

    config = DatasetArguments(
        seed=dict_config.get('seed')
    )
    config.relabel_config = DatasetReLabelArguments.from_dict(dict_config["relabel_config"])

    run_id = random.randint(0, 10 ** 6)
    __set_initial_random_seed(config.seed)
    logging_utils.init_console_logging()
    path_for_saving = config.relabel_config.output_dir + (f"_seed_{config.seed}_rm_rlhf_split_seed_{config.relabel_config.train_split_seed}_"
                                                          f"rid_{run_id}")
    logging_utils.init_file_logging(log_file_name_prefix=f"pref_data_gen", output_dir=path_for_saving)

    arguments_str = "\n".join([arg for arg in sys.argv[1:]])
    logging_utils.info(f"Arguments received by the script:\n{arguments_str}")
    os.makedirs(path_for_saving, exist_ok=True)
    with open(os.path.join(path_for_saving, "run_config.yaml"), "w") as file:
        yaml.safe_dump(dict_config, file, default_flow_style=False)

    dataset = PreferenceDatasetCreator(config.relabel_config).prepare_dataset()

    if config.relabel_config.push_to_hub:
        dataset.push_to_hub(path_for_saving, private=path_for_saving)
    else:
        dataset.save_to_disk(path_for_saving)
        logging_utils.info(f"Dataset saved locally at: {path_for_saving}")

if __name__ == "__main__":
    start_time = datetime.now(timezone.utc)
    try:
        main()
    except Exception:
        logging_utils.exception("Exception while running preference data creation script.")
        raise
    finally:
        end_time = datetime.now(timezone.utc)
        logging_utils.info(f"Finished preference dataset creation script. Time took: {end_time - start_time}")
