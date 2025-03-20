import logging
import os
import sys
import time
from datetime import datetime

__logger = logging.getLogger(__name__)


def init_logging(file_log: bool = False, log_dir: str = "", log_file_name_prefix: str = "", log_level=logging.INFO):
    if file_log:
        init_file_logging(log_file_name_prefix=log_file_name_prefix, output_dir=log_dir, log_level=log_level)
    else:
        init_console_logging(log_level=log_level)


def init_console_logging(log_level=logging.INFO):
    __logger.setLevel(log_level)

    ch = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    formatter.converter = time.gmtime
    ch.setFormatter(formatter)
    __logger.addHandler(ch)


def init_file_logging(log_file_name_prefix: str, output_dir: str, log_level=logging.INFO):
    __logger.setLevel(log_level)
    os.makedirs(output_dir, exist_ok=True)

    now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
    log_file_path = os.path.join(output_dir, f"{log_file_name_prefix}_{now_utc_str}.log")

    ch = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    formatter.converter = time.gmtime
    ch.setFormatter(formatter)
    __logger.addHandler(ch)


def get_default_logger():
    return __logger

def log(level: logging.INFO, msg, *args, **kwargs):
    if __logger is not None:
        __logger.log(level, msg, *args, **kwargs)

def debug(msg, *args, **kwargs):
    if __logger is not None:
        __logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    if __logger is not None:
        __logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    if __logger is not None:
        __logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    if __logger is not None:
        __logger.error(msg, *args, **kwargs)


def exception(msg, *args, exc_info=True, **kwargs):
    if __logger is not None:
        __logger.exception(msg, *args, exc_info=exc_info, **kwargs)


def create_logger(console_logging: bool = True, file_logging: bool = False, log_dir: str = "", log_file_name_prefix: str = "log",
                  create_dir: bool = True, add_time_stamp_to_log_name: bool = True, timestamp: datetime = None,
                  msg_format: str = "%(asctime)s - %(levelname)s - %(message)s", log_level: int = logging.INFO) -> logging.Logger:
    """
    :param console_logging: whether to output logs to the console (stdout)
    :param file_logging: whether to write logs to a file
    :param log_dir: directory to save log file in (default is cwd)
    :param log_file_name_prefix: name prefix for the file log
    :param create_dir: whether to create directory of the log file if it doesn't exist
    :param add_time_stamp_to_log_name: whether to add timestamp to the log file name
    :param timestamp: timestamp to add to the log file name (default is current utc time)
    :param msg_format: message format string
    :param log_level: log level of the Logger
    :return: Logger
    """
    curr_time_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
    logger_name = f"{log_file_name_prefix}_{curr_time_str}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    if file_logging:
        __add_file_handler(logger, log_dir,
                           log_file_name_prefix=log_file_name_prefix,
                           create_dir=create_dir,
                           timestamp=timestamp,
                           add_time_stamp_to_log_name=add_time_stamp_to_log_name,
                           msg_format=msg_format)

    if console_logging:
        __add_console_handler(logger, msg_format)

    return logger


def __add_file_handler(logger: logging.Logger, log_dir: str = "", log_file_name_prefix: str = "log", create_dir: bool = True,
                       timestamp: datetime = None, add_time_stamp_to_log_name: bool = True,
                       msg_format: str = "%(asctime)s - %(levelname)s - %(message)s"):
    if create_dir and log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if add_time_stamp_to_log_name:
        timestamp = timestamp if timestamp else datetime.utcnow()
        timestamp_str = timestamp.strftime("%Y_%m_%d-%H_%M_%S")
        log_file_name = f"{log_file_name_prefix}_{timestamp_str}.log"
    else:
        log_file_name = f"{log_file_name_prefix}.log"

    log_dir = log_dir if log_dir else os.getcwd()
    log_file_path = os.path.join(log_dir, log_file_name)

    fh = logging.FileHandler(log_file_path)
    formatter = logging.Formatter(msg_format)
    formatter.converter = time.gmtime
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def __add_console_handler(logger: logging.Logger, msg_format: str = "%(asctime)s - %(levelname)s - %(message)s"):
    ch = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(msg_format)
    formatter.converter = time.gmtime
    ch.setFormatter(formatter)
    logger.addHandler(ch)
