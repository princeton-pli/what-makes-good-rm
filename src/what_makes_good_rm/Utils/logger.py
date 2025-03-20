import logging
from torch.distributed import is_initialized, get_rank

def get_logger(name=__name__):
    """
    Returns a logger with the specified name.
    Configures the logger with a standard format and level.
    
    Args:
        name (str): The name of the logger, typically __name__ for the calling module.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # Avoid duplicate handlers in Jupyter environments or repeated imports
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)  # Set the default log level here
    return logger

def is_main_process():
    return not is_initialized() or get_rank() == 0