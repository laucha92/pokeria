import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    """Set up a logger that writes to a file."""
    # Create the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create the file handler
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.FileHandler(log_file)

    # Create the formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)

    return logger