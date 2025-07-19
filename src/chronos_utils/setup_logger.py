import os
import logging

def setup_logger(log_path: str):
    # Crea la directory se non esiste
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Formatter comune
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
