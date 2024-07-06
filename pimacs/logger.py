import logging
import os


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    level = os.environ.get("PIMACS_LOG_LEVEL", "INFO")
    level = getattr(logging, level)

    logger.setLevel(level)

    ch = logging.StreamHandler()

    ch.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger
