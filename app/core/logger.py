import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logger(name: str = "app"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers are already added to avoid duplicates
    if not logger.handlers:
        # Format for logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (Rotating)
        file_handler = RotatingFileHandler(
            "app.log", maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Create a default logger instance
logger = setup_logger()
