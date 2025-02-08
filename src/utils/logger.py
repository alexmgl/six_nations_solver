import os
import logging


def setup_logger(name=__name__, log_filename="six_nations_solver_log.log", level=logging.DEBUG):
    """
    Sets up a logger with a specific name and ensures the log file is always
    saved to the same absolute path, regardless of where it's run from.
    """

    # 1. Get the absolute directory of THIS logger.py file
    logger_file_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Optionally define a 'logs' subfolder (create if it doesn't exist)
    logs_folder = os.path.join(logger_file_dir, "..", "..", "logs")
    os.makedirs(logs_folder, exist_ok=True)

    # 3. Build the absolute path to the log file
    log_file_path = os.path.join(logs_folder, log_filename)

    # Create (or get) logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplication of handlers if already set up
    if not logger.handlers:
        # Create file handler, using the absolute path
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
