import logging
import colorlog

def configure_logging(level: int = logging.INFO) -> None:
    """
    Configures the global logging settings for the application.

    This function should typically be called once at the start of the main script
    or optionally within a class if logging hasn't been configured elsewhere.

    Args:
        level (int, optional): The logging level to set.
            Common values include:
                - logging.DEBUG
                - logging.INFO (default)
                - logging.WARNING
                - logging.ERROR
                - logging.CRITICAL
    """
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

def get_logger_handler():
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    ))

    return handler