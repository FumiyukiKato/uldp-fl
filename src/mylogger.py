import logging
import sys
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)


logger = logging.getLogger("acsilo")
stream_handler = logging.StreamHandler(stream=sys.stdout)
formatter = logging.Formatter(
    fmt="[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def logger_set_debug():
    logger.setLevel(logging.DEBUG)
    warnings.simplefilter(action="default", category=UserWarning)


def logger_set_info():
    logger.setLevel(logging.INFO)
    warnings.simplefilter(action="ignore", category=UserWarning)


def logger_set_warning():
    logger.setLevel(logging.WARNING)
    warnings.simplefilter(action="ignore", category=UserWarning)
