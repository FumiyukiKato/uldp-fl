import logging
import sys


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
