import logging

logging.basicConfig(
    format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s"
)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger