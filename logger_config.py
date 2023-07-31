import logging

logger = logging.getLogger("uvicorn")
logger.handlers = []
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)