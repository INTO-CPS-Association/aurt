import os
import logging

NONINTERACTIVE = "NONINTERACTIVE" in os.environ

# Set up logger
logger = logging.getLogger("CalibrationTool")
logger.setLevel(logging.ERROR)
formatter = logging.Formatter("%(levelname)s: %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)