#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys
import logging

logger = None


def setup_logger(loglevel: str = "INFO"):
    global logger
    logger = logging.getLogger()
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    level = getattr(logging, loglevel.upper(), logging.INFO)
    logger.setLevel(level)
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(fmt="PID: %(process)d - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def format_msg(dscp: str = "", **kwargs) -> str:
    kws = [f"{k}: {w}" for k, w in kwargs.items()]
    kws = ", ".join(kws)

    msg = f"{dscp} {kws}"
    return msg


def logerr(dscp: str = "", **kwargs):
    msg = format_msg(f"❌ ERROR: {dscp}", **kwargs)
    logger.error(msg)


def logwarn(dscp: str = "", **kwargs):
    msg = format_msg(f"⚠️ WARN: {dscp}", **kwargs)
    logger.warning(msg)


def loginfo(dscp: str = "", **kwargs):
    msg = format_msg(dscp, **kwargs)
    logger.info(msg)


def logdebug(dscp: str = "", **kwargs):
    msg = format_msg(f"{dscp}", **kwargs)
    logger.debug(msg)
