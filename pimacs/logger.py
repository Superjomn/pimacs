import logging
import os

logger = logging.getLogger(__name__)

log_level = os.getenv("PIMACS_LOG_LEVEL", "WARNING")

logger.setLevel(getattr(logging, log_level))
