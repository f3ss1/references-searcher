import sys

from loguru import logger

from references_searcher.constants import PROJECT_ROOT


log_file_path = PROJECT_ROOT / "logs/common_log.log"
logger.remove()
logger.add(log_file_path, level="DEBUG")
logger.add(sys.stderr, level="INFO")
