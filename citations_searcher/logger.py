from loguru import logger

from citations_searcher.constants import PROJECT_ROOT


log_file_path = PROJECT_ROOT / "logs/common_log.log"
logger.add(log_file_path)
