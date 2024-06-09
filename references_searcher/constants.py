import os
from pathlib import Path

from dotenv import load_dotenv
import hydra
from omegaconf import OmegaConf

RUNNING_IN_DOCKER = os.getenv("RUNNING_IN_DOCKER") == "true"
POSTGRES_HOST = "postgres" if RUNNING_IN_DOCKER else "localhost"

# Path variables
_current_file_path = Path(__file__).resolve()
PROJECT_ROOT = _current_file_path.parent.parent

# dotenv config variables
load_dotenv()
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"
SCHOLAR_API_KEY = os.getenv("SCHOLAR_API_KEY")
DJANGO_SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")

# TODO: handle this using singleton?
# Hydra load
hydra.initialize("../config", version_base="1.1")
config = hydra.compose(config_name="config")
CONFIG = OmegaConf.to_container(config, resolve=True)


__all__ = [
    "PROJECT_ROOT",
    "POSTGRES_HOST",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_URL",
    "SCHOLAR_API_KEY",
    "DJANGO_SECRET_KEY",
    "CONFIG",
]
