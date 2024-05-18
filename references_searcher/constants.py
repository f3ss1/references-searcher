import os
from pathlib import Path

from dotenv import load_dotenv
import hydra
from omegaconf import OmegaConf


# Path variables
_current_file_path = Path(__file__).resolve()
PROJECT_ROOT = _current_file_path.parent.parent

# dotenv config variables
load_dotenv()
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost/{POSTGRES_DB}"
SCHOLAR_API_KEY = os.getenv("SCHOLAR_API_KEY")
DJANGO_SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")

# TODO: handle this using singleton?
# Hydra load
hydra.initialize("../config")
config = hydra.compose(config_name="config")
CONFIG = OmegaConf.to_container(config, resolve=True)


__all__ = [
    "PROJECT_ROOT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_URL",
    "SCHOLAR_API_KEY",
    "DJANGO_SECRET_KEY",
    "CONFIG",
]
