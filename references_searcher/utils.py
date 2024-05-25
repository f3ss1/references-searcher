from typing import Literal

from pathlib import Path
import torch
import numpy as np
import random
from contextlib import ContextDecorator
from functools import wraps
import time
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from collections.abc import Hashable

from references_searcher import logger
from references_searcher.constants import PROJECT_ROOT


class log_with_message(ContextDecorator):
    def __init__(
        self,
        message: str,
        log_time: bool = True,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    ):
        self.message = message
        self.log_time = log_time
        self.log_level = log_level
        self.start_time = None

    def __enter__(self):
        logger.log(self.log_level, f"Started {self.message}.")
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.exception("An exception occurred:")
        else:
            elapsed_time = time.time() - self.start_time
            if self.log_time:
                hours, rem = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(rem, 60)
                time_str = (
                    f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
                    if hours
                    else f"{int(minutes)}m {seconds:.2f}s" if minutes else f"{seconds:.2f}s"
                )
                logger.log(self.log_level, f"Finished {self.message}. Time taken: {time_str}.")
            else:
                logger.log(self.log_level, f"Finished {self.message}.")

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapped


@hydra.main(version_base=None, config_path=PROJECT_ROOT / "config", config_name="config")
def get_config(config: DictConfig) -> dict:
    return OmegaConf.to_container(config, resolve=True)


def verbose_iterator(
    iterator,
    verbose,
    desc: str | None = None,
    leave: bool = False,
    **kwargs,
):
    if verbose:
        return tqdm(
            iterator,
            leave=leave,
            desc=desc,
            **kwargs,
        )
    return iterator


def get_safe_save_path(
    file_path: Path,
) -> Path:

    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Find an available filename
    counter = 1
    current_filepath = file_path
    while current_filepath.exists():
        current_filepath = add_suffix(file_path, f"_{counter}")
        counter += 1

    return current_filepath


# TODO: add .tar.gz extension type support.
def add_suffix(file_path: Path, suffix: str) -> Path:
    # Assuming there is only single extension, no .tar.gz stuff.
    new_filename = f"{file_path.stem}{suffix}{file_path.suffix}"
    new_path = file_path.with_name(new_filename)
    return new_path


def seed_everything(
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def upsert_add_dict(
    dictionary: dict,
    key: Hashable,
    value,
    default=0,
):
    dictionary[key] = dictionary.get(key, default) + value


def generate_device(use_cuda: bool = True) -> torch.device:
    """Provides a devices based on either you want to use `cuda` or not.

    Parameters
    ----------
    use_cuda : bool
        If using a `cuda` device if possible is required.

    Returns
    -------
    device : torch.device
        The available device for further usage.

    """
    if use_cuda:
        if not torch.cuda.is_available():
            message = "CUDA is not available while being asked for it. Falling back to CPU."
            logger.warning(message)
            return torch.device("cpu")

        return torch.device("cuda:0")

    return torch.device("cpu")


def ensure_list(
    required_to_be_a_list,
) -> list:
    if not isinstance(required_to_be_a_list, list) and not isinstance(required_to_be_a_list, tuple):
        return [required_to_be_a_list]

    return list(required_to_be_a_list)
