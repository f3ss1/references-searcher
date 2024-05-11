from pathlib import Path
import torch
import numpy as np
import random

from collections.abc import Hashable

from citations_searcher import logger


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
