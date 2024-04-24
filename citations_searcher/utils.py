from pathlib import Path


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
