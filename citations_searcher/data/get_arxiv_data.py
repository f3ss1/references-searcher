import json
import re
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from citations_searcher.constants import PROJECT_ROOT


def get_lines(
    file_path: Path,
) -> Generator[dict]:
    with file_path.open() as file:
        for line in file:
            yield json.loads(line)


def generate_dataframe(
    file_path: str | Path = PROJECT_ROOT / "data/arxiv-metadata-oai-snapshot.json",
    file_size: int = 2450893,
    verbose: bool = True,
    threshold_year: int | None = None,
) -> pd.DataFrame:

    if isinstance(file_path, str):
        file_path = Path(file_path)

    dataframe_dict = {
        "arxiv_id": [],
        "title": [],
        "date": [],
        "authors": [],
        "abstract": [],
        "category_1": [],
        "category_2": [],
    }

    data = get_lines(file_path)

    if verbose:
        pbar = tqdm(total=file_size)

    for paper in data:
        if verbose:
            pbar.update(1)

        try:
            raw_paper_date = paper["versions"][-1]["created"]
            paper_date = datetime.strptime(raw_paper_date, "%a, %d %b %Y %H:%M:%S GMT")
            if threshold_year is not None and paper_date.year < threshold_year:
                continue

        # TODO: replace with a better error
        except:
            continue

        categories = clear_line(paper["categories"]).split(" ")

        if len(categories) == 0:
            continue

        elif len(categories) == 1:
            if not _check_category(categories[0]):
                continue
            else:
                dataframe_dict["category_1"].append(categories[0])
                dataframe_dict["category_2"].append(None)

        elif not _check_category(categories[0]) and not _check_category(categories[1]):
            continue

        else:
            dataframe_dict["category_1"].append(categories[0])
            dataframe_dict["category_2"].append(categories[1])

        dataframe_dict["arxiv_id"].append(paper["id"])
        dataframe_dict["title"].append(clear_line(paper["title"]))
        dataframe_dict["date"].append(str(paper_date))
        dataframe_dict["authors"].append(
            clear_line(paper["authors"]).replace(" and ", ", ").replace(",,", ","),
        )
        dataframe_dict["abstract"].append(clear_line(paper["abstract"]))

    return pd.DataFrame(dataframe_dict)


def _check_category(category_str: str) -> bool:
    return category_str.startswith(("math.", "cs.", "econ.", "q-fin.", "stat."))


def clear_line(line_to_clean: str) -> str:
    line_to_clean = _simplify_latex_diacritics(line_to_clean)
    return (
        line_to_clean.lower()
        .replace("  ", " ")
        .replace("\\'", "")
        .replace("\n", "")
        .replace('\\"', "")
        .replace(r"\`", "'")
        .replace(r"\^", "")
        .replace(r"\=", "")
        .replace("{\aa}", "a")
        .replace("{\\~}", "")
        .replace("{\\ss}", "ss")
        .replace("{\\i}", "i")
        .strip()
    )


def _simplify_latex_diacritics(line_to_clean: str) -> str:
    pattern = r"\\[a-zA-Z]\{([a-zA-Z])\}"
    return re.sub(pattern, r"\1", line_to_clean)
