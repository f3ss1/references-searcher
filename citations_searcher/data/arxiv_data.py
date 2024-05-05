import json
import re
from collections.abc import Generator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from citations_searcher.constants import PROJECT_ROOT


@dataclass
class Paper:

    arxiv_id: str
    date: datetime
    title: str
    authors: str
    abstract: str
    category_1: str
    category_2: str | None = None

    # From Semantic Scholar
    semantic_id: str | None = None
    citation_count: int | None = None
    influential_citation_count: int | None = None
    embedding_v1: list[float] | None = None
    embedding_v2: list[float] | None = None
    tldr: str | None = None

    def to_dict(
        self,
    ) -> dict:
        return asdict(self)

    @classmethod
    def from_json_entry(
        cls,
        paper: dict,
        threshold_year: int | None,
    ) -> Optional["Paper"]:

        # Handle paper date
        raw_paper_date = paper["versions"][-1]["created"]
        paper_date = cls._validate_date(raw_paper_date, threshold_year)
        if paper_date is None:
            return None

        # Handle paper categories
        categories = cls._validate_categories(paper["categories"])
        if categories is None:
            return None

        return cls(
            paper["id"],
            paper_date,
            clear_line(paper["title"]),
            clear_line(paper["authors"]).replace(" and ", ", ").replace(",,", ","),
            clear_line(paper["abstract"]),
            **categories,
        )

    @staticmethod
    def _validate_date(
        date_str: str,
        threshold_year: int | None,
    ) -> datetime | None:
        try:
            paper_date = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S GMT")
            if threshold_year is not None and paper_date.year < threshold_year:
                return None

        # TODO: replace with a better error
        except:
            return None

        return paper_date

    @classmethod
    def _validate_categories(
        cls,
        categories_str: str,
    ):
        result = {"category_1": None, "category_2": None}
        categories = clear_line(categories_str).split(" ")

        if len(categories) == 0:
            return None

        elif len(categories) == 1:
            if not cls._check_category(categories[0]):
                return None
            else:
                result["category_1"] = categories[0]

        elif not cls._check_category(categories[0]) and not cls._check_category(categories[1]):
            return None

        else:
            result["category_1"] = categories[0]
            result["category_2"] = categories[1]

        return result

    @staticmethod
    def _check_category(category_str: str) -> bool:
        return category_str.startswith(("math.", "cs.", "econ.", "q-fin.", "stat."))


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
        "date": [],
        "title": [],
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

        paper_result = Paper.from_json_entry(paper, threshold_year)
        if paper_result is None:
            continue

        dataframe_dict["arxiv_id"].append(paper_result.arxiv_id)
        dataframe_dict["date"].append(str(paper_result.date))
        dataframe_dict["title"].append(paper_result.title)
        dataframe_dict["authors"].append(paper_result.authors)
        dataframe_dict["abstract"].append(paper_result.abstract)
        dataframe_dict["category_1"].append(paper_result.category_1)
        dataframe_dict["category_2"].append(paper_result.category_2)

    return pd.DataFrame(dataframe_dict)


def clear_line(line_to_clean: str) -> str:
    line_to_clean = _simplify_latex_diacritics(line_to_clean)
    return (
        line_to_clean.lower()
        .replace("\\'", "")
        .replace("\n", " ")
        .replace("  ", " ")
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
