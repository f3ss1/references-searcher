import json
from pathlib import Path
from time import sleep

import requests

from citations_searcher import logger
from citations_searcher.utils import get_safe_save_path


def get_semantic_data(
    api_key: str,
    arxiv_ids: list[str],
    save_path: str | Path,
    requested_fields: list | None = None,
    default_window_size: int = 400,
    successful_requests_to_save: int = 20,
) -> None:
    if requested_fields is None:
        requested_fields = [
            "citationCount",
            "externalIds",
            "influentialCitationCount",
            "s2FieldsOfStudy",
            "embedding.specter_v1",
            "embedding.specter_v2",
            "tldr",
            "references.externalIds",
            "references.citationCount",
            "references.influentialCitationCount",
            "references.s2FieldsOfStudy",
        ]

    if isinstance(save_path, str):
        save_path = Path(save_path)

    save_path = get_safe_save_path(save_path)

    valid_request_fields = ",".join(requested_fields)
    headers = {"x-api-key": api_key}

    start_index = 0
    window_size = default_window_size
    successful_requests = 0
    too_many_requests_count = 0
    unhandled_error_count = 0
    total_excluded_objects = 0
    total_result = []

    logger.info(f"Started requesting data from Semantic Scholar for {len(arxiv_ids)} papers.")
    while start_index < len(arxiv_ids):
        current_arxiv_ids_selection = arxiv_ids[start_index : start_index + window_size]
        requested_ids = [f"ARXIV:{x}" for x in current_arxiv_ids_selection]
        request_result = requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={"fields": valid_request_fields},
            json={"ids": requested_ids},
            headers=headers,
        )

        if request_result.status_code == 400:
            window_size //= 2
            logger.warning(f"The requested window was too big to process, narrowing it down to {window_size}!")

            # It appears that the issue is with a given paper, so we skip it.
            if window_size == 0:
                logger.warning(f"Got window size 0, skipping paper {requested_ids}!")
                total_excluded_objects += 1
                unhandled_error_count += 1
                start_index += 1
                window_size = 1

        elif request_result.status_code == 429:
            too_many_requests_count += 1
            logger.warning(f"Got {too_many_requests_count} 429 codes, going to timeout for 1 second!")
            sleep(1)

        elif request_result.status_code == 200:
            logger.info(
                f"Successful request for {window_size} ids,"
                f" {max(len(arxiv_ids) - start_index - window_size, 0)} remaining.",
            )
            too_many_requests_count = 0
            successful_requests += 1
            start_index += window_size
            window_size = default_window_size

            request_json = request_result.json()
            json_with_arxiv_ids = _inject_arxiv_data(current_arxiv_ids_selection, request_json)
            total_result.extend(json_with_arxiv_ids)
            total_excluded_objects += len(request_json) - len(json_with_arxiv_ids)

            if successful_requests % successful_requests_to_save == 0:
                logger.info(f"Dumping the list to {save_path}...")
                with save_path.open("a") as file:
                    json.dump(total_result, file)
                logger.info("Successful dump!")

                total_result = []

        else:
            logger.error(
                f"Unhandled error for batch with start {start_index} and window"
                f" size {window_size}: {request_result.status_code}! Skipping.",
            )
            unhandled_error_count += 1
            total_excluded_objects += window_size
            start_index += window_size
            too_many_requests_count = 0
            window_size = default_window_size

    logger.info(f"Dumping the list to {save_path}...")
    with save_path.open("a") as file:
        json.dump(total_result, file)
    logger.info("Successful dump!")

    logger.info(
        f"Finished processing {len(arxiv_ids)} objects, encountered"
        f" {unhandled_error_count} unhandled error, excluded {total_excluded_objects} papers.",
    )


def _inject_arxiv_data(
    arxiv_ids: list[str],
    injection_target: list[dict],
) -> list:

    logger.info(f"Starting injecting arxiv ids into {len(injection_target)} entries.")
    result = []
    for arxiv_id, paper_result in zip(arxiv_ids, injection_target, strict=True):
        if paper_result is None or "error" in paper_result:
            logger.warning(f"Encountered a None value for paper {arxiv_id}, skipping it!")
            continue
        paper_result["arxiv_id"] = arxiv_id
        result.append(paper_result)

    logger.info(f"Successfully injected arxiv ids for {len(result)} papers.")
    return result
