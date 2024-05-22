from pathlib import Path
import requests
from time import sleep
import pandas as pd

from sqlalchemy import create_engine, func as F, select, text
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from references_searcher import logger
from references_searcher.constants import POSTGRES_URL, PROJECT_ROOT, SCHOLAR_API_KEY
from references_searcher.data import Paper, get_lines
from references_searcher.data.sql import SQLPaper, SQLReferences


# TODO: convert to a singleton for a single postgres url.
class DatabaseInterface:

    def __init__(
        self,
        postgres_url: str = POSTGRES_URL,
    ) -> None:
        self.engine = create_engine(postgres_url)
        self.session_factory = sessionmaker(bind=self.engine)

        # Init tables if those do not exist
        SQLPaper.metadata.create_all(self.engine)
        SQLReferences.metadata.create_all(bind=self.engine)

    def get_positive_references(self, cutoff: int | None = None) -> pd.DataFrame:
        positive_query = """
        SELECT
            paper_arxiv_id,
            paper_title,
            paper_abstract,
            reference_arxiv_id,
            reference_title,
            reference_abstract,
            reference_citation_count,
            reference_influential_citation_count
        FROM joined_filtered_positive_references
        """

        with self.engine.connect() as conn:
            positive_references = pd.read_sql_query(text(positive_query), conn)

        print(positive_references.head())
        if cutoff is not None:
            return positive_references.iloc[:cutoff]
        return positive_references

    def get_negative_references(self, cutoff: int | None = None) -> pd.DataFrame:
        negative_query = """
        SELECT
            paper_arxiv_id,
            paper_title,
            paper_abstract,
            reference_title,
            reference_abstract,
            reference_citation_count,
            reference_influential_citation_count
        FROM joined_filtered_negative_references
        """
        with self.engine.connect() as conn:
            negative_references = pd.read_sql_query(text(negative_query), conn)

        if cutoff is not None:
            return negative_references.iloc[:cutoff]
        return negative_references

    def get_references_metadata(self):
        metadata_query = """
        SELECT
            arxiv_id,
            title,
            abstract
        FROM train_references_metadata
        """
        with self.engine.connect() as conn:
            metadata = pd.read_sql_query(text(metadata_query), conn)
            metadata.index = metadata["arxiv_id"]
            return metadata.drop("arxiv_id", axis=1)

    def upload_papers_to_db(
        self,
        file_path: str | Path = PROJECT_ROOT / "data/arxiv-metadata-oai-snapshot.json",
        file_size: int = 2450893,
        verbose: bool = True,
        threshold_year: int | None = None,
    ) -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        data = get_lines(file_path)
        result = []

        if verbose:
            pbar = tqdm(total=file_size)

        logger.info(f"Processing {file_size} items from {file_path}...")
        for paper in data:
            if verbose:
                pbar.update(1)

            paper_result = Paper.from_json_entry(paper, threshold_year)
            if paper_result is None:
                continue
            db_entry = SQLPaper.from_paper(paper_result)
            result.append(db_entry)

        logger.info("Finished processing!")

        logger.info(f"Dumping {len(result)} objects to database...")
        with self.session_factory() as session:
            session.add_all(result)
            session.commit()
        logger.info("Dump successful!")

    def get_semantic_data(
        self,
        requested_fields: list | None = None,
        default_window_size: int = 400,
        successful_requests_to_save: int = 20,
    ) -> None:
        if requested_fields is None:
            requested_fields = [
                "citationCount",
                # "externalIds",
                "influentialCitationCount",
                "embedding.specter_v1",
                "embedding.specter_v2",
                "tldr",
                "references.externalIds",
                "references.citationCount",
                "references.influentialCitationCount",
            ]

        valid_request_fields = ",".join(requested_fields)
        headers = {"x-api-key": SCHOLAR_API_KEY}

        start_index = 0
        window_size = default_window_size
        successful_requests = 0
        too_many_requests_count = 0
        unhandled_error_count = 0
        total_excluded_objects = 0
        total_result = []
        total_references_result = []

        database_paper_objects, arxiv_ids = self._get_valid_objects()
        enriched_objects = set(arxiv_ids)
        all_objects = self._get_all_objects()

        logger.info(f"Started requesting data from Semantic Scholar for {len(arxiv_ids)} papers.")
        while start_index < len(arxiv_ids):
            current_database_paper_objects = database_paper_objects[start_index : start_index + window_size]
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
                logger.error(f"The requested window was too big to process, narrowing it down to {window_size}!")

                # It appears that the issue is with a given paper, so we skip it.
                if window_size == 0:
                    logger.warning(f"Got window size 0, skipping paper {requested_ids}!")
                    total_excluded_objects += 1
                    unhandled_error_count += 1
                    start_index += 1
                    window_size = 1

            elif request_result.status_code == 429:
                too_many_requests_count += 1
                logger.error(f"Got {too_many_requests_count} 429 codes, going to timeout for 1 second!")
                sleep(1)

            elif request_result.status_code == 504:
                logger.error("Got 504 code, going to timeout for 1 second, hoping the host will wake up!")
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

                references_result, enriched_result = self._inject_references_data(
                    current_database_paper_objects,
                    request_json,
                    all_objects,
                    enriched_objects,
                )
                total_result.extend(current_database_paper_objects)
                total_result.extend(enriched_result)
                total_excluded_objects += len(request_json) - len(current_database_paper_objects)

                total_references_result.extend(references_result)

                if successful_requests % successful_requests_to_save == 0:
                    logger.info(
                        f"Reached dumping checkpoint, dumping {len(total_result)} objects to papers database...",
                    )
                    with self.session_factory() as session:
                        session.add_all(total_result)
                        session.commit()
                    logger.info("Successful dump!")
                    total_result = []

                    logger.info(
                        f"Dumping {len(total_references_result)} objects to references database...",
                    )
                    with self.session_factory() as session:
                        session.add_all(total_references_result)
                        session.commit()
                    logger.info("Successful dump!")
                    total_references_result = []

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

        logger.info(f"Making the last dump of {len(total_result)} objects to database...")
        with self.session_factory() as session:
            session.add_all(total_result)
            session.commit()
        logger.info("Successful dump!")

        logger.info(
            f"Finished processing {len(arxiv_ids)} objects, encountered"
            f" {unhandled_error_count} unhandled errors, excluded {total_excluded_objects} papers.",
        )

    def _get_valid_objects(
        self,
    ) -> (list[SQLPaper], list[str]):
        with self.session_factory() as session:
            result_objects = session.scalars(
                select(SQLPaper)
                .where(
                    (SQLPaper.category_1.startswith("cs.")) | (F.coalesce(SQLPaper.category_2, "").startswith("cs.")),
                )
                .order_by(SQLPaper.id),
            ).all()

        result_arxiv_ids = [item.arxiv_id for item in result_objects]

        return result_objects, result_arxiv_ids

    def _get_all_objects(
        self,
    ) -> dict[str, SQLPaper]:
        with self.session_factory() as session:
            result_objects = session.scalars(
                select(SQLPaper).order_by(SQLPaper.id),
            ).all()

        result_dict = {item.arxiv_id: item for item in result_objects}

        return result_dict

    @classmethod
    def _inject_references_data(
        cls,
        database_paper_objects: list[SQLPaper],
        request_json: list[dict],
        all_objects: dict[str, SQLPaper],
        enriched_objects: set[str],
    ) -> (list[SQLReferences], list[SQLPaper]):
        logger.info(f"Starting injecting arxiv ids into {len(database_paper_objects)} entries.")

        ignored = []
        no_references = []

        references_result = []
        enriched_result = []
        for paper, paper_request_result in zip(database_paper_objects, request_json, strict=True):
            if paper_request_result is None or "error" in paper_request_result:
                ignored.append(paper.arxiv_id)
                continue

            cls._inject_additional_data_in_paper(paper, paper_request_result)

            # References processing
            paper_references_list = paper_request_result["references"]
            if len(paper_references_list) == 0:
                no_references.append(paper.arxiv_id)
                continue

            for paper_reference in paper_references_list:
                paper_reference_entry = cls._get_reference_entry(paper.arxiv_id, paper_reference)
                if paper_reference_entry is not None:
                    references_result.append(paper_reference_entry)
                    if paper_reference_entry.reference_arxiv_id not in enriched_objects:
                        enriched_objects.add(paper_reference_entry.reference_arxiv_id)
                        enriched_paper = all_objects.get(paper_reference_entry.reference_arxiv_id, None)
                        if enriched_paper is not None:
                            enriched_paper.semantic_id = paper_reference["paperId"]
                            enriched_paper.citation_count = paper_reference["citationCount"]
                            enriched_paper.influential_citation_count = paper_reference["influentialCitationCount"]
                            enriched_result.append(enriched_paper)

        # Logging
        logger.debug(
            f"Finished processing arxiv_ids: {', '.join([paper.arxiv_id for paper in database_paper_objects])}",
        )
        if len(ignored) > 0:
            logger.warning(f"Encountered a None value for papers {', '.join(ignored)}, skipping them!")
        logger.info(f"Successfully injected arxiv ids for {len(database_paper_objects) - len(ignored)} papers.")

        return references_result, enriched_result

    @staticmethod
    def _get_reference_entry(
        paper_arxiv_id: str,
        reference: dict,
    ) -> SQLReferences | None:

        external_ids = reference["externalIds"]
        if external_ids is None:
            return None

        reference_arxiv_id = external_ids.get("ArXiv", None)
        if reference_arxiv_id is None:
            return None

        return SQLReferences(paper_arxiv_id=paper_arxiv_id, reference_arxiv_id=reference_arxiv_id)

    @staticmethod
    def _inject_additional_data_in_paper(
        paper: SQLPaper,
        paper_request_result: dict,
    ) -> None:
        paper.semantic_id = paper_request_result["paperId"]
        paper.citation_count = paper_request_result["citationCount"]
        paper.influential_citation_count = paper_request_result["influentialCitationCount"]

        if paper_request_result["tldr"] is not None:
            paper.tldr = paper_request_result["tldr"]["text"]

        embedding_dict = paper_request_result["embedding"]
        if embedding_dict is not None:
            if embedding_dict["model"] == "specter_v1":
                paper.embedding_v1 = embedding_dict["vector"]
            elif embedding_dict["model"] == "specter_v2":
                paper.embedding_v2 = embedding_dict["vector"]
