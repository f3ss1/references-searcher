from typing import Literal
from random import Random

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, PreTrainedTokenizer, BatchEncoding

from references_searcher.utils import log_with_message
from references_searcher import logger

torch.set_printoptions(threshold=torch.inf)


class ArxivDataset(Dataset):
    def __init__(
        self,
        positive_pairs: pd.DataFrame,
        mode: Literal[
            "fixed_negative_sampling",
            "uniform_negative_sampling",
            "distance_based_negative_sampling",
            "triplet_loss",
        ] = "fixed_negative_sampling",
        negative_pairs: pd.DataFrame | None = None,
        references_metadata: pd.DataFrame | None = None,
        negative_probability_matrix: torch.Tensor | None = None,
        paper_arxiv_id_to_index_mapping: dict[str, int] | None = None,
        references_arxiv_id_to_index_mapping: dict[str, int] | None = None,
        title_process_mode: Literal["separate", "combined"] = "combined",
        return_target: bool = True,
        tokenizer: PreTrainedTokenizer | None = None,
        seed: int = 42,
    ):
        super(ArxivDataset, self).__init__()
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=512)
        else:
            self.tokenizer = tokenizer

        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs
        self.references_metadata = references_metadata
        self.negative_probability_matrix = negative_probability_matrix

        self.paper_arxiv_id_to_index_mapping = paper_arxiv_id_to_index_mapping
        self.index_to_paper_arxiv_id_mapping = (
            {value: key for key, value in paper_arxiv_id_to_index_mapping.items()}
            if paper_arxiv_id_to_index_mapping is not None
            else None
        )
        self.references_arxiv_id_to_index_mapping = references_arxiv_id_to_index_mapping
        self.index_to_references_arxiv_id_mapping = (
            {value: key for key, value in references_arxiv_id_to_index_mapping.items()}
            if paper_arxiv_id_to_index_mapping is not None
            else None
        )

        self.return_target = return_target
        self.title_process_mode = title_process_mode

        self.mode = self.validate_mode(mode)
        self.negative_mapping = self._build_entries_negative_mapping() if self.mode == "triplet_loss" else None

        self.randomizer = Random(seed)

    def __len__(self):
        if self.mode == "triplet_loss":
            return len(self.positive_pairs)
        elif self.mode == "fixed_negative_sampling":
            return len(self.positive_pairs) + len(self.negative_pairs)
        else:
            return 2 * len(self.positive_pairs)

    def validate_mode(
        self,
        mode: Literal[
            "fixed_negative_sampling",
            "uniform_negative_sampling",
            "distance_based_negative_sampling",
            "triplet_loss",
        ],
    ) -> str:
        if mode not in [
            "fixed_negative_sampling",
            "uniform_negative_sampling",
            "distance_based_negative_sampling",
            "triplet_loss",
        ]:
            error_message = "Unsupported dataset mode is selected!"
            raise ValueError(error_message)

        if mode == "fixed_negative_sampling" and self.negative_pairs is None:
            raise ValueError("Expected 'negative_pairs' parameter for 'fixed_negative_sampling' mode, got None!")

        if (
            mode in ["triplet_loss", "uniform_negative_sampling", "distance_based_negative_sampling"]
            and self.references_metadata is None
        ):
            raise ValueError(f"Expected 'references_metadata' parameter for '{mode}' mode, got None!")

        if (
            mode in ["uniform_negative_sampling", "distance_based_negative_sampling"]
            and self.negative_probability_matrix is None
        ):
            raise ValueError(
                f"Expected '{mode}' parameter for 'distance_based_negative_sampling' mode, got None!",
            )

        return mode

    def __getitem__(
        self,
        index: int,
    ) -> dict | tuple[dict, bool]:
        if self.mode == "triplet_loss":
            return self._get_triplet(index)
        elif self.mode == "fixed_negative_sampling":
            return self._get_fixed_classification(index)
        elif self.mode in ["uniform_negative_sampling", "distance_based_negative_sampling"]:
            return self._get_probability_based_negative_sampling(index)
        else:
            raise ValueError("Unsupported mode!")

    def _get_fixed_classification(
        self,
        index,
    ) -> dict | tuple[dict, bool]:
        is_positive = index < len(self.positive_pairs)
        if is_positive:
            entry = self.positive_pairs.iloc[index]
        else:
            entry = self.negative_pairs.iloc[index % len(self.positive_pairs)]
        item = {
            "paper_title": entry["paper_title"],
            "paper_abstract": entry["paper_abstract"],
            "reference_title": entry["reference_title"],
            "reference_abstract": entry["reference_abstract"],
        }
        return (item, is_positive) if self.return_target else item

    def _get_probability_based_negative_sampling(
        self,
        index: int,
    ) -> dict | tuple[dict, bool]:
        is_positive = index < len(self.positive_pairs)
        if is_positive:
            entry = self.positive_pairs.iloc[index]
            item = {
                "paper_title": entry["paper_title"],
                "paper_abstract": entry["paper_abstract"],
                "reference_title": entry["reference_title"],
                "reference_abstract": entry["reference_abstract"],
            }
        else:
            entry = self.positive_pairs.iloc[index % len(self.positive_pairs)]
            try:
                negative_index = int(
                    torch.multinomial(
                        self.negative_probability_matrix[self.paper_arxiv_id_to_index_mapping[entry["paper_arxiv_id"]]],
                        1,
                    ),
                )
            except Exception as e:
                logger.debug(
                    self.negative_probability_matrix[self.paper_arxiv_id_to_index_mapping[entry["paper_arxiv_id"]]],
                )
                raise e

            negative_reference = self.references_metadata.loc[self.index_to_references_arxiv_id_mapping[negative_index]]
            item = {
                "paper_title": entry["paper_title"],
                "paper_abstract": entry["paper_abstract"],
                "reference_title": negative_reference["title"],
                "reference_abstract": negative_reference["abstract"],
            }
        return (item, is_positive) if self.return_target else item

    def _get_triplet(
        self,
        index,
    ) -> dict:
        positive_entry = self.positive_pairs.iloc[index]
        anchor_arxiv_id = positive_entry["paper_arxiv_id"]
        negative_indexes = self.negative_mapping[anchor_arxiv_id]
        negative_arxiv_id = self.randomizer.choice(negative_indexes)

        return {
            "anchor_arxiv_id": anchor_arxiv_id,
            "anchor_title": positive_entry["paper_title"],
            "anchor_abstract": positive_entry["paper_abstract"],
            "positive_arxiv_id": positive_entry["reference_arxiv_id"],
            "positive_title": positive_entry["reference_title"],
            "positive_abstract": positive_entry["reference_abstract"],
            "negative_arxiv_id": negative_arxiv_id,
            "negative_title": self.references_metadata.loc[negative_arxiv_id]["title"],
            "negative_abstract": self.references_metadata.loc[negative_arxiv_id]["abstract"],
        }

    @log_with_message("building negative mapping")
    def _build_entries_negative_mapping(
        self,
    ) -> dict[str, list]:
        all_references = set(self.references_metadata.index.to_list())
        paper_to_difference = (
            self.positive_pairs.groupby("paper_arxiv_id")["reference_arxiv_id"]
            .agg(lambda x: self._get_difference(x, all_references))
            .to_dict()
        )
        return paper_to_difference

    def _collate_fn(self, batch_data: list):
        if self.mode == "triplet_loss":
            return self._triplet_collate_fn(batch_data)
        else:
            return self._classification_collate_fn(batch_data)

    def _classification_collate_fn(
        self,
        batch_data: list,
    ) -> tuple[dict[str, BatchEncoding], torch.LongTensor] | dict[str, BatchEncoding]:

        if self.return_target:
            items = [x[0] for x in batch_data]
            targets = torch.LongTensor([x[1] for x in batch_data])
        else:
            items = batch_data

        paper_abstracts = [x["paper_abstract"] for x in items]
        paper_titles = [x["paper_title"] for x in items]
        reference_abstracts = [x["reference_abstract"] for x in items]
        reference_titles = [x["reference_title"] for x in items]

        encodings = {}
        if self.title_process_mode == "separate":
            items_to_encode = [[paper_abstracts], [paper_titles], [reference_abstracts], [reference_titles]]
            item_names = ["paper_text", "paper_title", "reference_text", "reference_title"]
        elif self.title_process_mode == "combined":
            items_to_encode = [[paper_titles, paper_abstracts], [reference_titles, reference_abstracts]]
            item_names = ["paper_text", "reference_text"]
        else:
            raise ValueError("The mode of the title processing should be either 'separate' or 'combined'!")

        for item_to_encode, item_name in zip(items_to_encode, item_names, strict=True):
            item_encoding = self.tokenizer(
                *item_to_encode,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encodings[item_name] = item_encoding

        return encodings, targets if self.return_target else encodings

    def _triplet_collate_fn(
        self,
        batch_data: list,
    ) -> dict[str, BatchEncoding]:

        anchor_abstracts = [x["anchor_abstract"] for x in batch_data]
        anchor_titles = [x["anchor_title"] for x in batch_data]
        positive_abstracts = [x["positive_abstract"] for x in batch_data]
        positive_titles = [x["positive_title"] for x in batch_data]
        negative_abstracts = [x["negative_abstract"] for x in batch_data]
        negative_titles = [x["negative_title"] for x in batch_data]

        encodings = {}
        if self.title_process_mode == "separate":
            items_to_encode = [
                [anchor_abstracts],
                [anchor_titles],
                [positive_abstracts],
                [positive_titles],
                [negative_abstracts],
                [negative_titles],
            ]
            item_names = (
                ["anchor_text", "anchor_title", "positive_text", "positive_title", "negative_text", "negative_title"],
            )
        elif self.title_process_mode == "combined":
            items_to_encode = [
                [anchor_titles, anchor_abstracts],
                [positive_titles, positive_abstracts],
                [negative_titles, negative_abstracts],
            ]
            item_names = ["anchor_text", "positive_text", "negative_text"]
        else:
            raise ValueError("The mode of the title processing should be either 'separate' or 'combined'!")

        for item_to_encode, item_name in zip(items_to_encode, item_names, strict=True):
            item_encoding = self.tokenizer(
                *item_to_encode,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encodings[item_name] = item_encoding

        return encodings

    @staticmethod
    def _get_difference(
        series: pd.Series,
        source: set,
    ) -> list:
        return list(source - set(series))
