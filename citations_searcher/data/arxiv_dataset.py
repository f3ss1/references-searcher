from typing import Literal
from random import Random

import pandas as pd

from transformers import BertTokenizer, PreTrainedTokenizer
from torch.utils.data import Dataset
import torch

from citations_searcher import logger
from citations_searcher.utils import log_with_message


class ArxivDataset(Dataset):
    def __init__(
        self,
        positive_pairs: pd.DataFrame,
        mode: Literal["classification", "triplet"] = "classification",
        negative_pairs: pd.DataFrame | None = None,
        references_metadata: pd.DataFrame | None = None,
        return_target: bool = True,
        tokenizer: PreTrainedTokenizer | None = None,
        seed: int = 42,
    ):
        super(ArxivDataset, self).__init__()
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=512)
        else:
            self.tokenizer = tokenizer

        self.return_target = return_target
        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs
        self.references_metadata = references_metadata

        self.triplet_mode = False
        self.negative_mapping = None
        self.change_mode(mode)

        self.reference_features_names = ["reference_citation_count", "reference_influential_citation_count"]
        self.randomizer = Random(seed)

    def __len__(self):
        if self.triplet_mode:
            return len(self.positive_pairs)
        return len(self.positive_pairs) + len(self.negative_pairs)

    def change_mode(
        self,
        mode: Literal["triplet", "classification"],
        negative_pairs: pd.DataFrame | None = None,
        references_metadata: pd.DataFrame | None = None,
    ):
        # Sanity checks
        if mode not in ["triplet", "classification"]:
            error_message = "The mode of the dataset should be either 'triplet' or 'classification'!"
            raise ValueError(error_message)
        if mode == "classification" and self.negative_pairs is None and negative_pairs is None:
            raise ValueError("Expected 'negative_pairs' parameter for 'classification' mode, got None!")
        if mode == "triplet" and self.references_metadata is None and references_metadata is None:
            raise ValueError("Expected 'references_metadata' parameter for 'triplet' mode, got None!")

        # Attribute override handling
        if mode == "classification":
            if negative_pairs is not None:
                logger.info("Updating negative pairs, caused by mode switch with provided parameter.")
                self.negative_pairs = negative_pairs
            if references_metadata is not None:
                logger.warning(
                    "Mode is being switched to 'classification', but updated"
                    " 'references_metadata' is provided, ignoring the latter!",
                )
        if mode == "triplet":
            if references_metadata is not None:
                logger.info("Updating references metadata, caused by mode switch with provided parameter.")
                self.references_metadata = references_metadata
            if negative_pairs is not None:
                logger.warning(
                    "Mode is being switched to 'triplet', but updated"
                    " 'negative_pairs' is provided, ignoring the latter!",
                )

        # Mode change
        self.triplet_mode = mode == "triplet"
        self.negative_mapping = (
            self._build_entries_negative_mapping() if self.triplet_mode and self.negative_mapping is None else None
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict | tuple[dict, bool]:
        if not self.triplet_mode:
            is_positive = index < len(self.positive_pairs)
            if is_positive:
                item = self.positive_pairs.iloc[index].to_dict()
            else:
                item = self.negative_pairs.iloc[index - len(self.positive_pairs)].to_dict()
            return (item, is_positive) if self.return_target else item

        else:
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

    # TODO: можно сделать чтобы тут были все кто не в позитиве, а ссылка шла на общую таблицу
    @log_with_message("building negative mapping")
    def _build_entries_negative_mapping(
        self,
    ) -> dict[str, list]:
        all_references = set(self.positive_pairs["reference_arxiv_id"].to_list())
        paper_to_difference = (
            self.positive_pairs.groupby("paper_arxiv_id")["reference_arxiv_id"]
            .agg(lambda x: self._get_difference(x, all_references))
            .to_dict()
        )
        return paper_to_difference

    def _collate_fn(
        self,
        batch_data: list,
        title_process_mode: Literal["separate", "combined"] = "separate",
    ):
        if not self.triplet_mode:
            return self._classification_collate_fn(batch_data, title_process_mode)
        else:
            return self._triplet_collate_fn(batch_data, title_process_mode)

    def _classification_collate_fn(
        self,
        batch_data: list,
        title_process_mode: Literal["separate", "combined"] = "separate",
    ) -> tuple[dict[str, dict], torch.LongTensor] | dict[str, dict]:
        if self.return_target:
            items = [x[0] for x in batch_data]
            targets = torch.LongTensor([x[1] for x in batch_data])
        else:
            items = batch_data

        paper_abstracts = [x["paper_abstract"] for x in items]
        paper_titles = [x["paper_title"] for x in items]
        reference_abstracts = [x["reference_abstract"] for x in items]
        reference_titles = [x["reference_title"] for x in items]
        # TODO: these features have different scales, we need to process them somehow. Batchnorm?
        reference_features = torch.FloatTensor(
            [x[feature_name] for feature_name in self.reference_features_names for x in items],
        )

        # TODO: add reference features
        if title_process_mode == "separate":
            paper_abstract_encodings = self.tokenizer(
                paper_abstracts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            paper_titles_encodings = self.tokenizer(
                paper_titles,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            reference_abstracts_encodings = self.tokenizer(
                reference_abstracts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            reference_titles_encodings = self.tokenizer(
                reference_titles,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encodings = {
                "paper_text": paper_abstract_encodings,
                "paper_title": paper_titles_encodings,
                "reference_text": reference_abstracts_encodings,
                "reference_title": reference_titles_encodings,
                # "reference_features": reference_features,
            }
            return encodings, targets if self.return_target else encodings

        elif title_process_mode == "combined":
            paper_encodings = self.tokenizer(
                paper_titles,
                paper_abstracts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            reference_encodings = self.tokenizer(
                reference_titles,
                reference_abstracts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encodings = {
                "paper_text": paper_encodings,
                "reference_text": reference_encodings,
                # "reference_features": reference_features,
            }
            return encodings, targets if self.return_target else encodings

        else:
            raise ValueError("The mode of the title processing should be either 'separate' or 'combined'!")

    def _triplet_collate_fn(
        self,
        batch_data: list,
        title_process_mode: Literal["separate", "combined"] = "separate",
    ) -> tuple[dict[str, dict], torch.LongTensor] | dict[str, dict]:

        anchor_abstracts = [x["anchor_abstract"] for x in batch_data]
        anchor_titles = [x["anchor_title"] for x in batch_data]
        positive_abstracts = [x["positive_abstract"] for x in batch_data]
        positive_titles = [x["positive_title"] for x in batch_data]
        negative_abstracts = [x["negative_abstract"] for x in batch_data]
        negative_titles = [x["negative_title"] for x in batch_data]

        if title_process_mode == "separate":
            anchor_abstract_encodings = self.tokenizer(
                anchor_abstracts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            anchor_titles_encodings = self.tokenizer(
                anchor_titles,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            positive_abstract_encodings = self.tokenizer(
                positive_abstracts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            positive_titles_encodings = self.tokenizer(
                positive_titles,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            negative_abstracts_encodings = self.tokenizer(
                negative_abstracts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            negative_titles_encodings = self.tokenizer(
                negative_titles,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encodings = {
                "anchor_text": anchor_abstract_encodings,
                "anchor_title": anchor_titles_encodings,
                "positive_text": positive_abstract_encodings,
                "positive_title": positive_titles_encodings,
                "negative_text": negative_abstracts_encodings,
                "negative_title": negative_titles_encodings,
            }

        elif title_process_mode == "combined":
            anchor_encodings = self.tokenizer(
                anchor_titles,
                anchor_abstracts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            positive_encodings = self.tokenizer(
                positive_titles,
                positive_abstracts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            negative_encodings = self.tokenizer(
                negative_titles,
                negative_abstracts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encodings = {
                "anchor_text": anchor_encodings,
                "positive_text": positive_encodings,
                "negative_text": negative_encodings,
            }

        else:
            raise ValueError("The mode of the title processing should be either 'separate' or 'combined'!")

        return encodings

    @staticmethod
    def _get_difference(
        series: pd.Series,
        source: set,
    ) -> list:
        return list(source - set(series))
