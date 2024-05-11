from typing import Literal
from random import Random

import pandas as pd

from transformers import BertTokenizer, PreTrainedTokenizer
from torch.utils.data import Dataset
import torch


class ArxivDataset(Dataset):
    def __init__(
        self,
        positive_pairs: pd.DataFrame,
        negative_pairs: pd.DataFrame,
        mode: Literal["classification", "triplet"] = "classification",
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
        # TODO: fix mode setup
        self.mode = mode
        self.change_mode(mode)

        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs
        self.reference_features_names = ["reference_citation_count", "reference_influential_citation_count"]
        self.negative_mapping = self._build_entries_negative_mapping()
        self.randomizer = Random(seed)

    def __len__(self):
        if self.triplet_mode:
            return len(self.positive_pairs)
        return len(self.positive_pairs) + len(self.negative_pairs)

    def change_mode(
        self,
        mode: Literal["triplet", "classification"],
    ):
        if mode not in ["triplet", "classification"]:
            error_message = "The mode of the dataset should be either 'triplet' or 'classification'!"
            raise AttributeError(error_message)
        self.triplet_mode = mode == "triplet"

    def __getitem__(
        self,
        index: int,
    ) -> pd.Series | tuple[pd.Series, bool]:
        if not self.triplet_mode:
            is_positive = index < len(self.positive_pairs)
            if is_positive:
                item = self.positive_pairs.iloc[index]
            else:
                item = self.negative_pairs.iloc[index - len(self.positive_pairs)]
            return (item, is_positive) if self.return_target else item

        else:
            positive_entry = self.positive_pairs.iloc[index]
            anchor_arxiv_id = positive_entry["paper_arxiv_id"]
            negative_indexes = self.negative_mapping[anchor_arxiv_id]
            negative_index = self.randomizer.choice(negative_indexes)
            negative_entry = self.negative_pairs.iloc[negative_index]

            # TODO:

            raise NotImplementedError

    # TODO: можно сделать чтобы тут были все кто не в позитиве, а ссылка шла на общую таблицу
    def _build_entries_negative_mapping(
        self,
    ):
        return {}

    def _collate_fn(
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
        reference_features = torch.FloatTensor([x[self.reference_features_names] for x in items])

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
