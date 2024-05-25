from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BertTokenizer, BatchEncoding


class InferenceArxivDataset(Dataset):
    def __init__(
        self,
        object_to_return: pd.DataFrame | torch.Tensor,
        mode: Literal["classification", "embeddings"] = "classification",
        title_process_mode: Literal["separate", "combined"] = "combined",
        tokenizer: PreTrainedTokenizer | None = None,
    ):
        super(InferenceArxivDataset, self).__init__()
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=512)
        else:
            self.tokenizer = tokenizer

        self.embeddings_mode = self._validate_and_process_mode(mode)
        self.object_to_return = object_to_return
        self.title_process_mode = title_process_mode

    def __getitem__(
        self,
        index: int,
    ) -> dict | tuple[dict, bool]:
        if isinstance(self.object_to_return, pd.DataFrame):
            return self.object_to_return.iloc[index]
        return self.object_to_return[index]

    def __len__(self):
        return len(self.object_to_return)

    @staticmethod
    def _validate_and_process_mode(
        mode: Literal["classification", "embeddings"],
    ) -> bool:
        if mode not in ["classification", "embeddings"]:
            error_message = "The mode of the dataset should be either 'classification' or 'embeddings'!"
            raise ValueError(error_message)
        return mode == "embeddings"

    def update_object_and_mode(
        self,
        new_object_to_return: pd.DataFrame | torch.Tensor,
        new_mode: Literal["classification", "embeddings"] = "classification",
    ):
        self.embeddings_mode = self._validate_and_process_mode(new_mode)
        self.object_to_return = new_object_to_return

    def _collate_fn(self, batch_data: list) -> dict[str, BatchEncoding]:
        return (
            self._embedding_collate_fn(batch_data)
            if self.embeddings_mode
            else self._classification_collate_fn(batch_data)
        )

    def _classification_collate_fn(
        self,
        batch_data: list,
    ) -> dict[str, BatchEncoding]:
        paper_abstracts = [x["paper_abstract"] for x in batch_data]
        paper_titles = [x["paper_title"] for x in batch_data]
        reference_abstracts = [x["reference_abstract"] for x in batch_data]
        reference_titles = [x["reference_title"] for x in batch_data]

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

        return encodings

    def _embedding_collate_fn(
        self,
        batch_data: list,
    ) -> dict[str, BatchEncoding]:
        abstracts = [x["abstract"] for x in batch_data]
        titles = [x["title"] for x in batch_data]

        encodings = {}
        if self.title_process_mode == "separate":
            items_to_encode = [[abstracts], [titles]]
            item_names = ["paper_text", "paper_title"]
        elif self.title_process_mode == "combined":
            items_to_encode = [[abstracts, titles]]
            item_names = ["paper_text"]
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
