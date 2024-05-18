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
        tokenizer: PreTrainedTokenizer | None = None,
    ):
        super(InferenceArxivDataset, self).__init__()
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=512)
        else:
            self.tokenizer = tokenizer

        self.embeddings_mode = self._validate_and_process_mode(mode)
        self.object_to_return = object_to_return

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

    def _collate_fn(
        self,
        batch_data: list,
        title_process_mode: Literal["separate", "combined"] = "separate",
    ) -> dict[str, BatchEncoding]:
        return (
            self._embedding_collate_fn(batch_data, title_process_mode)
            if self.embeddings_mode
            else self._classification_collate_fn(batch_data, title_process_mode)
        )

    def _classification_collate_fn(
        self,
        batch_data: list,
        title_process_mode: Literal["separate", "combined"] = "separate",
    ) -> dict[str, BatchEncoding]:
        paper_abstracts = [x["paper_abstract"] for x in batch_data]
        paper_titles = [x["paper_title"] for x in batch_data]
        reference_abstracts = [x["reference_abstract"] for x in batch_data]
        reference_titles = [x["reference_title"] for x in batch_data]

        # TODO: add reference features
        if title_process_mode == "separate":
            paper_abstracts_encodings = self.tokenizer(
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
                "paper_text": paper_abstracts_encodings,
                "paper_title": paper_titles_encodings,
                "reference_text": reference_abstracts_encodings,
                "reference_title": reference_titles_encodings,
                # "reference_features": reference_features,
            }
            return encodings

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
            return encodings

        else:
            raise ValueError("The mode of the title processing should be either 'separate' or 'combined'!")

    def _embedding_collate_fn(
        self,
        batch_data: list,
        title_process_mode: Literal["separate", "combined"] = "separate",
    ) -> dict[str, BatchEncoding]:
        abstracts = [x["abstract"] for x in batch_data]
        titles = [x["title"] for x in batch_data]

        if title_process_mode == "separate":
            abstract_encodings = self.tokenizer(
                abstracts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            titles_encodings = self.tokenizer(
                titles,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encodings = {
                "paper_text": abstract_encodings,
                "paper_title": titles_encodings,
            }
        elif title_process_mode == "combined":
            encodings = self.tokenizer(
                titles,
                abstracts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encodings = {
                "paper_text": encodings,
            }
        else:
            raise ValueError("The mode of the title processing should be either 'separate' or 'combined'!")

        return encodings
