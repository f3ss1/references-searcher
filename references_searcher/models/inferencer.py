from typing import Literal
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import numpy as np

from torch.utils.data import DataLoader

from references_searcher.models import CustomCatboostClassifier, CustomBert
from references_searcher.utils import verbose_iterator, generate_device, log_with_message, title_case
from references_searcher.data import InferenceArxivDataset
from references_searcher.exceptions import NotFittedError
from references_searcher.constants import PROJECT_ROOT
from references_searcher import logger


@dataclass
class ReferencePrediction:
    arxiv_id: str
    title: str = None

    @property
    def arxiv_link(self):
        return f"https://arxiv.org/abs/{self.arxiv_id}"


class Inferencer:
    def __init__(
        self,
        embedding_model: CustomCatboostClassifier | CustomBert,
        decision_model: CustomCatboostClassifier | CustomBert | None = None,
        title_process_mode: Literal["separate", "combined"] = "combined",
        batch_size: int = 1,
        n_predictions: int = 5,
        n_candidates: int = 10,
        device: torch.device | None = None,
    ) -> None:
        self.device = device if device is not None else generate_device()

        self.embedding_model = embedding_model
        if decision_model is None:
            self.decision_model = embedding_model

        self.title_process_mode = title_process_mode
        self.batch_size = batch_size
        self.n_predictions = n_predictions
        self.n_candidates = n_candidates

        self.references = None
        self.references_embeddings = None
        self.dataset = None
        self.is_fit = False

    @log_with_message("building embeddings for potential references")
    def fit(
        self,
        references: pd.DataFrame,
        prefer_saved_matrix: bool = True,
        references_embeddings_save_path: str | Path | None = None,
        verbose: bool = True,
    ):
        self.references = references
        if isinstance(references_embeddings_save_path, str):
            references_embeddings_save_path = Path(references_embeddings_save_path)

        absolute_file_path = None
        if references_embeddings_save_path is not None:
            if (
                references_embeddings_save_path.suffix == ".pth"
                and isinstance(
                    self.embedding_model,
                    CustomCatboostClassifier,
                )
                or references_embeddings_save_path.suffix == ".npy"
                and isinstance(
                    self.embedding_model,
                    CustomBert,
                )
            ):
                raise ValueError(
                    f"The embedding model is {type(self.embedding_model)},"
                    f" but path with '{references_embeddings_save_path.suffix}' extension is provided!",
                )
            absolute_file_path = PROJECT_ROOT / references_embeddings_save_path

        if prefer_saved_matrix:
            if absolute_file_path.exists():
                logger.info(f"Using saved embeddings matrix at path {absolute_file_path}.")
                if isinstance(self.embedding_model, CustomBert):
                    self.references_embeddings = torch.load(absolute_file_path)
                else:
                    self.references_embeddings = np.load(absolute_file_path)

                if len(self.references_embeddings) != len(references):
                    raise ValueError(
                        "The length of the loaded embeddings does not match the length of references.",
                    )

                self.is_fit = True
                return

            else:
                logger.warning(
                    f"Did not find a file with an embeddings matrix at path {absolute_file_path}"
                    f" while preferring saved, constructing it.",
                )

        self.references_embeddings = self._build_embeddings(
            references,
            verbose,
        )
        if references_embeddings_save_path is not None:
            if isinstance(self.references_embeddings, torch.Tensor):
                torch.save(self.references_embeddings, absolute_file_path)
            else:
                np.save(absolute_file_path, self.references_embeddings)
        self.is_fit = True

    def predict(
        self,
        target_objects: pd.DataFrame,
        n_predictions: int | None = None,
        n_candidates: int | None = None,
        verbose: bool = True,
        return_title: bool = True,
    ) -> list[list[ReferencePrediction]]:
        if not self.is_fit:
            raise NotFittedError(
                "The inferencer is not fit yet (the references embeddings are not calculated)."
                " Please call the `.fit` method first.",
            )

        if n_predictions is None:
            n_predictions = self.n_predictions
        if n_candidates is None:
            n_candidates = self.n_candidates

        if isinstance(self.embedding_model, CustomBert):
            predictions = self._predict_with_bert(target_objects, n_predictions, n_candidates, verbose)
        elif isinstance(self.embedding_model, CustomCatboostClassifier):
            predictions = self._predict_with_catboost(target_objects, n_predictions, n_candidates, verbose)
        else:
            raise ValueError("Unsupported prediction model chosen!")

        arxiv_ids = list(self.references.index)
        if return_title:
            titles = self.references["title"].tolist()
            capitalized_titles = [title_case(x) for x in titles]
            result = [
                [ReferencePrediction(arxiv_ids[index], capitalized_titles[index]) for index in row]
                for row in predictions
            ]

        else:
            result = [[ReferencePrediction(arxiv_ids[index]) for index in row] for row in predictions]
        return result

    # Predictions generation
    def _predict_with_bert(
        self,
        target_objects: pd.DataFrame,
        n_predictions: int = 5,
        n_candidates: int = 10,
        verbose: bool = True,
    ) -> np.ndarray:

        target_embeddings = self._build_embeddings(
            target_objects,
            verbose,
        )

        closest_items, pairs = self._get_bert_pairs_to_predict(
            target_embeddings,
            self.references_embeddings,
            n_candidates,
        )
        self.dataset.update_object_and_mode(pairs, "classification")
        dataloader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )
        result = []
        for items_batch in verbose_iterator(
            dataloader,
            verbose,
            leave=False,
            desc="Building references embeddings",
        ):
            items_batch = items_batch.to(self.device)
            predicted_probabilities = self.decision_model.predict_proba(items_batch)[:, 1]
            result.append(predicted_probabilities.cpu())

        pair_probabilities = torch.concat(result).reshape(closest_items.shape)
        row_indices = torch.arange(closest_items.shape[0])
        predicted_items = closest_items[
            row_indices[:, None],
            pair_probabilities.argsort(dim=1)[:, -n_predictions:],
        ].numpy()

        return predicted_items

    def _predict_with_catboost(
        self,
        target_objects: pd.DataFrame,
        n_predictions: int = 5,
        n_candidates: int = 10,
        verbose: bool = True,
    ) -> np.ndarray:
        target_embeddings = self._build_embeddings(
            target_objects,
            verbose,
        )
        closest_items, pairs = self._get_catboost_pairs_to_predict(
            target_embeddings,
            self.references_embeddings,
            target_objects,
            self.references,
            n_candidates,
        )

        pair_probabilities = self.decision_model.predict_proba(pairs)[:, 1].reshape(closest_items.shape)
        row_indices = np.arange(closest_items.shape[0])
        predicted_items = closest_items[
            row_indices[:, None],
            pair_probabilities.argsort(axis=1)[:, -n_predictions:],
        ]
        return predicted_items

    # Embeddings building
    def _build_embeddings(
        self,
        items_to_build_for: pd.DataFrame,
        verbose: bool = True,
    ) -> torch.Tensor | np.ndarray:
        if isinstance(self.embedding_model, CustomBert):
            return self._build_embeddings_with_bert(items_to_build_for, verbose)
        elif isinstance(self.embedding_model, CustomCatboostClassifier):
            return self._build_embeddings_with_catboost(items_to_build_for, verbose)
        else:
            raise ValueError("Unsupported embedding model chosen!")

    @torch.no_grad()
    def _build_embeddings_with_bert(
        self,
        items_to_build_for: pd.DataFrame,
        verbose: bool = True,
    ) -> torch.Tensor:

        if self.dataset is None:
            self.dataset = InferenceArxivDataset(items_to_build_for, mode="embeddings")
        else:
            self.dataset.update_object_and_mode(items_to_build_for, "embeddings")

        dataloader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda x: self.dataset._collate_fn(x),
        )
        result = []
        for items_batch in verbose_iterator(
            dataloader,
            verbose,
            leave=False,
            desc="Building references embeddings",
        ):
            items_batch = {key: tensor.to(self.device) for key, tensor in items_batch.items()}
            embeddings = self.embedding_model.get_embeddings(**items_batch)
            result.append(embeddings.cpu())

        result = torch.concat(result)
        return result

    def _build_embeddings_with_catboost(
        self,
        items_to_build_for: pd.DataFrame,
        verbose: bool = True,
    ) -> torch.Tensor:
        return self.embedding_model.get_embeddings(items_to_build_for, verbose)

    def _get_bert_pairs_to_predict(
        self,
        target_embeddings: torch.Tensor,
        references_embeddings: torch.Tensor,
        n_candidates: int,
    ):
        distances = self._calculate_pairwise_euclidean_distance(target_embeddings, references_embeddings)
        closest_items = distances.argsort(axis=1)[:, :n_candidates]
        result = []

        for i, indices in enumerate(closest_items):
            replicated_target = target_embeddings[torch.full([len(indices)], i)]
            rows_from_references_embeddings = references_embeddings[indices]
            combined_tensor = torch.concat([replicated_target, rows_from_references_embeddings], axis=1)
            result.append(combined_tensor)
        return closest_items, torch.concat(result)

    def _get_catboost_pairs_to_predict(
        self,
        target_embeddings,
        references_embeddings: torch.Tensor,
        target_metadata: pd.DataFrame,
        references_metadata: pd.DataFrame,
        n_candidates: int,
    ):
        distances = self._calculate_pairwise_euclidean_distance(target_embeddings, references_embeddings)
        closest_items = distances.argsort(axis=1)[:, :n_candidates]
        result = []

        for i, indices in enumerate(closest_items):
            replicated_target = (
                target_metadata.add_prefix("paper_").iloc[np.full(len(indices), i)].reset_index(drop=True)
            )
            rows_from_all_objects_metadata = (
                references_metadata.add_prefix("reference_").iloc[indices].reset_index(drop=True)
            )
            combined_df = pd.concat([replicated_target, rows_from_all_objects_metadata], axis=1)
            result.append(combined_df)

        return closest_items, pd.concat(result, ignore_index=True)

    @classmethod
    def _calculate_pairwise_euclidean_distance(
        cls,
        target: np.ndarray | torch.Tensor,
        source: np.ndarray | torch.Tensor,
    ):
        if isinstance(source, torch.Tensor):
            if isinstance(target, np.ndarray):
                target = torch.Tensor(target)
            return cls._torch_calculate_pairwise_euclidean_distance(target, source)
        elif isinstance(source, np.ndarray):
            if isinstance(target, torch.Tensor):
                target = target.numpy()
            return cls._numpy_calculate_pairwise_euclidean_distance(target, source)
        else:
            raise ValueError(
                f"Target value is expected to be either numpy array or pytorch tensor, got {type(target)}!",
            )

    @staticmethod
    def _numpy_calculate_pairwise_euclidean_distance(
        target: np.ndarray,
        source: np.ndarray,
    ) -> np.ndarray:
        return np.linalg.norm(np.atleast_2d(source) - np.atleast_2d(target)[:, np.newaxis, :], axis=2)

    @staticmethod
    def _torch_calculate_pairwise_euclidean_distance(
        target: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        if target.dim() == 1:
            target = target.unsqueeze(0)
        if source.dim() == 1:
            source = source.unsqueeze(0)
        target = target.unsqueeze(1)
        return torch.norm(source - target, dim=2)
