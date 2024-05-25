from pathlib import Path
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from typing import Literal

from references_searcher import logger
from references_searcher.models import Word2VecEmbeddings
from references_searcher.utils import log_with_message


class CustomCatboostClassifier:
    def __init__(
        self,
        text_features: list[str] | None = None,
        task_type: Literal["CPU", "GPU"] = "GPU",
        random_state: int = 42,
    ) -> None:
        if task_type not in ["CPU", "GPU"]:
            raise AttributeError(f"Unsupported task_type: {task_type}!")

        if text_features is None:
            text_features = ["paper_title", "paper_abstract", "reference_title", "reference_abstract"]

        self.catboost_model = CatBoostClassifier(
            text_features=text_features,
            task_type=task_type,
            random_state=random_state,
        )
        self.vectorizer = Word2VecEmbeddings(random_seed=random_state)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        references_metadata: pd.DataFrame,
        verbose: bool = True,
    ) -> None:

        with log_with_message("fitting catboost model"):
            self.catboost_model.fit(X_train, y_train, verbose=verbose)

        plain_text = self._preprocess_text_features(references_metadata)
        self.vectorizer.fit(plain_text)

    def predict_proba(
        self,
        candidate_pairs: pd.DataFrame,
    ):
        return self.catboost_model.predict_proba(candidate_pairs)

    def _validation_predict(
        self,
        X_val: pd.DataFrame,
    ):
        logger.info(f"Making validating predictions for {len(X_val)} objects...")
        predictions = self.catboost_model.predict(X_val)
        logger.info("Validating predictions generated!")
        return predictions

    def get_embeddings(
        self,
        items_to_build_for: pd.DataFrame,
        verbose: bool = True,
    ) -> np.ndarray:
        test_plain_text = self._preprocess_text_features(items_to_build_for)
        target_embeddings = self.vectorizer.transform(test_plain_text, verbose)
        return target_embeddings

    def save_model(
        self,
        file_prefix: str | Path,
    ):
        if isinstance(file_prefix, str):
            file_prefix = Path(file_prefix)

        self.vectorizer.save_model(f"{file_prefix}_vectorizer.model")
        self.catboost_model.save_model(f"{file_prefix}_catboost_model.cbm")

    def load_model(
        self,
        file_prefix: str | Path,
    ):
        if isinstance(file_prefix, str):
            file_prefix = Path(file_prefix)

        self.vectorizer.load_model(f"{file_prefix}_vectorizer.model")
        self.catboost_model.load_model(f"{file_prefix}_catboost_model.cbm")

    @staticmethod
    def _preprocess_text_features(
        data: pd.DataFrame,
    ) -> list[str]:
        text_features = ["title", "abstract"]
        result = data[text_features]
        result = result.apply(lambda row: " ".join(str(x) if x is not None else "" for x in row), axis=1).to_list()
        return result
