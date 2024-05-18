import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from typing import Literal
from collections.abc import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer

from references_searcher import logger


class CustomCatboostClassifier:
    def __init__(
        self,
        text_features: list[str] | None = None,
        category_features: list[str] | None = None,
        min_frequency: int | None = None,
        task_type: Literal["CPU", "GPU"] = "GPU",
        vectorizer: Literal["tfidf"] = "tfidf",
    ) -> None:
        if task_type not in ["CPU", "GPU"]:
            raise AttributeError(f"Unsupported task_type: {task_type}!")

        if text_features is None:
            text_features = ["paper_title", "paper_abstract", "reference_title", "reference_abstract"]

        if category_features is None:
            category_features = ["reference_category_1", "reference_category_2"]

        self.min_frequency = min_frequency
        self.vectorizer = None
        self._item_embeddings = None
        self.all_objects_metadata = None
        self.catboost_model = CatBoostClassifier(
            text_features=text_features,
            cat_features=category_features,
            task_type=task_type,
        )

        if vectorizer == "tfidf":
            self.vectorizer = TfidfVectorizer()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        all_objects_metadata: pd.DataFrame,
        metadata_text_features: list[str] | None = None,
        verbose: bool = True,
    ) -> None:

        if metadata_text_features is None:
            metadata_text_features = ["title", "abstract"]

        logger.debug("Processing categories into separate ones...")
        logger.debug("Successfully processed!")

        logger.info("Fitting the catboost model...")
        self.catboost_model.fit(X_train, y_train, verbose=verbose)
        logger.info("Successfully fitted!")

        logger.info("Fitting vectorizer...")
        self.all_objects_metadata = all_objects_metadata
        plain_text = self._collapse_text_features(all_objects_metadata, metadata_text_features)
        self._item_embeddings = self.vectorizer.fit_transform(plain_text).toarray()[np.newaxis, :, :]
        logger.info("Successfully fitted!")

    def predict(
        self,
        X_test: pd.DataFrame,
        text_features: list[str] | None = None,
        n_predictions: int = 1,
        n_candidates: int = 2,
    ):
        if text_features is None:
            text_features = ["title", "abstract"]

        if n_predictions > len(self.item_embeddings):
            logger.info(
                "The provided number of predictions per entry is too large,"
                f" setting it to train set size: {len(self.item_embeddings)}",
            )
            n_predictions = len(self.item_embeddings)

        if n_candidates > len(self.item_embeddings):
            logger.info(
                "The provided number of candidates is too large,"
                f" setting it to train set size: {len(self.item_embeddings)}",
            )
            n_candidates = len(self.item_embeddings)

        if n_predictions > n_candidates:
            raise AttributeError(
                f"The number of predictions cannot be larger than the number of"
                f" candidates, got {n_predictions} vs {n_candidates}!",
            )

        # TODO: add categorical features handling if needed.
        test_plain_text = self._collapse_text_features(X_test, text_features)
        target_embeddings = self.vectorizer.transform(test_plain_text).toarray()
        object_distances = self._calculate_euclidean_distance(target_embeddings)
        closest_items = object_distances.argsort(axis=1)[:, :n_candidates]
        candidate_pairs = self._generate_pairs(closest_items, X_test, self.all_objects_metadata)
        pair_probabilities = self.catboost_model.predict_proba(candidate_pairs)[:, 1].reshape(closest_items.shape)
        row_indices = np.arange(closest_items.shape[0])
        predicted_items = closest_items[row_indices[:, None], pair_probabilities.argsort(axis=1)[:, -n_predictions:]]
        return predicted_items

    @staticmethod
    def _generate_pairs(
        closest_items: np.ndarray,
        X_test: pd.DataFrame,
        all_objects_metadata: pd.DataFrame,
    ):
        resulting_rows = []
        for i, indices in enumerate(closest_items):
            # Extracting the row from df1 and replicating it by the number of indices
            X_test_replicated = X_test.add_prefix("paper_").loc[np.full(len(indices), i)].reset_index(drop=True)

            # Extracting rows from df2 using indices from numpy_array
            rows_from_all_objects_metadata = (
                all_objects_metadata.add_prefix("reference_").loc[indices].reset_index(drop=True)
            )

            # Concatenate replicated_df1 with rows_df2
            combined_df = pd.concat([X_test_replicated, rows_from_all_objects_metadata], axis=1)
            resulting_rows.append(combined_df)

        # Concatenating all results into a single DataFrame
        return pd.concat(resulting_rows, ignore_index=True)

    def _validation_predict(
        self,
        X_val: pd.DataFrame,
    ):
        logger.info(f"Making validating predictions for {len(X_val)} objects...")
        predictions = self.catboost_model.predict(X_val)
        logger.info("Validating predictions generated!")
        return predictions

    def _calculate_euclidean_distance(
        self,
        target: np.ndarray,
    ) -> np.ndarray:
        return np.linalg.norm(self._item_embeddings - np.atleast_2d(target)[:, np.newaxis, :], axis=2)

    @staticmethod
    def _collapse_text_features(
        data: pd.DataFrame,
        text_features: list[str],
    ) -> pd.DataFrame:
        result = data[text_features]
        result = result.apply(lambda row: " ".join(str(x) if x is not None else "" for x in row), axis=1)
        return result

    @staticmethod
    def _process_large_categories(
        data: pd.DataFrame,
        category_features: str | list[str] | None = None,
    ) -> (pd.DataFrame, list[str]):

        if not isinstance(category_features, Iterable):
            category_features = [category_features]

        result_data = data.copy(deep=True)
        result_columns = []

        print(data.columns)

        for category_feature in category_features:
            new_columns = [f"{category_feature}_large", f"{category_feature}_small"]
            result_columns.extend(new_columns)

            result_data[new_columns] = (
                result_data[category_feature]
                .str.split(
                    ".",
                    expand=True,
                )
                .fillna("")
            )

        return result_data.drop(category_features, axis=1), result_columns

    @property
    def item_embeddings(self):
        if self._item_embeddings is None:
            return None

        return self._item_embeddings[0, :, :]
