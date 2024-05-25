import pandas as pd

from sklearn.model_selection import train_test_split

from references_searcher.data.sql import DatabaseInterface
from references_searcher.utils import log_with_message
from references_searcher.models import CustomCatboostClassifier, CustomBert, Inferencer
from references_searcher.metrics import precision_at_k, recall_at_k
from references_searcher import logger


class BasePipeline:
    def __init__(
        self,
        database_interface: DatabaseInterface,
    ):
        # Get data from database
        self.positive_df = database_interface.get_positive_references()
        self.negative_df = database_interface.get_negative_references()
        self.references_metadata = database_interface.get_references_metadata()

    @staticmethod
    def _classic_train_test_split(
        positive_df: pd.DataFrame,
        negative_df: pd.DataFrame,
        val_percentage: float,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_positive_df, val_positive_df = train_test_split(
            positive_df,
            test_size=val_percentage,
            random_state=random_state,
        )
        train_negative_df, val_negative_df = train_test_split(
            negative_df,
            test_size=val_percentage,
            random_state=random_state,
        )
        return train_positive_df, train_negative_df, val_positive_df, val_negative_df

    @classmethod
    @log_with_message("evaluating metrics at k")
    def _evaluate_at_k(
        cls,
        references_metadata: pd.DataFrame,
        val_positive_df: pd.DataFrame,
        model: CustomBert | CustomCatboostClassifier,
        batch_size: int,
        n_predictions: int,
        n_candidates: int,
    ) -> tuple[float, float]:
        inferencer = Inferencer(
            model,
            batch_size=batch_size,
            n_predictions=n_predictions,
            n_candidates=n_candidates,
        )

        inferencer.fit(
            references_metadata,
            prefer_saved_matrix=False,
        )

        val_positive_df = val_positive_df.rename(columns={"paper_title": "title", "paper_abstract": "abstract"})
        test_items = val_positive_df[["title", "abstract", "paper_arxiv_id"]].drop_duplicates()

        reference_dict = val_positive_df.groupby("paper_arxiv_id")["reference_arxiv_id"].apply(list).to_dict()
        true_references = [reference_dict[paper_id] for paper_id in test_items["paper_arxiv_id"]]

        predictions = []
        for i in range(0, len(test_items), batch_size):
            batch = test_items.iloc[i : i + batch_size][["title", "abstract"]]
            predictions.extend(
                [[x.arxiv_id for x in y] for y in inferencer.predict(batch, return_title=False)],
            )

        precision_at_k_score = precision_at_k(predictions, true_references, k=n_predictions)
        recall_at_k_score = recall_at_k(predictions, true_references, k=n_predictions)

        logger.info(
            f"Evaluated scores:"
            f" precision@{n_predictions}={precision_at_k_score}; recall@{n_predictions}={recall_at_k_score}",
        )

        return precision_at_k_score, recall_at_k_score
