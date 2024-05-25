import pandas as pd

import wandb

from references_searcher.data.sql import DatabaseInterface
from references_searcher.models import CustomCatboostClassifier
from references_searcher.pipelines import BasePipeline
from references_searcher.metrics import evaluate_predictions


class CatboostPipeline(BasePipeline):
    def __init__(
        self,
        database_interface: DatabaseInterface,
    ):
        super().__init__(database_interface)
        self.positive_df["target"] = 1
        self.negative_df["target"] = 0

    def run(
        self,
        config: dict,
        watcher: str | None,
    ):
        model_train_config = config["model"]

        # Sanity check for final evaluation
        if (
            config["evaluate_at_k"]
            and model_train_config["data"]["val_size"] is None
            or model_train_config["data"]["val_size"] == 0
        ):
            raise ValueError("Requested evaluation at k, but no validation data is provided!")

        if model_train_config["data"]["cutoff"] is not None:
            positive_df = self.positive_df.iloc[: model_train_config["data"]["cutoff"]]
            negative_df = self.negative_df.iloc[: model_train_config["data"]["cutoff"]]
        else:
            positive_df = self.positive_df
            negative_df = self.negative_df

        # Preparing data for the mode
        validation_present = (
            model_train_config["data"]["val_size"] is not None and model_train_config["data"]["val_size"] != 0
        )
        if validation_present:
            train_positive_df, train_negative_df, val_positive_df, val_negative_df = self._classic_train_test_split(
                positive_df,
                negative_df,
                model_train_config["data"]["val_size"],
                config["random_seed"],
            )
            X_train, y_train = self._prepare_data(train_positive_df, train_negative_df)
            X_val, y_val = self._prepare_data(val_positive_df, val_negative_df)

        else:
            X_train, y_train = self._prepare_data(positive_df, negative_df)

        # Model training
        task_type = "GPU" if config["use_cuda_for_train"] else "CPU"
        model = CustomCatboostClassifier(task_type=task_type, random_state=config["random_seed"])
        model.fit(X_train, y_train, self.references_metadata, verbose=True)

        scores_to_commit = {}

        # Scores obtaining
        train_predictions = model._validation_predict(X_train)
        train_proxy_scores = evaluate_predictions(train_predictions, y_train)
        scores_to_commit["train"] = train_proxy_scores

        if validation_present:
            val_predictions = model._validation_predict(X_val)
            val_proxy_scores = evaluate_predictions(val_predictions, y_val)
            scores_to_commit["val"] = val_proxy_scores

        if config["evaluate_at_k"]:
            precision_at_k_score, recall_at_k_score = self._evaluate_at_k(
                self.references_metadata,
                val_positive_df,
                model,
                batch_size=model_train_config["batch_size"],
                n_predictions=config["inference"]["n_predictions"],
                n_candidates=config["inference"]["n_candidates"],
            )
            scores_to_commit[f'precision@{config["inference"]["n_predictions"]}'] = precision_at_k_score
            scores_to_commit[f'recall@{config["inference"]["n_predictions"]}'] = recall_at_k_score

        if watcher == "wandb":
            wandb.log(scores_to_commit)

        model.save_model(model_train_config["save_prefix"])

    @staticmethod
    def _prepare_data(
        positive_df: pd.DataFrame,
        negative_df: pd.DataFrame,
        columns_to_choose: list | None = None,
        target_column_name: str = "target",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if columns_to_choose is None:
            columns_to_choose = [
                "paper_title",
                "paper_abstract",
                "reference_title",
                "reference_abstract",
                "target",
            ]

        combined_df = pd.concat(
            [positive_df[columns_to_choose], negative_df[columns_to_choose]],
            ignore_index=True,
        )
        X_data = combined_df.drop(target_column_name, axis=1)
        y_data = combined_df[target_column_name]

        return X_data, y_data
