import pandas as pd

from sklearn.model_selection import train_test_split

from references_searcher.data.sql import DatabaseInterface
from references_searcher.models import CustomCatboostClassifier, Inferencer
from references_searcher.metrics import evaluate_predictions, precision_at_k, recall_at_k


def catboost_pipeline(
    database_interface: DatabaseInterface,
    config: dict,
):
    model_config = config["model"]

    positive_df = database_interface.get_positive_references(model_config["data"]["cutoff"])
    negative_df = database_interface.get_negative_references(model_config["data"]["cutoff"])
    metadata_df = database_interface.get_references_metadata()
    positive_df["target"] = 1
    negative_df["target"] = 0

    validation_present = model_config["data"]["val_size"] is not None and model_config["data"]["val_size"] != 0
    if validation_present:
        train_positive_df, val_positive_df = train_test_split(
            positive_df,
            test_size=model_config["data"]["val_size"],
            random_state=config["random_seed"],
        )
        train_negative_df, val_negative_df = train_test_split(
            negative_df,
            test_size=model_config["data"]["val_size"],
            random_state=config["random_seed"],
        )
        overall_train_df = pd.concat(
            [
                train_positive_df[
                    [
                        "paper_title",
                        "paper_abstract",
                        "reference_title",
                        "reference_abstract",
                        "target",
                    ]
                ],
                train_negative_df[
                    [
                        "paper_title",
                        "paper_abstract",
                        "reference_title",
                        "reference_abstract",
                        "target",
                    ]
                ],
            ],
            ignore_index=True,
        )
        X_train = overall_train_df.drop("target", axis=1)
        y_train = overall_train_df["target"]

        overall_val_df = pd.concat(
            [
                val_positive_df[
                    [
                        "paper_title",
                        "paper_abstract",
                        "reference_title",
                        "reference_abstract",
                        "target",
                    ]
                ],
                val_negative_df[
                    [
                        "paper_title",
                        "paper_abstract",
                        "reference_title",
                        "reference_abstract",
                        "target",
                    ]
                ],
            ],
            ignore_index=True,
        )
        X_val = overall_val_df.drop("target", axis=1)
        y_val = overall_val_df["target"]

    else:
        overall_train_df = pd.concat(
            [
                positive_df[
                    [
                        "paper_title",
                        "paper_abstract",
                        "reference_title",
                        "reference_abstract",
                        "target",
                    ]
                ],
                negative_df[
                    [
                        "paper_title",
                        "paper_abstract",
                        "reference_title",
                        "reference_abstract",
                        "target",
                    ]
                ],
            ],
            ignore_index=True,
        )
        X_train = overall_train_df.drop("target", axis=1)
        y_train = overall_train_df["target"]

    task_type = "GPU" if config["use_cuda_for_train"] else "CPU"
    model = CustomCatboostClassifier(task_type=task_type)
    model.fit(X_train, y_train, metadata_df, verbose=True)

    if validation_present:
        val_predictions = model._validation_predict(X_val)
        val_proxy_scores = evaluate_predictions(val_predictions, y_val)
        print(val_proxy_scores)

        # TODO: separate from inference.
        if config["evaluate_at_k"]:
            inferencer = Inferencer(
                model,
                batch_size=1,
                n_predictions=config["inference"]["n_predictions"],
                n_candidates=config["inference"]["n_candidates"],
            )

            inferencer.fit(
                metadata_df,
                prefer_saved_matrix=False,
            )

            val_positive_df = val_positive_df.rename(columns={"paper_title": "title", "paper_abstract": "abstract"})
            test_items = val_positive_df[["title", "abstract", "paper_arxiv_id"]].drop_duplicates()

            reference_dict = val_positive_df.groupby("paper_arxiv_id")["reference_arxiv_id"].apply(list).to_dict()
            true_references = [reference_dict[paper_id] for paper_id in test_items["paper_arxiv_id"]]

            predictions = []
            batch_size = 10
            for i in range(0, len(test_items), batch_size):
                # Get the batch of test items
                batch = test_items.iloc[i : i + batch_size][["title", "abstract"]]

                predictions.extend(
                    [[x.arxiv_id for x in y] for y in inferencer.predict(batch, return_title=False)],
                )

            print(len(test_items), len(predictions), len(true_references))

            print(
                precision_at_k(predictions, true_references, k=5),
                recall_at_k(predictions, true_references, k=5),
            )

    model.save_model(model_config["save_prefix"])
