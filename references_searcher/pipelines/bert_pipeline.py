from typing import Literal

import numpy as np
import pandas as pd
import wandb

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from references_searcher.pipelines import BasePipeline
from references_searcher.constants import PROJECT_ROOT
from references_searcher.data.sql import DatabaseInterface
from references_searcher.data import ArxivDataset, InferenceArxivDataset
from references_searcher.models import CustomBert, Trainer
from references_searcher.utils import verbose_iterator, log_with_message, generate_device
from references_searcher import logger

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class BertPipeline(BasePipeline):
    def __init__(
        self,
        database_interface: DatabaseInterface,
    ) -> None:
        super().__init__(database_interface)

    def run(
        self,
        config: dict,
        watcher: str | None,
    ) -> None:
        device = generate_device(config["use_cuda_for_train"])

        if config["model"]["pretrain"]["execute"]:
            self.pretrain(config, device, watcher)

        load_triplet_pretrained_bert = self._validate_triplet_pretrain(
            config["model"]["train"]["use_triplet_pretrain"],
            config["model"]["pretrain"]["execute"],
            config["model"]["pretrain"]["save_path"],
        )
        self.train(config, device, load_triplet_pretrained_bert, watcher)

    def pretrain(
        self,
        config: dict,
        device: torch.device,
        watcher: str | None,
    ) -> None:
        model_pretrain_config = config["model"]["pretrain"]

        if model_pretrain_config["data"]["cutoff"] is not None:
            positive_df = self.positive_df.iloc[: model_pretrain_config["data"]["cutoff"]]
        else:
            positive_df = self.positive_df

        pretrain_dataset = ArxivDataset(
            positive_df,
            mode="triplet_loss",
            references_metadata=self.references_metadata,
            seed=config["random_seed"],
        )

        pretrain_dataloader = DataLoader(
            pretrain_dataset,
            shuffle=True,
            collate_fn=pretrain_dataset._collate_fn,
            **model_pretrain_config["dataloader"],
        )

        model = CustomBert(**config["model"]["bert_model"])
        model.to(device)
        optimizer = AdamW(model.parameters(), **model_pretrain_config["optimizer"])

        trainer = Trainer(watcher=watcher, device=device)
        trainer.pretrain(model, optimizer, pretrain_dataloader, n_epochs=model_pretrain_config["n_epochs"])

        torch.save(model.bert.state_dict(), PROJECT_ROOT / model_pretrain_config["save_path"])

    def train(
        self,
        config: dict,
        device: torch.device,
        load_triplet_pretrained_bert: bool,
        watcher: str | None,
    ) -> None:
        model_train_config = config["model"]["train"]

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

        # Load the model
        model = CustomBert(**config["model"]["bert_model"])
        if load_triplet_pretrained_bert:
            model.bert.load_state_dict(
                torch.load(PROJECT_ROOT / config["model"]["pretrain"]["save_path"], map_location=device),
            )
        model.to(device)

        # Dataloaders creation logic
        validation_present = (
            model_train_config["data"]["val_size"] is not None and model_train_config["data"]["val_size"] != 0
        )
        if model_train_config["dataset_mode"] == "fixed_negative_sampling":
            if validation_present:
                train_positive_df, train_negative_df, val_positive_df, val_negative_df = self._classic_train_test_split(
                    positive_df,
                    negative_df,
                    model_train_config["data"]["val_size"],
                )
            else:
                train_positive_df = positive_df
                train_negative_df = negative_df
                val_positive_df = None
                val_negative_df = None

            train_dataloader, val_dataloader = self._get_fixed_dataloaders(
                train_positive_df,
                train_negative_df,
                val_positive_df,
                val_negative_df,
                **model_train_config["dataloaders"],
            )

        elif model_train_config["dataset_mode"] in ["uniform_negative_sampling", "distance_based_negative_sampling"]:
            embedding_based = model_train_config["dataset_mode"] == "distance_based_negative_sampling"
            negative_probability_matrix, paper_arxiv_id_to_index_mapping, references_arxiv_id_to_index_mapping = (
                self._build_probability_matrix(
                    positive_df,
                    self.references_metadata,
                    embedding_based,
                    model,
                    device,
                    **model_train_config["dataloaders"],
                )
            )

            if validation_present:
                (
                    train_positive_df,
                    train_negative_probability_matrix,
                    val_positive_df,
                    val_negative_probability_matrix,
                ) = self._matrix_based_train_test_split(
                    positive_df,
                    negative_probability_matrix,
                    model_train_config["data"]["val_size"],
                    random_state=config["random_seed"],
                )
            else:
                train_positive_df = positive_df
                train_negative_probability_matrix = negative_probability_matrix
                val_positive_df = None
                val_negative_probability_matrix = None

            train_dataloader, val_dataloader = self._get_sampling_dataloaders(
                train_positive_df,
                train_negative_probability_matrix,
                paper_arxiv_id_to_index_mapping,
                references_arxiv_id_to_index_mapping,
                self.references_metadata,
                model_train_config["dataset_mode"],
                val_positive_df=val_positive_df,
                val_negative_probability_matrix=val_negative_probability_matrix,
                random_state=config["random_seed"],
                n_negative=model_train_config["n_negative"],
                **model_train_config["dataloaders"],
            )
        else:
            raise ValueError("Wrong train dataset mode!")

        # Fit the model
        optimizer = AdamW(model.parameters(), **model_train_config["optimizer"])
        trainer = Trainer(watcher=watcher, device=device)
        trainer.train(
            model,
            optimizer,
            train_dataloader,
            val_dataloader=val_dataloader,
            n_epochs=model_train_config["n_epochs"],
        )

        torch.save(model.state_dict(), PROJECT_ROOT / model_train_config["save_path"])

        # Final evaluation if required
        if config["evaluate_at_k"]:
            precision_at_k_score, recall_at_k_score = self._evaluate_at_k(
                self.references_metadata,
                val_positive_df,
                model,
                batch_size=model_train_config["dataloaders"]["batch_size"],
                n_predictions=config["inference"]["n_predictions"],
                n_candidates=config["inference"]["n_candidates"],
            )
            if watcher == "wandb":
                wandb.log(
                    {
                        f'precision@{config["inference"]["n_predictions"]}': precision_at_k_score,
                        f'recall@{config["inference"]["n_predictions"]}': recall_at_k_score,
                    },
                )

    @classmethod
    @log_with_message("building probability matrix for negative sampling")
    def _build_probability_matrix(
        cls,
        positive_pairs: pd.DataFrame,
        references_metadata: pd.DataFrame,
        embedding_based: bool = True,
        model: torch.nn.Module | None = None,
        device: torch.device | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, int], dict[str, int]]:

        unique_papers = positive_pairs.drop_duplicates(subset=["paper_arxiv_id"]).reset_index(drop=True)[
            ["paper_arxiv_id", "paper_title", "paper_abstract"]
        ]
        paper_arxiv_id_to_index_mapping = unique_papers.reset_index().set_index("paper_arxiv_id")["index"].to_dict()
        references_metadata["old_index"] = np.arange(len(references_metadata))
        references_arxiv_id_to_index_mapping = references_metadata["old_index"].to_dict()
        references_metadata = references_metadata.drop("old_index", axis=1)

        if not embedding_based:
            distances = torch.ones([len(unique_papers), len(references_metadata)])
        else:
            if model is None:
                raise ValueError("No model provided for embedding-based probability matrix creation!")

            papers_embeddings = cls.get_embeddings(
                unique_papers[["paper_title", "paper_abstract"]].rename(
                    columns={"paper_title": "title", "paper_abstract": "abstract"},
                ),
                model,
                device,
                desc="Building papers embeddings",
                **kwargs,
            )
            references_embeddings = cls.get_embeddings(
                references_metadata[["title", "abstract"]],
                model,
                device,
                desc="Building references embeddings",
                **kwargs,
            )
            distances = torch.cdist(papers_embeddings, references_embeddings, p=2)

        probabilities = cls._distances_to_probabilities(
            distances,
            positive_pairs,
            paper_arxiv_id_to_index_mapping,
            references_arxiv_id_to_index_mapping,
        )

        return probabilities, paper_arxiv_id_to_index_mapping, references_arxiv_id_to_index_mapping

    @staticmethod
    def _distances_to_probabilities(
        distances: torch.Tensor,
        positive_pairs: pd.DataFrame,
        paper_arxiv_id_to_index_mapping: dict,
        references_arxiv_id_to_index_mapping: dict,
    ) -> torch.Tensor:

        for _, row in positive_pairs.iterrows():
            source_index = paper_arxiv_id_to_index_mapping[row["paper_arxiv_id"]]
            reference_index = references_arxiv_id_to_index_mapping[row["reference_arxiv_id"]]
            distances[source_index, reference_index] = torch.inf

        for arxiv_id, source_index in paper_arxiv_id_to_index_mapping.items():
            if arxiv_id in references_arxiv_id_to_index_mapping:
                distances[source_index, references_arxiv_id_to_index_mapping[arxiv_id]] = torch.inf

        # Absolute identities might occur and usually indicate presence of retracted and re-submitted papers.
        if (distances == 0).any():
            x_axis_dict = {value: key for key, value in paper_arxiv_id_to_index_mapping.items()}
            y_axis_dict = {value: key for key, value in references_arxiv_id_to_index_mapping.items()}
            array = distances.numpy()
            zero_coords = np.argwhere(array == 0)
            zero_name_pairs = [(x_axis_dict[x], y_axis_dict[y]) for x, y in zero_coords]
            zero_arxiv_ids_df = pd.DataFrame(zero_name_pairs, columns=["X_Axis", "Y_Axis"])
            logger.error(
                "Detected zeros in distances matrix! Might mean you have"
                " re-appearing data with various arxiv ids. Check your data!",
            )
            logger.error(zero_arxiv_ids_df)

            distances[distances == 0] = torch.inf

        probabilities = 1 / distances
        probabilities /= probabilities.sum(dim=1, keepdim=True)
        return probabilities

    @classmethod
    def _matrix_based_train_test_split(
        cls,
        positive_df: pd.DataFrame,
        probability_matrix: torch.Tensor,
        val_percentage: float,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, torch.Tensor, pd.DataFrame, torch.Tensor]:
        train_positive_df, val_positive_df = train_test_split(
            positive_df,
            test_size=val_percentage,
            random_state=random_state,
        )
        train_negative_probability_matrix, val_negative_probability_matrix = cls.split_probability_matrix(
            probability_matrix,
            val_percentage,
        )

        return train_positive_df, train_negative_probability_matrix, val_positive_df, val_negative_probability_matrix

    @staticmethod
    def split_probability_matrix(
        probability_matrix: torch.Tensor,
        val_percentage: float,
    ) -> [torch.Tensor, torch.Tensor]:
        train_probs = probability_matrix.clone()
        val_probs = torch.zeros_like(probability_matrix)

        # Iterate over each row to split non-zero items
        for i in range(probability_matrix.size(0)):
            row_probs = probability_matrix[i]
            non_zero_indices = torch.nonzero(row_probs, as_tuple=True)[0]
            num_non_zero_elements = non_zero_indices.size(0)
            num_val_elements = int(num_non_zero_elements * val_percentage)

            if num_val_elements > 0:
                selected_indices = non_zero_indices[torch.randperm(num_non_zero_elements)[:num_val_elements]]
                val_probs[i, selected_indices] = row_probs[selected_indices]
                train_probs[i, selected_indices] = 0

        # Rescale the train probabilities
        train_probs_sum = train_probs.sum(dim=1, keepdim=True)
        train_probs /= train_probs_sum + (train_probs_sum == 0).float()  # avoid division by zero

        # Rescale the validation probabilities
        val_probs_sum = val_probs.sum(dim=1, keepdim=True)
        val_probs /= val_probs_sum + (val_probs_sum == 0).float()  # avoid division by zero

        return train_probs, val_probs

    @staticmethod
    def _get_fixed_dataloaders(
        train_positive_df: pd.DataFrame,
        train_negative_df: pd.DataFrame,
        val_positive_df: pd.DataFrame | None = None,
        val_negative_df: pd.DataFrame | None = None,
        title_process_mode: Literal["separate", "combined"] = "combined",
        random_state: int = 42,
        **kwargs,
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader] | tuple[torch.utils.data.DataLoader, None]:
        train_dataset = ArxivDataset(
            train_positive_df,
            mode="fixed_negative_sampling",
            negative_pairs=train_negative_df,
            title_process_mode=title_process_mode,
            seed=random_state,
        )
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=train_dataset._collate_fn,
            **kwargs,
        )

        if val_positive_df is not None and val_negative_df is not None:
            val_dataset = ArxivDataset(
                val_positive_df,
                mode="fixed_negative_sampling",
                negative_pairs=val_negative_df,
                title_process_mode=title_process_mode,
                seed=random_state,
            )
            val_dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                collate_fn=val_dataset._collate_fn,
                **kwargs,
            )
            return train_dataloader, val_dataloader

        elif val_positive_df is not None or val_negative_df is not None:
            raise ValueError("Only one of the validation dataframes is provided!")

        return train_dataloader, None

    @staticmethod
    def _get_sampling_dataloaders(
        train_positive_df: pd.DataFrame,
        train_negative_probability_matrix: torch.Tensor,
        paper_arxiv_id_to_index_mapping: dict[str, int],
        references_arxiv_id_to_index_mapping: dict[str, int],
        references_metadata: pd.DataFrame,
        dataset_mode: Literal[
            "uniform_negative_sampling",
            "distance_based_negative_sampling",
        ] = "distance_based_negative_sampling",
        title_process_mode: Literal["separate", "combined"] = "combined",
        val_positive_df: pd.DataFrame | None = None,
        val_negative_probability_matrix: torch.Tensor | None = None,
        random_state: int = 42,
        n_negative: int = 1,
        **kwargs,
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader] | tuple[torch.utils.data.DataLoader, None]:
        train_dataset = ArxivDataset(
            train_positive_df,
            mode=dataset_mode,
            references_metadata=references_metadata,
            negative_probability_matrix=train_negative_probability_matrix,
            paper_arxiv_id_to_index_mapping=paper_arxiv_id_to_index_mapping,
            references_arxiv_id_to_index_mapping=references_arxiv_id_to_index_mapping,
            title_process_mode=title_process_mode,
            seed=random_state,
            n_negative=n_negative,
        )
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=train_dataset._collate_fn,
            **kwargs,
        )

        if val_positive_df is not None and val_negative_probability_matrix is not None:
            val_dataset = ArxivDataset(
                val_positive_df,
                mode=dataset_mode,
                references_metadata=references_metadata,
                negative_probability_matrix=val_negative_probability_matrix,
                paper_arxiv_id_to_index_mapping=paper_arxiv_id_to_index_mapping,
                references_arxiv_id_to_index_mapping=references_arxiv_id_to_index_mapping,
                title_process_mode=title_process_mode,
                n_negative=n_negative,
                seed=random_state,
            )
            val_dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                collate_fn=val_dataset._collate_fn,
                **kwargs,
            )
            return train_dataloader, val_dataloader

        elif val_positive_df is not None or val_negative_probability_matrix is not None:
            raise ValueError("Only one of the validation dataframes is provided!")

        return train_dataloader, None

    @staticmethod
    @torch.no_grad()
    def get_embeddings(
        items_to_build_for: pd.DataFrame,
        model: torch.nn.Module,
        device: torch.device,
        verbose: bool = True,
        desc: str = "",
        **kwargs,
    ) -> torch.Tensor:
        dataset = InferenceArxivDataset(items_to_build_for, mode="embeddings")
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            collate_fn=dataset._collate_fn,
            **kwargs,
        )

        result = []
        for items_batch in verbose_iterator(
            dataloader,
            verbose,
            leave=False,
            desc=desc,
        ):
            device_items_batch = {key: tensor.to(device) for key, tensor in items_batch.items()}
            embeddings = model.get_embeddings(**device_items_batch)
            result.append(embeddings.cpu())

        result = torch.concat(result)
        return result

    @staticmethod
    def _validate_triplet_pretrain(
        use_pretrain: bool,
        execute_pretrain: bool,
        pretrain_model_path: str | None,
    ) -> bool:
        if use_pretrain:
            if pretrain_model_path is None:
                raise ValueError(
                    "No `pretrained_model_path` is provided in the `pretrain` section of the config"
                    " while a pretrained model is requested!",
                )
            if not execute_pretrain:
                logger.warning("Using the previously triplet pretrained BERT model if available.")
            return True

        elif execute_pretrain:
            logger.warning(
                "The model is being pretrained, but is not later used in the train sequence! A mistake in the config?",
            )
        return False
