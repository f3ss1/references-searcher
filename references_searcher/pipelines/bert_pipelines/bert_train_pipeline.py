from sklearn.model_selection import train_test_split
import wandb

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from references_searcher.constants import PROJECT_ROOT
from references_searcher.data.sql import DatabaseInterface
from references_searcher.data import ArxivDataset
from references_searcher.models import CustomBert, Trainer


def train_bert(
    database_interface: DatabaseInterface,
    config: dict,
    device: torch.device,
    load_triplet_pretrained_bert: bool,
):
    model_train_config = config["model"]["train"]

    if config["model"]["watcher"]:
        wandb.init(
            project="references-searcher",
            config=config,
        )
        watcher = "wandb"
    else:
        watcher = None

    positive_df = database_interface.get_positive_references(model_train_config["data"]["cutoff"])
    negative_df = database_interface.get_negative_references(model_train_config["data"]["cutoff"])

    if model_train_config["data"]["val_size"] is not None and model_train_config["data"]["val_size"] != 0:
        train_positive_df, val_positive_df = train_test_split(
            positive_df,
            test_size=model_train_config["data"]["val_size"],
            random_state=config["random_seed"],
        )
        train_negative_df, val_negative_df = train_test_split(
            negative_df,
            test_size=model_train_config["data"]["val_size"],
            random_state=config["random_seed"],
        )

        train_dataset = ArxivDataset(train_positive_df, negative_pairs=train_negative_df, seed=config["random_seed"])
        val_dataset = ArxivDataset(val_positive_df, negative_pairs=val_negative_df, seed=config["random_seed"])
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=lambda x: train_dataset._collate_fn(x, title_process_mode="combined"),
            **model_train_config["dataloaders"],
        )
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            collate_fn=lambda x: val_dataset._collate_fn(x, title_process_mode="combined"),
            **model_train_config["dataloaders"],
        )
    else:
        train_dataset = ArxivDataset(positive_df, negative_pairs=negative_df, seed=config["random_seed"])
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=lambda x: train_dataset._collate_fn(x, title_process_mode="combined"),
            **model_train_config["dataloaders"],
        )
        val_dataloader = None

    model = CustomBert(**config["model"]["bert_model"])
    if load_triplet_pretrained_bert:
        model.bert.load_state_dict(torch.load(PROJECT_ROOT / config["model"]["pretrain"]["save_path"]))
    model.to(device)

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

    if config["model"]["watcher"]:
        wandb.finish()
