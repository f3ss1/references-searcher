import wandb

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from references_searcher.data import ArxivDataset
from references_searcher.data.sql import DatabaseInterface
from references_searcher.models import CustomBert, Trainer
from references_searcher.constants import PROJECT_ROOT


def triplet_pretrain_bert(
    database_interface: DatabaseInterface,
    config: dict,
    device: torch.device,
):
    model_pretrain_config = config["model"]["pretrain"]

    if config["model"]["watcher"]:
        wandb.init(
            project="references-searcher",
            config=config,
        )
        watcher = "wandb"
    else:
        watcher = None

    positive_df = database_interface.get_positive_references(model_pretrain_config["data"]["cutoff"])
    metadata_df = database_interface.get_references_metadata()

    pretrain_dataset = ArxivDataset(
        positive_df,
        mode="triplet",
        references_metadata=metadata_df,
        seed=config["random_seed"],
    )

    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        shuffle=True,
        collate_fn=lambda x: pretrain_dataset._collate_fn(x, title_process_mode="combined"),
        **model_pretrain_config["dataloader"],
    )

    model = CustomBert(**config["model"]["bert_model"])
    model.to(device)
    optimizer = AdamW(model.parameters(), **model_pretrain_config["optimizer"])

    trainer = Trainer(watcher=watcher, device=device)
    trainer.pretrain(model, optimizer, pretrain_dataloader, n_epochs=model_pretrain_config["n_epochs"])

    torch.save(model.bert.state_dict(), PROJECT_ROOT / model_pretrain_config["save_path"])

    if config["model"]["watcher"]:
        wandb.finish()
