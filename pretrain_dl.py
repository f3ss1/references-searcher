import torch

from citations_searcher.data import ArxivDataset
from citations_searcher.models import CustomBert, Trainer

from sqlalchemy import create_engine, text
import pandas as pd
import wandb

from citations_searcher.constants import POSTGRES_URL, PROJECT_ROOT
from citations_searcher.utils import seed_everything, generate_device

from torch.utils.data import DataLoader
from torch.optim import AdamW
import transformers
import warnings

transformers.logging.set_verbosity_error()
warnings.simplefilter("ignore", FutureWarning)

MAIN_SEED = 42
DEVICE = generate_device()
watcher = "wandb"
seed_everything(MAIN_SEED)

if watcher is not None:
    wandb.init(
        project="references-searcher",
    )

# Create an engine
engine = create_engine(POSTGRES_URL)
positive_query = """
SELECT
    paper_arxiv_id,
    paper_title,
    paper_abstract,
    reference_arxiv_id,
    reference_title,
    reference_abstract,
    reference_citation_count,
    reference_influential_citation_count
FROM joined_filtered_positive_references
"""
metadata_query = """
SELECT
    arxiv_id,
    title,
    abstract
FROM train_references_metadata
"""

with engine.connect() as conn:
    positive_df = pd.read_sql_query(text(positive_query), conn)
    metadata_df = pd.read_sql_query(text(metadata_query), conn)

positive_df = positive_df  # .iloc[:500]
metadata_df.index = metadata_df["arxiv_id"]


pretrain_dataset = ArxivDataset(positive_df, mode="triplet", references_metadata=metadata_df, seed=MAIN_SEED)

pretrain_dataloader = DataLoader(
    pretrain_dataset,
    shuffle=True,
    batch_size=6,
    num_workers=4,
    pin_memory=True,
    collate_fn=lambda x: pretrain_dataset._collate_fn(x, title_process_mode="combined"),
)

model = CustomBert(concatenate_title=False)
model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=0.00001)

trainer = Trainer(watcher=watcher, device=DEVICE)

trainer.pretrain(model, optimizer, pretrain_dataloader, n_epochs=4)

# TODO: Save only bert core model
torch.save(model.state_dict(), PROJECT_ROOT / "model_weights/pretrained_model.pth")

if watcher is not None:
    wandb.finish()
