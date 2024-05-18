from references_searcher.data import ArxivDataset
from references_searcher.models import CustomBert, Trainer

from sqlalchemy import create_engine, text
import pandas as pd

from sklearn.model_selection import train_test_split

from references_searcher.constants import POSTGRES_URL, PROJECT_ROOT
from references_searcher.utils import seed_everything, generate_device

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import transformers
import warnings
import wandb

transformers.logging.set_verbosity_error()
warnings.simplefilter("ignore", FutureWarning)

MAIN_SEED = 42
DEVICE = generate_device()
seed_everything(MAIN_SEED)
wandb.init(
    project="references-searcher",
)

engine = create_engine(POSTGRES_URL)
positive_query = """
SELECT
    paper_arxiv_id,
    paper_title,
    paper_abstract,
    reference_title,
    reference_abstract,
    reference_citation_count,
    reference_influential_citation_count
FROM joined_filtered_positive_references
"""
negative_query = """
SELECT
    paper_arxiv_id,
    paper_title,
    paper_abstract,
    reference_title,
    reference_abstract,
    reference_citation_count,
    reference_influential_citation_count
FROM joined_filtered_negative_references
"""

with engine.connect() as conn:
    positive_df = pd.read_sql_query(text(positive_query), conn)
    negative_df = pd.read_sql_query(text(negative_query), conn)

positive_df = positive_df.iloc[:50000]
negative_df = negative_df.iloc[:50000]

train_positive_df, val_positive_df = train_test_split(
    positive_df,
    test_size=0.2,
    random_state=MAIN_SEED,
)
train_negative_df, val_negative_df = train_test_split(
    negative_df,
    test_size=0.2,
    random_state=MAIN_SEED,
)


train_dataset = ArxivDataset(train_positive_df, negative_pairs=train_negative_df, seed=MAIN_SEED)
val_dataset = ArxivDataset(val_positive_df, negative_pairs=val_negative_df, seed=MAIN_SEED)
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    collate_fn=lambda x: train_dataset._collate_fn(x, title_process_mode="combined"),
)
val_dataloader = DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    collate_fn=lambda x: val_dataset._collate_fn(x, title_process_mode="combined"),
)

model = CustomBert(concatenate_title=False)
model.bert.load_state_dict(torch.load(PROJECT_ROOT / "model_weights/pretrained_bert.pth"))
model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=0.00001)

trainer = Trainer(watcher="wandb", device=DEVICE)

trainer.train(model, optimizer, train_dataloader, val_dataloader=val_dataloader, n_epochs=4)
torch.save(model.state_dict(), PROJECT_ROOT / "model_weights/pretrained_model.pth")
wandb.finish()
