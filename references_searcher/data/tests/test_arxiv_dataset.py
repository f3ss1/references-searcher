import pandas as pd
from transformers import BertTokenizer

from references_searcher.data import ArxivDataset

sample_positive_dataframe = pd.DataFrame(
    {
        "paper_title": ["test title"],
        "paper_abstract": ["this is a test abstract"],
        "reference_title": ["test reference title"],
        "reference_abstract": ["this is a test reference abstract"],
        "reference_citation_count": [12],
        "reference_influential_citation_count": [2],
    },
)
sample_negative_dataframe = pd.DataFrame(
    {
        "paper_title": ["test title"],
        "paper_abstract": ["this is a test abstract"],
        "reference_title": ["bad test reference title"],
        "reference_abstract": ["bad this is a test reference abstract"],
        "reference_citation_count": [123],
        "reference_influential_citation_count": [7],
    },
)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=512)


def test_dataset_creation():
    ArxivDataset(
        sample_positive_dataframe,
        mode="classification",
        negative_pairs=sample_negative_dataframe,
        tokenizer=tokenizer,
    )


def test_get_item():
    dataset = ArxivDataset(
        sample_positive_dataframe,
        mode="classification",
        negative_pairs=sample_negative_dataframe,
        tokenizer=tokenizer,
    )
    dataset_output = dataset[0]
    assert dataset_output[0] == sample_positive_dataframe.iloc[0].to_dict()
    assert dataset_output[1]


def test_negative_jump():
    dataset = ArxivDataset(
        sample_positive_dataframe,
        mode="classification",
        negative_pairs=sample_negative_dataframe,
        tokenizer=tokenizer,
    )
    dataset_output = dataset[1]
    assert dataset_output[0] == sample_negative_dataframe.iloc[0].to_dict()
    assert not dataset_output[1]


def test_dataset_creation_no_target():
    ArxivDataset(
        sample_positive_dataframe,
        mode="classification",
        negative_pairs=sample_negative_dataframe,
        tokenizer=tokenizer,
        return_target=False,
    )


def test_get_item_no_target():
    dataset = ArxivDataset(
        sample_positive_dataframe,
        mode="classification",
        negative_pairs=sample_negative_dataframe,
        return_target=False,
    )
    assert dataset[0] == sample_positive_dataframe.iloc[0].to_dict()


def test_collate_fn():
    dataset = ArxivDataset(
        sample_positive_dataframe,
        mode="classification",
        negative_pairs=sample_negative_dataframe,
        return_target=False,
    )
    print(dataset._collate_fn([dataset[0], dataset[1]]))
