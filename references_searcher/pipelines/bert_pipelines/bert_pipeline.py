from references_searcher.utils import validate_triplet_pretrain, generate_device
from references_searcher.data.sql import DatabaseInterface
from references_searcher.pipelines.bert_pipelines import triplet_pretrain_bert, train_bert


def bert_pipeline(
    database_interface: DatabaseInterface,
    config: dict,
) -> None:
    device = generate_device(config["use_cuda_for_train"])

    if config["model"]["pretrain"]["execute"]:
        triplet_pretrain_bert(database_interface, config, device)

    load_triplet_pretrained_bert = validate_triplet_pretrain(
        config["model"]["train"]["use_triplet_pretrain"],
        config["model"]["pretrain"]["execute"],
        config["model"]["pretrain"]["save_path"],
    )
    train_bert(database_interface, config, device, load_triplet_pretrained_bert)
