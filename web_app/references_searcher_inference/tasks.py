import os

import torch
import pandas as pd

from celery import shared_task

from references_searcher.models.inferencer import ReferencePrediction

from references_searcher.constants import CONFIG, PROJECT_ROOT
from references_searcher.utils import generate_device
from references_searcher.data.sql import DatabaseInterface
from references_searcher.models import CustomBert, CustomCatboostClassifier, Inferencer
from references_searcher import logger


CELERY_WORKER = os.getenv("CELERY_WORKER", False)
if CELERY_WORKER:
    database_interface = DatabaseInterface()
    metadata_df = database_interface.get_references_metadata()

    if CONFIG["model"]["type"] == "bert":
        device = generate_device(CONFIG["inference"]["use_cuda_for_inference"])

        model = CustomBert(**CONFIG["model"]["bert_model"])
        model.load_state_dict(
            torch.load(PROJECT_ROOT / CONFIG["model"]["train"]["save_path"], map_location=device),
        )
        model.eval()
        model.to(device)

        inferencer = Inferencer(
            model,
            batch_size=CONFIG["model"]["inference"]["batch_size"],
            n_predictions=CONFIG["inference"]["n_predictions"],
            n_candidates=CONFIG["inference"]["n_candidates"],
            device=device,
        )

    else:
        task_type = "GPU" if CONFIG["inference"]["use_cuda_for_inference"] else "CPU"

        model = CustomCatboostClassifier(task_type=task_type)
        model.load_model(PROJECT_ROOT / CONFIG["model"]["save_prefix"])
        inferencer = Inferencer(
            model,
            batch_size=1,
            n_predictions=CONFIG["inference"]["n_predictions"],
            n_candidates=CONFIG["inference"]["n_candidates"],
        )

    inferencer.fit(
        metadata_df,
        prefer_saved_matrix=CONFIG["inference"]["prefer_saved_matrix"],
        references_embeddings_save_path=CONFIG["inference"]["references_embeddings_save_path"],
    )
    logger.info("Finished processing inferencer, ready to make predictions.")


@shared_task
def make_predictions(predictions_input: dict) -> list[ReferencePrediction]:
    logger.info("Celery is trying to make predictions!")
    model_input = pd.DataFrame(predictions_input)
    predictions = inferencer.predict(model_input)[0]
    return [x.to_dict() for x in predictions]
