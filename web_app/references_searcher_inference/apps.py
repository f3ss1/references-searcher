from django.apps import AppConfig
import warnings
import transformers
import os

import torch

from references_searcher.constants import CONFIG, PROJECT_ROOT
from references_searcher.utils import generate_device
from references_searcher.data.sql import DatabaseInterface
from references_searcher.models import CustomBert, CustomCatboostClassifier, Inferencer

transformers.logging.set_verbosity_error()
warnings.simplefilter("ignore", FutureWarning)


class ReferenceSearcherInferenceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "references_searcher_inference"
    inferencer = None
    has_run = False

    def ready(self):
        if os.environ.get("RUN_MAIN") and not self.has_run:
            database_interface = DatabaseInterface()
            metadata_df = database_interface.get_references_metadata()

            if CONFIG["model"]["type"] == "bert":
                device = generate_device(CONFIG["inference"]["use_cuda_for_inference"])

                model = CustomBert(**CONFIG["model"]["bert_model"])
                model.load_state_dict(torch.load(PROJECT_ROOT / CONFIG["model"]["train"]["save_path"]))
                model.eval()
                model.to(device)

                inferencer = Inferencer(
                    model,
                    batch_size=CONFIG["model"]["inference"]["batch_size"],
                    n_predictions=CONFIG["inference"]["n_predictions"],
                    n_candidates=CONFIG["inference"]["n_candidates"],
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
            self.__class__.inferencer = inferencer
            self.__class__.has_run = True
