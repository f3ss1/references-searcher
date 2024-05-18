from django.apps import AppConfig
import warnings
import transformers
import os

import torch

from references_searcher.constants import CONFIG, PROJECT_ROOT
from references_searcher.utils import generate_device
from references_searcher.data.sql import DatabaseInterface
from references_searcher.models import CustomBert, Inferencer

transformers.logging.set_verbosity_error()
warnings.simplefilter("ignore", FutureWarning)


class ReferenceSearcherInferenceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "reference_searcher_inference"
    inferencer = None
    has_run = False

    def ready(self):
        if os.environ.get("RUN_MAIN") and not self.has_run:
            database_interface = DatabaseInterface()
            metadata_df = database_interface.get_references_metadata()
            device = generate_device(CONFIG["use_cuda_for_inference"])

            model = CustomBert(**CONFIG["model"]["bert_model"])
            model.load_state_dict(torch.load(PROJECT_ROOT / CONFIG["model"]["train"]["save_path"]))
            model.eval()
            model.to(device)

            inferencer = Inferencer(model, batch_size=CONFIG["model"]["inference"]["batch_size"])
            inferencer.fit(metadata_df)
            print(inferencer.allowed_references.head())
            self.__class__.inferencer = inferencer
            self.__class__.has_run = True
