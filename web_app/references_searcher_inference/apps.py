from django.apps import AppConfig
import warnings
import transformers


transformers.logging.set_verbosity_error()
warnings.simplefilter("ignore", FutureWarning)


class ReferenceSearcherInferenceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "references_searcher_inference"
    inferencer = None
    has_run = False

    def ready(self):
        pass
