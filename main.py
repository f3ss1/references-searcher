import warnings

import wandb
import transformers

from references_searcher.constants import CONFIG
from references_searcher.data.sql import DatabaseInterface
from references_searcher.pipelines import BertPipeline, CatboostPipeline
from references_searcher.utils import seed_everything


transformers.logging.set_verbosity_error()
warnings.simplefilter("ignore", FutureWarning)


def main() -> None:
    seed_everything(CONFIG["random_seed"])
    database_interface = DatabaseInterface()

    # WandB handling
    if CONFIG["model"]["use_watcher"]:
        wandb.init(
            project="references-searcher",
            config=CONFIG,
        )
        watcher = "wandb"
    else:
        watcher = None

    if CONFIG["model"]["type"] == "bert":
        pipeline = BertPipeline(database_interface)

    elif CONFIG["model"]["type"] == "catboost":
        pipeline = CatboostPipeline(database_interface)
    else:
        raise ValueError("Unsupported model type: should be either `bert` or `catboost`!")

    pipeline.run(CONFIG, watcher)

    if CONFIG["model"]["use_watcher"]:
        wandb.finish()


if __name__ == "__main__":
    main()
