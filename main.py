import warnings

import transformers

from references_searcher.constants import CONFIG
from references_searcher.data.sql import DatabaseInterface
from references_searcher.pipelines import bert_pipeline, catboost_pipeline


transformers.logging.set_verbosity_error()
warnings.simplefilter("ignore", FutureWarning)


def main() -> None:
    database_interface = DatabaseInterface()

    if CONFIG["model"]["type"] == "bert":
        bert_pipeline(database_interface, CONFIG)
    elif CONFIG["model"]["type"] == "catboost":
        catboost_pipeline(database_interface, CONFIG)
    else:
        raise ValueError("Unsupported model type: should be either `bert` or `catboost`!")


if __name__ == "__main__":
    main()
