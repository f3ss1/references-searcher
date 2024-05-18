import warnings

import hydra
from omegaconf import DictConfig, OmegaConf
import transformers

from references_searcher.data.sql import DatabaseInterface
from references_searcher.pipelines import bert_pipeline, catboost_pipeline


transformers.logging.set_verbosity_error()
warnings.simplefilter("ignore", FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    config = OmegaConf.to_container(config, resolve=True)
    database_interface = DatabaseInterface()

    if config["model"]["type"] == "bert":
        bert_pipeline(database_interface, config)
    elif config["model"]["type"] == "catboost":
        catboost_pipeline(database_interface, config)
    else:
        raise ValueError("Unsupported model type: should be either `bert` or `catboost`!")


if __name__ == "__main__":
    main()
