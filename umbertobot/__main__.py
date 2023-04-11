from umbertobot import index_utils
import re
import tqdm

from typing import Union, Optional
from umbertobot.indexing import (
    load_or_get_default,
    load_raw_docs,
    PreprocessingConfig,
    LoaderConfig,
    DocStoreBuilder,
    PersistenceConfig,
    EmbeddingConfig,
)
import logging
import fire


class Main:
    @staticmethod
    def index(
        path,
        loader_type,
        glob_pattern=None,
        collection_name=None,
        persist_directory="vectordb",
        text_col=None,
        embedding_config_path: Optional[str] = None,
        preprocessing_config_path: Optional[str] = None,
    ):
        embedding_config = load_or_get_default(EmbeddingConfig, embedding_config_path)
        preprocessing_config = load_or_get_default(
            PreprocessingConfig, preprocessing_config_path
        )
        loader_config = LoaderConfig(
            path=path,
            loader_type=loader_type,
            glob_pattern=glob_pattern or "**/*",
            text_col=text_col,
        )
        persistence_settings = PersistenceConfig(
            collection_name=collection_name or loader_config.get_index_name(),
            persist_directory=persist_directory,
        )
        builder = DocStoreBuilder(
            embedding_config=embedding_config,
            preprocessing_config=preprocessing_config,
            persistence_config=persistence_settings,
        )
        logging.info(f"loader config{loader_config.get_index_name()}")
        logging.info(f"building doc store from {loader_config.get_index_name()}")
        logging.info(f"doc store collection: {persistence_settings.collection_name}")
        logging.info(f"using {embedding_config.embedding_model_name} model")
        doc_store = builder.setup_doc_store(loader_config, persistence_settings)
        logging.info("built doc store")
        doc_store.persist()


if __name__ == "__main__":
    fire.Fire(Main())
