import logging
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path as P
from time import perf_counter
from typing import List, Optional

import fire
import pandas as pd
import qdrant_client
import tiktoken
import tqdm
import yaml
from langchain.chains import VectorDBQA
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    ReadTheDocsLoader,
    UnstructuredHTMLLoader,
)
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Qdrant
from pydantic import BaseModel, Field

import index_utils
from configs import (
    EmbeddingConfig,
    LoaderConfig,
    PersistenceConfig,
    PreprocessingConfig,
)

logging.basicConfig(level="INFO")

supported_loader_types = ["rtdocs", "html", "text_files", "pdf"]


def strip_html_whitespaces(html_str):
    return re.sub("\n+", "\n", html_str.page_content)


def load_html_docs(path):
    html_files = list(P(path).rglob("*html"))
    docs = (UnstructuredHTMLLoader(p).load()[0] for p in html_files)
    for doc in tqdm.tqdm(docs):
        doc.page_content = re.sub("\n+", "\n", doc.page_content)
        yield doc


def load_raw_docs(path, loader_type: str = "rtdocs", glob_pattern="**/*"):
    """
    TODO: what is actual return type? Is this list by default or
    """
    assert loader_type in supported_loader_types
    if loader_type == "rtdocs":
        return ReadTheDocsLoader(path).load()
    elif loader_type == "html":
        return load_html_docs(path)
    elif loader_type == "text_files":
        return DirectoryLoader(path, glob=glob_pattern).load()
    elif loader_type == "pdf":
        return PyPDFLoader(path).load()


# langchain_loader = ReadTheDocsLoader("rtdocs/langchain.readthedocs.io/en/latest/")
# llama_loader = ReadTheDocsLoader("rtdocs/gpt-index.readthedocs.io/en/latest")
# llama_path = "rtdocs/gpt-index.readthedocs.io/en/latest"

# docs = llama_loader.load()


def preprocess_docs(
    raw_docs,
    preprocessing_config: PreprocessingConfig = PreprocessingConfig.get_default(),
    min_char_length=10,
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=preprocessing_config.chunk_size,
        chunk_overlap=preprocessing_config.chunk_overlap,
    )
    documents = text_splitter.split_documents(raw_docs)
    return (doc for doc in documents if len(doc.page_content) > min_char_length)


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f"Time: {perf_counter() - start:.3f} seconds")


class DocStoreBuilder(BaseModel):

    preprocessing_config: PreprocessingConfig
    embedding_config: EmbeddingConfig

    def make_doc_store_from_documents(
        self, documents, collection_name, persistence_config
    ):
        assert persistence_config.index_type in ["chroma", "qdrant"]
        logging.info(f"building a docstore using {str(persistence_config)}")
        embeddings = self.embedding_config.load_embeddings()
        if persistence_config.index_type == "chroma":
            return self._make_chroma_doc_store(
                documents, embeddings, collection_name, persistence_config
            )
        elif persistence_config.index_type == "qdrant":
            return self._make_qdrant_doc_store(
                documents, embeddings, collection_name, persistence_config
            )

    def _make_chroma_doc_store(
        self, documents, embeddings, collection_name, persistence_config
    ):
        with catchtime():
            doc_store = Chroma.from_documents(
                self.filter_texts(documents),
                embeddings,
                collection_name=persistence_config.collection_name,
                persist_directory=persistence_config.persist_directory,
            )
        return doc_store

    def _update_qdrant_alias(self, qdrant_path, collection_name):
        qdrant = qdrant_client.QdrantClient(qdrant_path)
        old_collection_name = list(qdrant._client.collections)[0]
        collection_schema = qdrant.describe_collection(collection_name)
        collection_schema["alias"] = collection_name

    def _make_qdrant_doc_store(
        self, documents, embeddings, collection_name, persistence_config
    ):
        with catchtime():
            qdrant_path = (
                P(persistence_config.persist_directory)
                / "qdrant"
                / persistence_config.collection_name
            )
            if qdrant_path.exists():
                logging.info(
                    f"index persistence path exist, removing {str(qdrant_path)}"
                )
                shutil.rmtree(qdrant_path)
            qdrant_path.mkdir(parents=True)
            doc_store = Qdrant.from_documents(
                self.filter_texts(documents),
                embeddings,
                path=qdrant_path,
            )
            self._update_qdrant_alias(collection_name, qdrant_path)
            return doc_store

    def setup_doc_store(
        self, loader_config: LoaderConfig, persistence_config: PersistenceConfig
    ):
        raw_docs = load_raw_docs(
            loader_config.path,
            loader_config.loader_type,
            glob_pattern=loader_config.glob_pattern,
        )
        docs = preprocess_docs(raw_docs, self.preprocessing_config)
        return self.make_doc_store_from_documents(
            docs,
            collection_name=persistence_config.collection_name,
            persistence_config=persistence_config,
        )

    def get_doc_store(
        self, loader_config: LoaderConfig, persistence_config: PersistenceConfig
    ):
        # if self.check_if_exists(loader_config):
        #     return None
        # else:
        return self.setup_doc_store(loader_config, persistence_config)

    # def check_if_exists(self, loader_config: LoaderConfig):
    #     qdrant_client = index_utils.get_default_qdrant_client()
    #     return loader_config.get_index_name() in qdrant_client.get_collections()

    def filter_texts(self, documents):
        return [doc for doc in documents]


class Main:
    @staticmethod
    def index(
        path,
        loader_type,
        glob_pattern=None,
        collection_name=None,
        persist_directory="vectordb",
        persistence_config_path: str = "conf/persistence_config.yaml",
        embedding_config_path: str = "conf/embedding_config.yaml",
        preprocessing_config_path: str = "conf/preprocessing_config.yaml",
    ):
        persistence_config = load_model_from_yaml(
            PersistenceConfig, persistence_config_path
        )
        if collection_name:
            persistence_config.collection_name = collection_name
        embedding_config = load_model_from_yaml(EmbeddingConfig, embedding_config_path)
        preprocessing_config = load_model_from_yaml(
            PreprocessingConfig, preprocessing_config_path
        )
        loader_config = LoaderConfig(
            path=path, loader_type=loader_type, glob_pattern=glob_pattern or "**/*"
        )
        builder = DocStoreBuilder(
            embedding_config=embedding_config,
            preprocessing_config=preprocessing_config,
            persistence_config=persistence_config,
        )
        logging.info(f"loader config{loader_config.get_index_name()}")
        # if builder.check_if_exists(loader_config):
        #     logging.info("skipping building, {loader_config.get_index_name()} exists")
        # else:
        logging.info(f"building doc store from {loader_config.get_index_name()}")
        logging.info(f"using {embedding_config.embedding_model_name} model")
        doc_store = builder.setup_doc_store(loader_config, persistence_config)
        logging.info("built doc store")
        # doc_store.persist()


if __name__ == "__main__":
    fire.Fire(Main())

# for doc in doc_store.similarity_search("What are key concepts of llamaindex?"):
#     print(doc.page_content)
#     print()

# docs[0].page_content
