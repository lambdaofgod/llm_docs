import index_utils
from pathlib import Path as P
from itertools import islice
from llama_index import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant, Chroma
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredHTMLLoader
import re
import tqdm

import logging
import fire
from pydantic import BaseModel, Field
import pandas as pd
import tiktoken
from dataclasses import dataclass
from typing import List, Optional
import os
from langchain.document_loaders import ReadTheDocsLoader, DirectoryLoader


logging.basicConfig(level="INFO")

supported_loader_types = ["rtdocs", "html", "text_files"]

from pydantic import BaseModel, Field


import yaml
from pydantic import BaseModel, Field


def load_model_from_yaml(base_model_cls, yaml_path):
    with open(yaml_path) as f:
        raw_obj = yaml.safe_load(f)
    return base_model_cls.parse_obj(raw_obj)


class LoaderConfig(BaseModel):

    path: str
    loader_type: str
    glob_pattern: str = Field(default="**/*")

    def get_index_name(self):
        return f"{P(self.path).name}-{self.loader_type}"


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


# langchain_loader = ReadTheDocsLoader("rtdocs/langchain.readthedocs.io/en/latest/")
# llama_loader = ReadTheDocsLoader("rtdocs/gpt-index.readthedocs.io/en/latest")
# llama_path = "rtdocs/gpt-index.readthedocs.io/en/latest"

# docs = llama_loader.load()


class EmbeddingConfig(BaseModel):

    embedding_model_name: str
    timeout: int

    @staticmethod
    def get_default():
        return EmbeddingConfig(
            embedding_model_name="sentence-transformers/all-mpnet-base-v2", timeout=180
        )

    def load_embeddings(self):
        return HuggingFaceEmbeddings(model_name=self.embedding_model_name)


class PreprocessingConfig(BaseModel):

    chunk_size: int
    chunk_overlap: int

    @staticmethod
    def get_default():
        return PreprocessingConfig(chunk_size=512, chunk_overlap=128)


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


class PersistenceConfig(BaseModel):
    persist_directory: str
    collection_name: Optional[str]


from time import perf_counter
from contextlib import contextmanager


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
        embeddings = self.embedding_config.load_embeddings()
        with catchtime():
            doc_store = Chroma.from_documents(
                self.filter_texts(documents),
                embeddings,
                collection_name=persistence_config.collection_name,
                persist_directory=persistence_config.persist_directory,
            )
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
    def index_with_chroma(
        path,
        loader_type,
        glob_pattern=None,
        collection_name=None,
        persist_directory="vectordb",
        embedding_config_path: str = "embedding_config.yaml",
        preprocessing_config_path: str = "preprocessing_config.yaml",
    ):
        embedding_config = load_model_from_yaml(EmbeddingConfig, embedding_config_path)
        preprocessing_config = load_model_from_yaml(
            PreprocessingConfig, preprocessing_config_path
        )
        loader_config = LoaderConfig(
            path=path, loader_type=loader_type, glob_pattern=glob_pattern or "**/*"
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
        # if builder.check_if_exists(loader_config):
        #     logging.info("skipping building, {loader_config.get_index_name()} exists")
        # else:
        logging.info(f"building doc store from {loader_config.get_index_name()}")
        logging.info(f"using {embedding_config.embedding_model_name} model")
        doc_store = builder.setup_doc_store(loader_config, persistence_settings)
        logging.info("built doc store")
        doc_store.persist()


if __name__ == "__main__":
    fire.Fire(Main())

# for doc in doc_store.similarity_search("What are key concepts of llamaindex?"):
#     print(doc.page_content)
#     print()

# docs[0].page_content
