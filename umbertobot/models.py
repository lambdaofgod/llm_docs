from pydantic import BaseModel, Field
import yaml
from langchain.embeddings import HuggingFaceEmbeddings
from typing import Optional, List
from pathlib import Path as P
from enum import Enum


class EmbeddingConfig(BaseModel):

    embedding_model_name: str
    timeout: int = Field(default=60)

    @staticmethod
    def get_default():
        return EmbeddingConfig(
            embedding_model_name="sentence-transformers/all-mpnet-base-v2", timeout=180
        )

    def load_embeddings(self):
        return HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    @staticmethod
    def load_from_yaml(yaml_path):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return EmbeddingConfig(**config)


class PreprocessingConfig(BaseModel):

    chunk_size: int
    chunk_overlap: int

    @staticmethod
    def get_default():
        return PreprocessingConfig(chunk_size=512, chunk_overlap=128)


class IndexType(str, Enum):
    chroma = "chroma"


class PersistenceConfig(BaseModel):
    index_type: IndexType
    persist_directory: str
    collection_name: Optional[str]

    class Config:
        use_enum_values = True


class LoaderConfig(BaseModel):

    path: str
    loader_type: str
    glob_pattern: str = Field(default="**/*")
    text_col: Optional[str] = Field(default=None)
    included_cols: Optional[List[str]] = Field(default_factory=list)
    gitignore_path: Optional[str] = Field(default=None)

    @property
    def index_name(self):
        pathname = P(self.path)
        clean_pathname = pathname.name.replace(pathname.suffix, "")
        return f"{clean_pathname}-{self.loader_type}".lower()
