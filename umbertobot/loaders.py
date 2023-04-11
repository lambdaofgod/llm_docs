from langchain.document_loaders.base import BaseLoader
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class PandasLoader(BaseLoader):
    """Loader that loads ReadTheDocs documentation directory dump."""

    def __init__(
        self,
        path: str,
        text_col: str,
        loader_type: str,
        errors: Optional[str] = None,
        **kwargs: Optional[Any]
    ):
        self.file_path = path
        self.errors = errors
        self.text_col = text_col
        self.load_file = pd.read_csv if loader_type == "csv" else pd.read_parquet

    def load(self) -> List[Document]:
        """Load documents."""

        df = self.load_file(self.file_path)
        docs = []
        for __, row in df.iterrows():
            text = row[self.text_col]
            metadata = row.to_dict()
            metadata.pop(self.text_col)
            docs.append(Document(page_content=text, metadata=metadata))
        return docs
