import qdrant_client
from langchain.vectorstores import Qdrant
from typing import Optional


def get_default_qdrant_client():
    url: Optional[str] = None
    port: Optional[int] = 6333
    grpc_port: int = 6334
    return qdrant_client.QdrantClient(
        host="localhost", url=url, port=port, grpc_port=grpc_port
    )
