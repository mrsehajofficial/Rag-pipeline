"""ragpipe -- a production-shaped, zero-dependency RAG pipeline.

Quick start:
    from ragpipe import RAGPipeline
    pipe = RAGPipeline()
    pipe.index_path("./docs")
    print(pipe.query("how do I rotate credentials?").answer)
"""

from .config import Settings, load_settings
from .pipeline import Citation, RAGPipeline, RAGResponse

__version__ = "0.1.0"

__all__ = [
    "Citation",
    "RAGPipeline",
    "RAGResponse",
    "Settings",
    "load_settings",
    "__version__",
]
