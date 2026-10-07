"""ragstat: deterministic retrieval evaluation for RAG systems."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ragstat")
except PackageNotFoundError:
    __version__ = "0.0.0"
