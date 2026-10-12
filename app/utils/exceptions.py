class RagBaseError(Exception):
    """Base for all RAG pipeline errors."""
    pass


class UnsupportedFileTypeError(RagBaseError):
    """Raised when an unsupported file extension is uploaded."""
    pass


class DocumentParseError(RagBaseError):
    """Raised when Docling fails to parse a document."""
    pass


class ChunkingError(RagBaseError):
    """Raised when semantic chunking fails."""
    pass


class EmbeddingError(RagBaseError):
    """Raised when the embedding model fails."""
    pass


class VectorStoreError(RagBaseError):
    """Raised when Qdrant operations fail."""
    pass


class CollectionNotFoundError(VectorStoreError):
    """Raised when the target Qdrant collection does not exist."""
    pass


class QueryReformulationError(RagBaseError):
    """Raised when LLM query reformulation fails."""
    pass


class RetrievalError(RagBaseError):
    """Raised when retrieval fails."""
    pass


class LLMProviderError(RagBaseError):
    """Raised when an LLM provider call fails."""
    pass


class AllProvidersFailedError(LLMProviderError):
    """Raised when every configured LLM provider fails."""
    pass


class InformationUnavailableError(RagBaseError):
    """
    Raised (or caught) when retrieved context does not contain
    enough information to answer the question.
    """
    pass
