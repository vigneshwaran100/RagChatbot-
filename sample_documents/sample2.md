# AI Knowledge Base — Technical Concepts

## What is a Vector Database?

A vector database is a type of database optimized for storing, indexing, and querying high-dimensional vector embeddings. Unlike traditional relational databases that store structured rows and columns, vector databases store dense numerical vectors and support similarity search operations such as nearest-neighbor search.

Popular vector databases include Qdrant, Pinecone, Weaviate, Chroma, and Milvus. Qdrant is an open-source, high-performance vector database written in Rust that supports filtering, payload storage, and multiple distance metrics including cosine, Euclidean, and dot product.

## RAG Pipeline Components

A complete RAG pipeline consists of the following components:

1. **Document Ingestion**: Documents (PDF, TXT, Markdown) are loaded and converted to plain text.
2. **Chunking**: Documents are split into smaller, semantically meaningful chunks.
3. **Embedding**: Each chunk is converted to a vector embedding using a language model.
4. **Vector Storage**: Chunks and their embeddings are stored in a vector database.
5. **Query Processing**: User questions are embedded and used to search for similar chunks.
6. **Answer Generation**: Retrieved chunks are passed as context to an LLM, which generates a grounded answer.

## Comparing RAG vs Fine-tuning

RAG and fine-tuning are two complementary approaches to building knowledge-aware AI systems:

**RAG advantages:**
- Knowledge base can be updated without retraining
- Answers cite sources, improving trust and traceability
- Works well for domain-specific or rapidly changing information
- Lower computational cost compared to full model training

**Fine-tuning advantages:**
- Model learns implicit patterns and styles from domain data
- Faster inference (no retrieval step)
- Better for tasks requiring learned behavior, not factual recall

In production systems, RAG is generally preferred for question-answering over documents because it maintains a clear separation between the model's reasoning capability and the factual knowledge base.

## Limitations of RAG Systems

RAG systems have several known limitations:

- **Retrieval failures**: If the retrieval step returns irrelevant chunks, the LLM will produce poor answers or hallucinate.
- **Chunk boundary issues**: Splitting documents at wrong boundaries can lose important context.
- **Long-context limitations**: LLMs have context window limits, restricting how many chunks can be passed at once.
- **Embedding model quality**: The quality of the embedding model directly affects retrieval accuracy.
- **Latency**: RAG adds retrieval latency on top of LLM inference time.

## Semantic Chunking

Semantic chunking is an advanced document splitting technique that uses embedding similarity between adjacent sentences to determine chunk boundaries. Unlike fixed-size chunking (splitting every N tokens), semantic chunking preserves the natural topic flow of a document. When the semantic similarity between adjacent sentences drops below a threshold, a new chunk is started.
