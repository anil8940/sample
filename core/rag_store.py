from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from config import settings


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    """Return a Qdrant vector store, creating its collection when needed."""
    embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    if not client.collection_exists(settings.qdrant_collection):
        vector_size = len(embeddings.embed_query("dimension check"))
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )


def ingest_documents(documents: list[Document]) -> int:
    """Split documents into retrieval-friendly chunks and store them in Qdrant."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1_000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    get_vector_store().add_documents(chunks)
    return len(chunks)


def ingest_texts(texts: list[str], source: str) -> int:
    """Store plain-text documents in Qdrant."""
    documents = [Document(page_content=text, metadata={"source": source}) for text in texts]
    return ingest_documents(documents)
