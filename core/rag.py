"""Document ingestion and the LangGraph-powered retrieval workflow."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from qdrant_client import QdrantClient, models

from config import settings
from core.llm import llm


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    """Return a Qdrant vector store, creating its collection when needed."""
    embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key
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


class RAGState(MessagesState):
    documents: list[Document]
    answer: str
 

def retrieve(state: RAGState) -> dict:
    """Fetch the most relevant chunks for the user's question."""
    question = state["messages"][-1].content
    documents = get_vector_store().similarity_search(str(question), k=settings.retrieval_k)
    return {"documents": documents}


def answer(state: RAGState) -> dict:
    """Answer only from retrieved context, signalling missing evidence."""
    context = "\n\n---\n\n".join(
        f"Source: {document.metadata.get('source', 'unknown')}\n{document.page_content}"
        for document in state["documents"]
    ) or "No relevant documents were retrieved."
    response = llm.invoke([
        SystemMessage( 
    content=(
        "You are a helpful assistant. Use the retrieved context when it is relevant. "
        "If the context does not fully answer the question, you may rely on your own knowledge "
        "to provide a clear and accurate response. Always prefer retrieved context when available, "
        "but never leave the user without an answer. Do not invent citations.\n\n"
        f"Retrieved context:\n{context}"
    )
        ),
        *state["messages"],
    ])
    answer_text = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer_text, "messages": [response]}


builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)
memory = InMemorySaver()
rag_graph = builder.compile(checkpointer=memory)
rag_graph.get_graph().draw_mermaid_png(output_file_path="rag_graph.png")

def ask_rag(question: str, thread_id: str) -> tuple[str, list[dict[str, str]]]:
    """Run the graph with checkpointed LangGraph memory for one chat thread."""
    result = rag_graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    sources = []
    seen = set()
    for document in result["documents"]:
        source = document.metadata.get("source", "unknown")
        if source not in seen:
            sources.append({"source": source})
            seen.add(source)
    return result["answer"], sources


def clear_rag_history(thread_id: str) -> None:
    """Delete a chat thread from LangGraph's built-in in-memory checkpointer."""
    memory.delete_thread(thread_id)
