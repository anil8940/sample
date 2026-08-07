from __future__ import annotations

import re
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from config import settings
from core.llm import chain, llm
from core.rag_store import get_vector_store
from constants import DOC_RELEVANCE_TERMS


class RAGState(MessagesState):
    documents: list
    answer: str


def retrieve(state: RAGState) -> dict:
    question = state["messages"][-1].content
    documents = get_vector_store().similarity_search(str(question), k=settings.retrieval_k)
    return {"documents": documents}


def answer(state: RAGState) -> dict:
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


def answer_with_general_llm(question: str) -> str:
    response = chain.invoke({"user_input": question})
    return response.content if hasattr(response, "content") else str(response)


def direct_answer(state: RAGState) -> dict:
    question = state["messages"][-1].content
    answer_text = answer_with_general_llm(question)
    return {"answer": answer_text}


def should_use_rag(question: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    if not normalized:
        return False
    if any(term in normalized for term in DOC_RELEVANCE_TERMS):
        return True
    return any(
        phrase in normalized
        for phrase in (
            "according to",
            "based on",
            "from the",
            "in the document",
            "in the uploaded",
            "summarize the document",
            "summarize this document",
            "what does the",
            "what does this",
            "tell me about",
        )
    )


def route_question(state: RAGState) -> str:
    question = state["messages"][-1].content
    return "retrieve" if should_use_rag(question) else "direct_answer"


builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)
builder.add_node("direct_answer", direct_answer)
builder.add_conditional_edges(
    START,
    route_question,
    {"retrieve": "retrieve", "direct_answer": "direct_answer"},
)
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)
builder.add_edge("direct_answer", END)
memory = InMemorySaver()
rag_graph = builder.compile(checkpointer=memory)


def ask_rag(question: str, thread_id: str) -> tuple[str, list[dict[str, str]]]:
    result = rag_graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    documents = result.get("documents", [])
    if not documents:
        return result.get("answer", answer_with_general_llm(question)), []

    sources = []
    seen = set()
    for document in documents:
        source = document.metadata.get("source", "unknown")
        if source not in seen:
            sources.append({"source": source})
            seen.add(source)
    return result["answer"], sources


def clear_rag_history(thread_id: str) -> None:
    memory.delete_thread(thread_id)
