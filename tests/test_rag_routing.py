from core.rag import should_use_rag


def test_document_questions_route_to_rag() -> None:
    assert should_use_rag("What does the uploaded PDF say about pricing?") is True
    assert should_use_rag("Summarize the document content for me") is True
    assert should_use_rag("According to the source file, what is the refund policy?") is True


def test_general_questions_do_not_route_to_rag() -> None:
    assert should_use_rag("What is the capital of France?") is False
    assert should_use_rag("Write a short Python function") is False
    assert should_use_rag("How does photosynthesis work?") is False
