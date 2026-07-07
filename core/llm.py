"""LLM initialization and chain setup."""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter
from config import settings


def initialize_llm():
    """Initialize the OpenRouter LLM."""
    # Set the API key from settings
    if settings.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key
    
    llm = ChatOpenRouter(
        model="openrouter/free",
        temperature=0.7,
    )
    return llm


def create_chain(llm):
    """Create the LLM chain with prompt template."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Provide clear and concise answers. Remember the conversation context."),
        ("user", "{user_input}")
    ])
    
    chain = prompt_template | llm
    return chain


def answer_with_history(llm, history_text: str, user_question: str) -> str:
    """
    Generate a response considering conversation history from LangChain memory.
    
    Args:
        llm: The language model
        history_text: Formatted conversation history from LangChain memory
        user_question: Current user question
    
    Returns:
        The AI response
    """
    # Build messages with history context
    messages = []
    
    # Add system prompt
    messages.append(("system", "You are a helpful AI assistant. Provide clear and concise answers. Remember the conversation context."))
    
    # Add history if available
    if history_text:
        messages.append(("human", history_text))
        messages.append(("ai", "I understand the conversation history above."))
    
    # Add current question
    messages.append(("human", user_question))
    
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm
    
    # Generate response
    response = chain.invoke({})
    return response.content if hasattr(response, 'content') else str(response)


def stream_answer_with_history(llm, history_text: str, user_question: str):
    """
    Stream a response considering conversation history from LangChain memory.
    
    Args:
        llm: The language model
        history_text: Formatted conversation history from LangChain memory
        user_question: Current user question
    
    Yields:
        Response chunks
    """
    # Build messages with history context
    messages = []
    
    # Add system prompt
    messages.append(("system", "You are a helpful AI assistant. Provide clear and concise answers. Remember the conversation context."))
    
    # Add history if available
    if history_text:
        messages.append(("human", history_text))
        messages.append(("ai", "I understand the conversation history above."))
    
    # Add current question
    messages.append(("human", user_question))
    
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm
    
    # Stream response
    for chunk in chain.stream({}):
        if hasattr(chunk, 'content'):
            yield chunk.content
        else:
            yield str(chunk)


# Initialize LLM and chain at module load time
llm = initialize_llm()
chain = create_chain(llm)
