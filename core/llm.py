"""LLM initialization and chain setup."""

import os
from langchain_core.prompts import ChatPromptTemplate
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
        ("system", "You are a helpful AI assistant. Provide clear and concise answers."),
        ("user", "{user_input}")
    ])
    
    chain = prompt_template | llm
    return chain


# Initialize LLM and chain at module load time
llm = initialize_llm()
chain = create_chain(llm)
