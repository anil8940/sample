"""Conversation history management using LangChain memory."""

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List


class LangChainConversationMemory:
    """Wrapper around LangChain's message history."""
    
    def __init__(self, k: int = 10):
        """Initialize memory with window size k."""
        self.history: List[BaseMessage] = []
        self.k = k
    
    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Save input/output to memory."""
        if "input" in inputs and inputs["input"]:
            self.history.append(HumanMessage(content=inputs["input"]))
        if "output" in outputs and outputs["output"]:
            self.history.append(AIMessage(content=outputs["output"]))
        self._trim_history()
    
    def load_memory_variables(self, inputs: dict) -> dict:
        """Load memory variables."""
        return {"history": self._format_messages()}
    
    def _format_messages(self) -> str:
        """Format messages as a string."""
        if not self.history:
            return ""
        formatted = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                formatted.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")
        return "\n".join(formatted)
    
    def _trim_history(self) -> None:
        """Keep only the last k message pairs."""
        if len(self.history) > self.k * 2:
            self.history = self.history[-(self.k * 2):]
    
    def clear(self) -> None:
        """Clear all history."""
        self.history = []


# Initialize memory with window size of 10 (keeps last 10 message pairs)
memory = LangChainConversationMemory(k=10)


def add_user_message(content: str) -> None:
    """Add a user message to LangChain memory."""
    memory.save_context({"input": content}, {"output": ""})


def add_ai_message(content: str) -> None:
    """Add an AI message to LangChain memory."""
    memory.save_context({"input": ""}, {"output": content})


def get_messages() -> str:
    """Get formatted conversation history."""
    variables = memory.load_memory_variables({})
    return variables.get("history", "")


def clear() -> None:
    """Clear conversation history."""
    memory.clear()
