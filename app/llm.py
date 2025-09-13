from typing import Protocol
from .config import settings

# Providers
class SupportsChat(Protocol):
    def invoke(self, messages: list[dict]) -> str: ...

class OpenAIChat:
    def __init__(self):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model=settings.openai_model, temperature=0.2)

    def invoke(self, messages: list[dict]) -> str:
        from langchain.schema import HumanMessage, SystemMessage
        lmsgs = []
        for m in messages:
            if m["role"] == "system":
                lmsgs.append(SystemMessage(content=m["content"]))
            else:
                lmsgs.append(HumanMessage(content=m["content"]))
        out = self.llm.invoke(lmsgs)
        return out.content

class OllamaChat:
    def __init__(self):
        from langchain_community.chat_models import ChatOllama
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.2,
        )

    def invoke(self, messages: list[dict]) -> str:
        text = "\n".join([m["content"] for m in messages])
        out = self.llm.invoke(text)
        return getattr(out, "content", str(out))

class MockChat:
    def invoke(self, messages: list[dict]) -> str:
        # Very simple fallback: just echo last user message and say "based on provided context"
        user = [m for m in messages if m["role"] != "system"][-1]["content"]
        return f"(Mock answer) Based on the provided context, here is a concise response to your query: {user[:180]}"

def make_chat() -> SupportsChat:
    prov = settings.llm_provider.lower()
    if prov == "openai" and settings.openai_api_key:
        return OpenAIChat()
    if prov == "ollama":
        return OllamaChat()
    return MockChat()
