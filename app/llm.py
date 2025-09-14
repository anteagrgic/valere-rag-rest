
from time import perf_counter
from typing import List
from .config import settings
from .interfaces import ChatMessage, ChatResult, SupportsChat


class OpenAIChat:
    def __init__(self):
        from langchain_openai import ChatOpenAI

        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.2,
            timeout=60,
            max_retries=1,
        )

    def invoke(self, messages: List[ChatMessage]) -> ChatResult:
        from langchain.schema import HumanMessage, SystemMessage, AIMessage

        
        lmsgs = []
        for m in messages:
            if m["role"] == "system":
                lmsgs.append(SystemMessage(content=m["content"]))

            elif m["role"] == "assistant":
                lmsgs.append(AIMessage(content=m["content"]))
            else:
                lmsgs.append(HumanMessage(content=m["content"]))
        t0 = perf_counter()
        out = self.llm.invoke(lmsgs)
        dt = int((perf_counter() - t0) * 1000)
        meta = getattr(out, "response_metadata", {})
        usage = meta.get("token_usage", {}) or {}
        return ChatResult(
            content=out.content,
            model=meta.get("model_name", settings.openai_model),
            provider="openai",
            finish_reason=meta.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=dt,
        )

    def provider(self) -> str: return "openai"
    def model(self) -> str | None: return settings.openai_model


class OllamaChat:
    def __init__(self):
        from langchain_community.chat_models import ChatOllama
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.2,

            timeout=60,
        )

    def invoke(self, messages: List[ChatMessage]) -> ChatResult:
        # ChatOllama može i dict poruke, ali radi konzistentnosti koristimo LC message objekte
        from langchain.schema import HumanMessage, SystemMessage, AIMessage
        lmsgs = []
        for m in messages:
            if m["role"] == "system":
                lmsgs.append(SystemMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lmsgs.append(AIMessage(content=m["content"]))
            else:
                lmsgs.append(HumanMessage(content=m["content"]))
        t0 = perf_counter()
        out = self.llm.invoke(lmsgs)
        dt = int((perf_counter() - t0) * 1000)
        text = getattr(out, "content", str(out))
        return ChatResult(
            content=text,
            model=settings.ollama_model,
            provider="ollama",
            latency_ms=dt,
        )

    def provider(self) -> str: return "ollama"
    def model(self) -> str | None: return settings.ollama_model

class MockChat:
    def invoke(self, messages: List[ChatMessage]) -> ChatResult:
        user = [m for m in messages if m["role"] != "system"][-1]["content"]
        return ChatResult(
            content=f"(Mock answer) Based on the provided context, here is a concise response: {user[:180]}",
            model="mock",
            provider="mock",
            finish_reason="stop",
            prompt_tokens=0,
            completion_tokens=len(user[:180].split()),
            latency_ms=1,
        )
    def provider(self) -> str: return "mock"
    def model(self) -> str | None: return "mock"

def make_chat() -> SupportsChat:
    prov = (settings.llm_provider or "mock").lower()    
    if prov == "openai" and settings.openai_api_key:
        return OpenAIChat()
    if prov == "ollama":
        return OllamaChat()
    return MockChat()
