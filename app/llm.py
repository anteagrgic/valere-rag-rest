from time import perf_counter
from typing import List
from .config import settings
from .interfaces import ChatMessage, ChatResult, SupportsChat
from .errors import ProviderError  # ako već postoji u tvojem projektu


class OpenAIChat:
    def __init__(self):
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ProviderError("OPENAI_API_KEY is not set")

        self.llm = ChatOpenAI(
            model=settings.llm_model,  # alias iz config.py
            api_key=settings.openai_api_key,
            temperature=0.2,
            timeout=settings.llm_timeout,  
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
        try:
            out = self.llm.invoke(lmsgs)
        except Exception as e:
            raise ProviderError(str(e))
        dt = int((perf_counter() - t0) * 1000)

        meta = getattr(out, "response_metadata", {}) or {}
        usage = meta.get("token_usage", {}) or {}

        return ChatResult(
            content=out.content,
            model=meta.get("model_name", settings.llm_model),
            provider="openai",
            finish_reason=meta.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=dt,
        )

    def provider(self) -> str: return "openai"
    def model(self) -> str | None: return settings.llm_model


class OllamaChat:
    def __init__(self):
        from langchain_community.chat_models import ChatOllama
        try:
            self.llm = ChatOllama(
                base_url=settings.ollama_url,
                model=settings.llm_model,
                temperature=0.2,
                client_kwargs={"timeout": settings.llm_timeout},  
            )
        except TypeError:
            # fallback za starije verzije
            self.llm = ChatOllama(
                base_url=settings.ollama_url,
                model=settings.llm_model,
                temperature=0.2,
                timeout=settings.llm_timeout,  
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
        try:
            out = self.llm.invoke(lmsgs)
        except Exception as e:
            from .errors import ProviderError
            raise ProviderError(str(e))
        dt = int((perf_counter() - t0) * 1000)

        text = getattr(out, "content", str(out))
        return ChatResult(
            content=text,
            model=settings.llm_model,
            provider="ollama",
            latency_ms=dt,
        )

    def provider(self) -> str: return "ollama"
    def model(self) -> str | None: return settings.llm_model


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
    if prov == "openai":
        return OpenAIChat()
    if prov == "ollama":
        return OllamaChat()
    return MockChat()
