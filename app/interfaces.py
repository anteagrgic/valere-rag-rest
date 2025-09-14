from typing import Protocol, Iterable, List, Dict, Any, TypedDict, Literal, runtime_checkable
from dataclasses import dataclass
from langchain.schema import Document

# poruke i rezultat chata
Role = Literal["system", "user", "assistant"]

class ChatMessage(TypedDict):
    role: Role
    content: str

@dataclass
class ChatResult:
    content: str
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None

@runtime_checkable
class SupportsChat(Protocol):
    def invoke(self, messages: List[ChatMessage]) -> ChatResult: ...
    def provider(self) -> str: ...
    def model(self) -> str | None: ...

# kompatibilnost unatrag: stari naziv pokazuje na novo sučelje
LLMChat = SupportsChat

class EmbeddingService(Protocol):
    def embed_texts(self, texts: List[str]) -> List[List[float]]: ...
    def dim(self) -> int: ...

class VectorStore(Protocol):
    def upsert(self, docs: Iterable[Document]) -> int: ...
    def as_retriever(self, k: int): ...  # returns a callable: str -> List[Document]

class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> List[Document]: ...

class RAGChain(Protocol):
    def answer(self, question: str, k: int = 5) -> tuple[str, List[Document]]: ...
