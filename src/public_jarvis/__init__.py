"""Public-Jarvis: Orquestrador de agentes, roteamento multi-LLM e memória SQLite."""

from public_jarvis.config import JarvisSettings, load_settings
from public_jarvis.memory import (
    JarvisMemoryStore,
    MemoryEntry,
    TaskItem,
    cosine_similarity,
    deterministic_vectorize,
)
from public_jarvis.orchestrator import (
    AgentContext,
    JarvisOrchestrator,
    OrchestratorResult,
    classify_intent,
    paginate_text,
)
from public_jarvis.router import LLMResponse, MultiLLMRouter
from public_jarvis.security import (
    ALLOWED_BINARIES,
    ALLOWED_SUBCOMMANDS,
    SafeShellError,
    is_user_authorized,
    run_safe,
    validate_command,
)

__version__ = "0.1.0"

__all__ = [
    "JarvisSettings",
    "load_settings",
    "ALLOWED_BINARIES",
    "ALLOWED_SUBCOMMANDS",
    "SafeShellError",
    "validate_command",
    "run_safe",
    "is_user_authorized",
    "JarvisMemoryStore",
    "MemoryEntry",
    "TaskItem",
    "cosine_similarity",
    "deterministic_vectorize",
    "LLMResponse",
    "MultiLLMRouter",
    "AgentContext",
    "JarvisOrchestrator",
    "OrchestratorResult",
    "classify_intent",
    "paginate_text",
]
