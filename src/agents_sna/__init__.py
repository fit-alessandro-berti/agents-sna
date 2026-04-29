"""OpenRouter-backed multi-agent discussion orchestration."""

from agents_sna.config import AgentSpec, AgenticConfig, load_config
from agents_sna.orchestrator import AgentAnswer, AgenticOrchestrator, OrchestrationResult

__all__ = [
    "AgentAnswer",
    "AgentSpec",
    "AgenticConfig",
    "AgenticOrchestrator",
    "OrchestrationResult",
    "load_config",
]
