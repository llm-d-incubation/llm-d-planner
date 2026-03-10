"""Intent-related schemas for deployment requirements."""

from typing import Literal

from pydantic import BaseModel, Field


class DeploymentIntent(BaseModel):
    """Extracted deployment requirements from user conversation."""

    use_case: Literal[
        "chatbot_conversational",
        "code_completion",
        "code_generation_detailed",
        "translation",
        "content_generation",
        "summarization_short",
        "document_analysis_rag",
        "long_document_summarization",
        "research_legal_analysis",
    ] = Field(..., description="Primary use case type")

    experience_class: Literal["instant", "conversational", "interactive", "deferred", "batch"] = (
        Field(..., description="User experience class defining latency expectations")
    )

    user_count: int = Field(..., description="Number of users or scale")

    domain_specialization: list[str] = Field(
        default_factory=lambda: ["general"],
        description="Domain requirements (general, code, multilingual, enterprise)",
    )

    # Hardware preference extracted from natural language
    preferred_gpu_types: list[str] = Field(
        default_factory=list,
        description="List of user's preferred GPU types (empty = any GPU). "
        "Canonical names: L4, A100-40, A100-80, H100, H200, B200",
    )

    # Priority hints extracted from natural language (used for weight calculation)
    accuracy_priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Accuracy/quality importance"
    )
    cost_priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Cost sensitivity (high = very cost sensitive)"
    )
    latency_priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Latency importance"
    )
    complexity_priority: Literal["low", "medium", "high"] = Field(
        default="medium", description="Preference for simpler deployments"
    )

    additional_context: str | None = Field(
        None, description="Any other relevant details from conversation"
    )


class ConversationMessage(BaseModel):
    """Single message in the conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str | None = None
