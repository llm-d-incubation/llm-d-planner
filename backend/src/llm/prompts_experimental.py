"""
Experimental prompt templates for future conversational features.

These prompts are not currently used in the production codebase but are
preserved for potential Phase 2 conversational interface enhancements.

DO NOT DELETE - These may be useful for future multi-turn dialogue features.
"""

CONVERSATIONAL_RESPONSE_TEMPLATE = """You are a helpful assistant for NeuralNav.

The user is working on deploying a Large Language Model for their use case. You are here to have a natural conversation with them to understand their needs.

Current context:
{context}

User message: {user_message}

Based on what we know so far:
{current_understanding}

Respond naturally to the user. If we still need critical information (use case, scale, latency requirements), ask clarifying questions in a conversational way. If we have enough information, let them know we're ready to generate recommendations.

Keep your response concise (2-3 sentences max).
"""


def build_conversational_prompt(
    user_message: str, current_understanding: dict, conversation_history: list = None
) -> str:
    """
    Build prompt for conversational AI responses.

    NOTE: This function is not currently used but preserved for future
    multi-turn conversation features where the system asks follow-up
    questions to gather more detailed requirements.

    Args:
        user_message: Latest user message
        current_understanding: Current extracted deployment intent
        conversation_history: Previous conversation messages

    Returns:
        Formatted prompt
    """
    context = ""
    if conversation_history:
        context = "Previous messages:\n"
        for msg in conversation_history[-2:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context += f"- {role}: {content}\n"

    understanding = ""
    if current_understanding:
        understanding = f"""- Use case: {current_understanding.get('use_case', 'unknown')}
- User count: {current_understanding.get('user_count', 'unknown')}
- Latency requirement: {current_understanding.get('latency_requirement', 'unknown')}
"""

    return CONVERSATIONAL_RESPONSE_TEMPLATE.format(
        context=context, user_message=user_message, current_understanding=understanding
    )


YAML_EXPLANATION_TEMPLATE = """Explain the following KServe deployment configuration in simple terms for a user who may not be familiar with Kubernetes:

{yaml_content}

Provide a brief 2-3 sentence explanation of:
1. What model is being deployed
2. What GPU resources are being used
3. Key configuration settings (replicas, scaling, etc.)

Keep it non-technical and focused on the business value.
"""
