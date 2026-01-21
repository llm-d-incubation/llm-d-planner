"""Prompt templates for LLM interactions."""

INTENT_EXTRACTION_SCHEMA = """
Expected JSON schema:
{
  "use_case": "chatbot_conversational|code_completion|code_generation_detailed|translation|content_generation|summarization_short|document_analysis_rag|long_document_summarization|research_legal_analysis",
  "experience_class": "instant|conversational|interactive|deferred|batch",
  "user_count": <integer>,
  "domain_specialization": ["general"|"code"|"multilingual"|"enterprise"],
  "preferred_gpu_types": ["<list of GPU types if mentioned, empty list if not specified>"],
  "accuracy_priority": "low|medium|high",
  "cost_priority": "low|medium|high",
  "latency_priority": "low|medium|high",
  "complexity_priority": "low|medium|high",
  "additional_context": "<any other relevant details mentioned>"
}

Use case descriptions:
- chatbot_conversational: Real-time conversational chatbots (short prompts, short responses)
- code_completion: Fast code completion/autocomplete (short prompts, short completions)
- code_generation_detailed: Detailed code generation with explanations (medium prompts, long responses)
- translation: Document translation (medium prompts, medium responses)
- content_generation: Content creation, marketing copy (medium prompts, medium responses)
- summarization_short: Short document summarization (medium prompts, short summaries)
- document_analysis_rag: RAG-based document Q&A (long prompts with context, medium responses)
- long_document_summarization: Long document summarization (very long prompts, medium summaries)
- research_legal_analysis: Research/legal document analysis (very long prompts, detailed analysis)

Experience class guidance:
- instant: Extremely low latency required (<200ms TTFT) - code completion, autocomplete
- conversational: Real-time user interaction (chatbots, interactive tools) - low latency needed
- interactive: User waiting but can tolerate slight delay (RAG Q&A, content generation) - balanced
- deferred: User can wait for quality (long summarization, detailed analysis) - quality over speed
- batch: Background/async processing (research, legal analysis) - optimize for quality and cost
"""


def build_intent_extraction_prompt(user_message: str, conversation_history: list = None) -> str:
    """
    Build prompt for extracting deployment intent from user conversation.

    Args:
        user_message: Latest user message
        conversation_history: Optional list of previous messages

    Returns:
        Formatted prompt string
    """
    context = ""
    if conversation_history:
        context = "Previous conversation:\n"
        for msg in conversation_history[-3:]:  # Last 3 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context += f"{role}: {content}\n"
        context += "\n"

    prompt = f"""You are an expert AI assistant for NeuralNav helping users deploy Large Language Models (LLMs) for production use cases.

{context}Current user message: {user_message}

Your task is to extract structured information about their deployment requirements. Analyze what they've said and infer:

1. **Use case**: What type of application (chatbot, customer service, code generation, summarization, etc.)
2. **User count**: How many users or scale mentioned (estimate if not explicit)
3. **Domain specialization**: Any specific domains mentioned (code, multilingual, enterprise, etc.)
4. **Latency requirement**: How important is low latency? (very_high = sub-500ms, high = sub-2s, medium = 2-5s, low = >5s acceptable)
5. **Throughput priority**: Is high request volume more important than low latency?
6. **Budget constraint**: How price-sensitive are they?
7. **Domain specialization**: Any specific domains mentioned (code, multilingual, enterprise, etc.)
8. **Preferred GPU(s)**: If user mentions specific GPU types (H100, H200, A100, A100-80, A100-40, L4, B200), extract them as a list

Be intelligent about inference:
- "thousands of users" → estimate specific number
- "document Q&A" or "knowledge base" or "document search" → use_case: document_analysis_rag
- "RAG" or "retrieval" → use_case: document_analysis_rag
- "chatbot" or "customer service" or "conversational" → use_case: chatbot_conversational
- "summarize document" or "summarization" → use_case: summarization_short or long_document_summarization

GPU extraction examples (canonical names: L4, A100-40, A100-80, H100, H200, B200):
- "running on h200" or "h200" or "H200" → preferred_gpu_types: ["H200"]
- "h100 or h200" → preferred_gpu_types: ["H100", "H200"]
- "a100" or "A100" (unspecified variant) → preferred_gpu_types: ["A100-80", "A100-40"]
- "a100-80" or "A100-80GB" → preferred_gpu_types: ["A100-80"]
- "l4" or "L4" → preferred_gpu_types: ["L4"]
- No GPU mentioned → preferred_gpu_types: []

Priority extraction (for scoring weights - use "medium" as baseline, adjust based on context):
- accuracy_priority: "high" if user mentions accuracy matters, quality is important, accuracy is critical, best model, or top quality. "low" if user says good enough or accuracy less important.
- cost_priority: "high" if user mentions cost-effective, cost-sensitive, budget constrained, minimize cost, cost is important, or budget is tight. "low" if user says cost doesn't matter or budget is unlimited. Default to "medium" if not mentioned.
- latency_priority: "high" if the use case requires fast responses (e.g., real-time, interactive, instant). "low" if async/batch is acceptable.
- complexity_priority: "high" if user wants simple deployment, easy setup. "low" if they're okay with complex setups.
IMPORTANT: Explicit user statements override inferences (e.g., "cost-effective preferred" → cost_priority: high)

{INTENT_EXTRACTION_SCHEMA}
"""
    return prompt


# NOTE: Experimental prompts for future conversational features have been
# moved to prompts_experimental.py to keep this file focused on production code.
