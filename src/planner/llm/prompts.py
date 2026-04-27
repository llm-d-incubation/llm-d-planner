"""Prompt templates for LLM interactions."""


def build_intent_extraction_prompt(
    user_message: str, conversation_history: list | None = None
) -> str:
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

    prompt = f"""Extract deployment requirements from the user message.

Return JSON only. No explanation.

{context}User message: {user_message}

Schema:
{{
  "use_case": "chatbot_conversational|code_completion|code_generation_detailed|translation|content_generation|summarization_short|document_analysis_rag|long_document_summarization|research_legal_analysis",
  "user_count": 0,
  "domain_specialization": [],
  "preferred_gpu_types": [],
  "preferred_models": [],
  "accuracy_mentioned": false,
  "accuracy_priority": "medium",
  "cost_mentioned": false,
  "cost_priority": "medium",
  "latency_mentioned": false,
  "latency_priority": "medium"
}}

Rules:

GENERAL:

* Output valid JSON only.

USE CASE:

* Apply the first matching rule from this list, top to bottom.
* If the message mentions "legal", "law", "lawyers", "attorneys", "research", or "legal analysis" => research_legal_analysis
* If the message mentions "code generation", "full code", or "implementing features" => code_generation_detailed
* If the message mentions "code completion" or "autocomplete" => code_completion
* If the message mentions "chatbot", "customer service", or "conversational" => chatbot_conversational
* If the message mentions "long document summarization" => long_document_summarization
* If the message mentions "translation" => translation
* If the message mentions "content generation", "marketing", or "blog" => content_generation
* If the message mentions "summarization" => summarization_short
* If the message mentions "RAG", "retrieval", "knowledge base", "document Q&A", or "document analysis" => document_analysis_rag

USER COUNT (number of people who will use the system):

* Count people, teams, employees, developers, attorneys, analysts, etc.
* Use explicit number if present.
* Otherwise estimate an integer.
* Must be >= 1.

DOMAIN:

* Start with an empty list.
* If use_case is code_completion or code_generation_detailed, include "code".
* If multilingual or translation is explicitly mentioned, include "multilingual".
* If enterprise or knowledge base is explicitly mentioned, include "enterprise".
* If the list is empty, use ["general"].

GPUs:

* Valid GPU names: L4, A100-40, A100-80, H100, H200, B200.
* Extract only if explicitly mentioned.
* "A100" (unspecified variant) => ["A100-80", "A100-40"].
* If no GPU is explicitly mentioned => [].

MODELS:

* GPU names (L4, A100, H100, H200, B200) are NOT models. Never put them in preferred_models.
* A Hugging Face model ID has the format "org/model-name" (e.g., "meta-llama/Llama-3.1-8B-Instruct").
* Extract only if explicitly mentioned.
* If the message contains a Hugging Face model ID, copy it exactly into preferred_models.
* If no model is explicitly mentioned => [].

ACCURACY / COST / LATENCY:

* Set *_mentioned = true ONLY if that topic is explicitly stated.
* NEVER infer these fields from use_case or context.
* If *_mentioned = false, then *_priority MUST be "medium".
* *_priority measures importance to the user.
* Wanting lower cost means cost_priority = "high".
* Wanting lower latency or faster responses means latency_priority = "high".
* Saying quality, accuracy, or correctness is critical means accuracy_priority = "high".
* Saying a topic is unimportant, does not matter, or is not a concern means that topic's priority = "low".
* Otherwise, when explicitly mentioned but not strongly emphasized => "medium".

Return valid JSON only."""
    return prompt
