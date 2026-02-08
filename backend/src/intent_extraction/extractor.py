"""Intent extraction from conversational input."""

import logging
from datetime import datetime
from pathlib import Path

from src.llm.ollama_client import OllamaClient
from src.llm.prompts import INTENT_EXTRACTION_SCHEMA, build_intent_extraction_prompt
from src.shared.schemas import ConversationMessage, DeploymentIntent

logger = logging.getLogger(__name__)

# Create prompts directory for easy copy-paste access
PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "logs" / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)


class IntentExtractor:
    """Extract structured deployment intent from natural language conversation."""

    def __init__(self, llm_client: OllamaClient | None = None):
        """
        Initialize intent extractor.

        Args:
            llm_client: Optional Ollama client (creates default if not provided)
        """
        self.llm_client = llm_client or OllamaClient()

    def extract_intent(
        self, user_message: str, conversation_history: list[ConversationMessage] | None = None
    ) -> DeploymentIntent:
        """
        Extract deployment intent from user message.

        Args:
            user_message: Latest user message
            conversation_history: Optional previous conversation context

        Returns:
            DeploymentIntent with extracted requirements

        Raises:
            ValueError: If extraction fails or LLM returns invalid data
        """
        # Convert conversation history to dict format for LLM
        history_dicts = None
        if conversation_history:
            history_dicts = [
                {"role": msg.role, "content": msg.content} for msg in conversation_history
            ]

        # Build extraction prompt
        prompt = build_intent_extraction_prompt(user_message, history_dicts)

        # Save prompt to file for easy copy-paste testing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_file = PROMPTS_DIR / f"intent_extraction_{timestamp}.txt"

        full_prompt_with_schema = f"{prompt}\n\n{INTENT_EXTRACTION_SCHEMA}"

        with open(prompt_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("INTENT EXTRACTION PROMPT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"User Message: {user_message}\n")
            f.write("=" * 80 + "\n\n")
            f.write(full_prompt_with_schema)
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("Copy everything above this line to test in other LLMs\n")
            f.write("=" * 80 + "\n")

        # Log the complete prompt being sent to LLM (always show at INFO level)
        logger.info("=" * 80)
        logger.info("[FULL INTENT EXTRACTION PROMPT - START]")
        logger.info(prompt)
        logger.info("[SCHEMA BEING USED]")
        logger.info(INTENT_EXTRACTION_SCHEMA)
        logger.info("[FULL INTENT EXTRACTION PROMPT - END]")
        logger.info(f"💾 Prompt saved to: {prompt_file}")
        logger.info("=" * 80)

        try:
            # Extract structured data from LLM
            extracted = self.llm_client.extract_structured_data(
                prompt,
                INTENT_EXTRACTION_SCHEMA,
                temperature=0.3,  # Lower temperature for more consistent extraction
            )

            # Log extracted intent
            logger.info(f"[EXTRACTED INTENT] {extracted}")

            # Validate and parse into Pydantic model
            intent = self._parse_extracted_intent(extracted)
            logger.info(f"Extracted intent: use_case={intent.use_case}, users={intent.user_count}")

            return intent

        except Exception as e:
            logger.error(f"Failed to extract intent: {e}")
            raise ValueError(f"Intent extraction failed: {e}") from e

    def _parse_extracted_intent(self, raw_data: dict) -> DeploymentIntent:
        """
        Parse and validate raw LLM output into DeploymentIntent.

        Args:
            raw_data: Raw dict from LLM

        Returns:
            Validated DeploymentIntent

        Raises:
            ValueError: If data is invalid
        """
        # Handle common LLM mistakes
        cleaned_data = self._clean_llm_output(raw_data)

        try:
            return DeploymentIntent(**cleaned_data)
        except Exception as e:
            logger.error(f"Failed to parse intent from: {cleaned_data}")
            raise ValueError(f"Invalid intent data: {e}") from e

    def _clean_llm_output(self, data: dict) -> dict:
        """
        Clean common LLM output mistakes.

        Args:
            data: Raw LLM output

        Returns:
            Cleaned data dict
        """
        cleaned = data.copy()

        # Fix use_case if it contains the full enum string
        if "use_case" in cleaned and "|" in str(cleaned["use_case"]):
            # LLM sometimes returns "chatbot|customer_service|..." instead of just "chatbot"
            # Take the first option
            cleaned["use_case"] = cleaned["use_case"].split("|")[0].strip()

        # Infer experience_class if not provided
        if "experience_class" not in cleaned or not cleaned.get("experience_class"):
            # Infer from use_case based on traffic_and_slos.md definitions
            use_case = cleaned.get("use_case", "")
            if use_case == "code_completion":
                cleaned["experience_class"] = "instant"  # Sub-200ms TTFT
            elif use_case in [
                "chatbot_conversational",
                "code_generation_detailed",
                "translation",
                "content_generation",
                "summarization_short",
            ]:
                cleaned["experience_class"] = "conversational"  # Interactive real-time
            elif use_case == "document_analysis_rag":
                cleaned["experience_class"] = "interactive"  # Can tolerate slight delay
            elif use_case == "long_document_summarization":
                cleaned["experience_class"] = "deferred"  # Quality over speed
            elif use_case == "research_legal_analysis":
                cleaned["experience_class"] = "batch"  # Background processing
            else:
                cleaned["experience_class"] = "conversational"  # Default
            logger.info(
                f"Inferred experience_class='{cleaned['experience_class']}' from use_case='{use_case}'"
            )

        # Fix user_count if it's a descriptive string instead of integer
        if "user_count" in cleaned and isinstance(cleaned["user_count"], str):
            # Extract integer from strings like "thousands of users (estimated: 5,000 - 10,000)"
            import re

            user_count_str = cleaned["user_count"]

            # Try to find numbers with commas or ranges
            # Match patterns like "5,000", "5000", "5,000 - 10,000", etc.
            numbers = re.findall(r"[\d,]+", user_count_str.replace(",", ""))

            if numbers:
                # If it's a range, take the midpoint
                if len(numbers) >= 2:
                    try:
                        low = int(numbers[0])
                        high = int(numbers[1])
                        cleaned["user_count"] = (low + high) // 2
                        logger.info(
                            f"Extracted user_count range [{low}, {high}], using midpoint: {cleaned['user_count']}"
                        )
                    except (ValueError, IndexError):
                        # Fallback to first number
                        cleaned["user_count"] = int(numbers[0])
                        logger.info(
                            f"Extracted user_count from string '{user_count_str}': {cleaned['user_count']}"
                        )
                else:
                    # Single number found
                    cleaned["user_count"] = int(numbers[0])
                    logger.info(
                        f"Extracted user_count from string '{user_count_str}': {cleaned['user_count']}"
                    )
            else:
                # No numbers found, try to infer from keywords
                user_count_str_lower = user_count_str.lower()
                if "thousand" in user_count_str_lower or "1k" in user_count_str_lower:
                    cleaned["user_count"] = 1000
                elif "million" in user_count_str_lower or "1m" in user_count_str_lower:
                    cleaned["user_count"] = 1000000
                elif "hundred" in user_count_str_lower:
                    cleaned["user_count"] = 100
                else:
                    # Default fallback
                    cleaned["user_count"] = 1000
                    logger.warning(
                        f"Could not parse user_count from '{user_count_str}', defaulting to 1000"
                    )

        # Ensure domain_specialization is a list
        if "domain_specialization" in cleaned:
            if isinstance(cleaned["domain_specialization"], str):
                # Convert single string to list
                cleaned["domain_specialization"] = [cleaned["domain_specialization"]]
            elif "|" in str(cleaned.get("domain_specialization", "")):
                # Handle "general|code" format
                cleaned["domain_specialization"] = [
                    d.strip() for d in cleaned["domain_specialization"].split("|")
                ]

        # Ensure priority fields have valid values (default to "medium" if invalid/missing)
        valid_priorities = ["low", "medium", "high"]
        for priority_field in [
            "accuracy_priority",
            "cost_priority",
            "latency_priority",
            "complexity_priority",
        ]:
            if priority_field in cleaned:
                # Normalize to lowercase and validate
                priority_value = str(cleaned[priority_field]).lower().strip()
                if priority_value not in valid_priorities:
                    logger.info(
                        f"Invalid {priority_field}='{cleaned[priority_field]}', defaulting to 'medium'"
                    )
                    cleaned[priority_field] = "medium"
                else:
                    cleaned[priority_field] = priority_value
            else:
                # Field not provided by LLM, default to medium
                cleaned[priority_field] = "medium"

        # Remove any unexpected fields that aren't in the schema
        valid_fields = DeploymentIntent.model_fields.keys()
        cleaned = {k: v for k, v in cleaned.items() if k in valid_fields}

        return cleaned

    def infer_missing_fields(self, intent: DeploymentIntent) -> DeploymentIntent:
        """
        Infer missing optional fields based on available information.

        Args:
            intent: Partially filled intent

        Returns:
            Intent with inferred fields
        """
        # Infer domain specialization from use case if not specified
        if intent.domain_specialization == ["general"]:
            if intent.use_case in ["code_generation_detailed", "code_completion"]:
                intent.domain_specialization = ["general", "code"]
            elif intent.use_case == "translation" or (
                "multilingual" in intent.additional_context.lower()
                if intent.additional_context
                else False
            ):
                intent.domain_specialization = ["general", "multilingual"]

        return intent
