"""Ollama client wrapper for LLM interactions."""

import json
import logging
from typing import Any

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama library not available. LLM features will be limited.")


logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama LLM service."""

    def __init__(self, model: str = "qwen2.5:7b", host: str | None = None):
        """
        Initialize Ollama client.

        Args:
            model: Model name to use (default: llama3.1:8b)
            host: Optional Ollama host URL (defaults to localhost:11434)
        """
        self.model = model
        self.host = host

        if not OLLAMA_AVAILABLE:
            logger.error("Ollama library not installed. Install with: pip install ollama")

    def chat(
        self,
        messages: list[dict[str, str]],
        format_json: bool = False,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Send chat messages to Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            format_json: If True, request JSON formatted response
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Response dict with 'message' containing 'content'
        """
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama library not available")

        try:
            # Log the request (last message is typically the user prompt)
            if messages:
                last_msg = messages[-1]
                logger.info(
                    f"[LLM REQUEST] Role: {last_msg.get('role')}, Content length: {len(last_msg.get('content', ''))} chars"
                )
                logger.debug(
                    f"[LLM PROMPT] {last_msg.get('content', '')[:500]}..."
                )  # Log first 500 chars at debug level

            kwargs = {
                "model": self.model,
                "messages": messages,
                "options": {"temperature": temperature},
            }

            if format_json:
                kwargs["format"] = "json"

            if self.host:
                kwargs["host"] = self.host

            response = ollama.chat(**kwargs)

            # Log the full response
            response_content = response.get("message", {}).get("content", "")
            logger.info("=" * 80)
            logger.info(
                f"[LLM RESPONSE] Model: {self.model}, Response length: {len(response_content)} chars"
            )
            logger.info("[LLM RESPONSE CONTENT - START]")
            logger.info(response_content)
            logger.info("[LLM RESPONSE CONTENT - END]")
            logger.info("=" * 80)

            return response

        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise

    def generate_completion(
        self,
        prompt: str,
        format_json: bool = False,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate completion for a prompt.

        Args:
            prompt: Input prompt string
            format_json: If True, request JSON formatted response
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        logger.info(
            f"[LLM GENERATE] Prompt length: {len(prompt)} chars, JSON format: {format_json}, Temperature: {temperature}"
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, format_json=format_json, temperature=temperature)
        return response["message"]["content"]

    def extract_structured_data(
        self,
        prompt: str,
        schema_description: str,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """
        Extract structured data from prompt using JSON format.

        Args:
            prompt: Input prompt describing what to extract
            schema_description: Description of expected JSON schema
            temperature: Lower temperature for more consistent extraction

        Returns:
            Parsed JSON dict
        """
        full_prompt = f"""{prompt}

{schema_description}

Return ONLY valid JSON matching the schema above. Do not include any explanation or additional text."""

        response_text = self.generate_completion(
            full_prompt, format_json=True, temperature=temperature
        )

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response_text}")
            logger.error(f"JSON error: {e}")
            raise ValueError(f"LLM did not return valid JSON: {e}") from e

    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        if not OLLAMA_AVAILABLE:
            return False

        try:
            # Try to list models to verify connection
            ollama.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama service not available: {e}")
            return False

    def ensure_model_pulled(self) -> bool:
        """
        Ensure the configured model is pulled locally.

        Returns:
            True if model is available, False otherwise
        """
        if not OLLAMA_AVAILABLE:
            return False

        try:
            models = ollama.list()
            model_names = [m["name"] for m in models.get("models", [])]

            if self.model not in model_names:
                logger.info(f"Pulling model {self.model}...")
                ollama.pull(self.model)
                logger.info(f"Model {self.model} pulled successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to pull model {self.model}: {e}")
            return False
