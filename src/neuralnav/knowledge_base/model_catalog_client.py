"""HTTP client for the RHOAI Model Catalog API."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# Default ServiceAccount token path inside a pod
_SA_TOKEN_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")


class ModelCatalogClient:
    """Thin HTTP wrapper for the Model Catalog v1alpha1 REST API.

    Returns raw dicts from the API -- no domain mapping is performed here.
    Configuration comes from constructor args or environment variables:

        MODEL_CATALOG_URL        -- base URL of the catalog service
        MODEL_CATALOG_TOKEN      -- bearer token (or auto-read from SA mount)
        MODEL_CATALOG_SOURCE_ID  -- catalog source name
        MODEL_CATALOG_VERIFY_SSL -- "true" / "false"
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        source_id: str | None = None,
        verify_ssl: bool | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv(
                "MODEL_CATALOG_URL",
                "https://model-catalog.rhoai-model-registries.svc:8443",
            )
        ).rstrip("/")

        self.token = token or os.getenv("MODEL_CATALOG_TOKEN") or self._read_sa_token()
        self.source_id = source_id or os.getenv(
            "MODEL_CATALOG_SOURCE_ID", "redhat_ai_validated_models"
        )

        verify_env = os.getenv("MODEL_CATALOG_VERIFY_SSL", "true")
        self.verify_ssl = verify_ssl if verify_ssl is not None else (verify_env.lower() == "true")

        self._api_base = f"{self.base_url}/api/model_catalog/v1alpha1"

    @staticmethod
    def _read_sa_token() -> str:
        """Read ServiceAccount token from pod mount."""
        if _SA_TOKEN_PATH.exists():
            return _SA_TOKEN_PATH.read_text().strip()
        return ""

    def _headers(self) -> dict[str, str]:
        """Build request headers with optional bearer token."""
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get_json(self, path: str) -> dict:
        """GET request returning parsed JSON."""
        url = f"{self._api_base}{path}"
        with httpx.Client(verify=self.verify_ssl, timeout=30.0) as http:
            resp = http.get(url, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    def list_models(self, page_size: int = 200) -> list[dict]:
        """List all models from the configured source.

        Handles pagination automatically via nextPageToken.

        Args:
            page_size: Number of items per page request.

        Returns:
            List of raw model dicts from the API.
        """
        all_items: list[dict] = []
        next_token = ""
        while True:
            token_param = f"&nextPageToken={next_token}" if next_token else ""
            data = self._get_json(
                f"/sources/{self.source_id}/models" f"?pageSize={page_size}{token_param}"
            )
            all_items.extend(data.get("items", []))
            next_token = data.get("nextPageToken", "")
            if not next_token:
                break
        logger.info("Fetched %d models from Model Catalog", len(all_items))
        return all_items

    def get_model_artifacts(self, model_name: str, page_size: int = 200) -> list[dict]:
        """Get all artifacts (model + metrics) for a model.

        Handles pagination automatically via nextPageToken.

        Args:
            model_name: Full model name (e.g. "RedHatAI/granite-3.1-8b-instruct").
            page_size: Number of items per page request.

        Returns:
            List of raw artifact dicts from the API.
        """
        encoded = quote(model_name, safe="")
        all_items: list[dict] = []
        next_token = ""
        while True:
            token_param = f"&nextPageToken={next_token}" if next_token else ""
            data = self._get_json(
                f"/sources/{self.source_id}/models/{encoded}/artifacts"
                f"?pageSize={page_size}{token_param}"
            )
            all_items.extend(data.get("items", []))
            next_token = data.get("nextPageToken", "")
            if not next_token:
                break
        return all_items
