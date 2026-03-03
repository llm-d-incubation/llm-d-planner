"""HTTP client for the RHOAI Model Catalog API."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# Default ServiceAccount token path inside a pod
_SA_TOKEN_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")

# Client-level cache TTL (1 hour, matches source-level TTL)
_CLIENT_CACHE_TTL = 3600


class ModelCatalogClient:
    """Thin HTTP wrapper for the Model Catalog v1alpha1 REST API.

    Returns raw dicts from the API -- no domain mapping is performed here.
    Caches list_models() and get_model_artifacts() results so multiple
    sources sharing the same client avoid redundant HTTP calls.

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

        # Client-level cache
        self._models_cache: list[dict] | None = None
        self._artifacts_cache: dict[str, list[dict]] = {}
        self._cache_loaded_at: float = 0

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
        with httpx.Client(verify=self.verify_ssl, timeout=60.0) as http:
            resp = http.get(url, headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    def _is_cache_stale(self) -> bool:
        return (time.time() - self._cache_loaded_at) >= _CLIENT_CACHE_TTL

    def list_models(self, page_size: int = 200) -> list[dict]:
        """List all models from the catalog.

        Results are cached for _CLIENT_CACHE_TTL seconds.
        Handles pagination automatically via nextPageToken.
        """
        if self._models_cache is not None and not self._is_cache_stale():
            return self._models_cache

        all_items: list[dict] = []
        next_token = ""
        while True:
            token_param = f"&nextPageToken={next_token}" if next_token else ""
            data = self._get_json(f"/models?pageSize={page_size}{token_param}")
            all_items.extend(data.get("models", data.get("items", [])))
            next_token = data.get("nextPageToken", "")
            if not next_token:
                break
        logger.info("Fetched %d models from Model Catalog", len(all_items))
        self._models_cache = all_items
        self._cache_loaded_at = time.time()
        return all_items

    def get_model_artifacts(
        self, model_name: str, source_id: str | None = None, page_size: int = 200
    ) -> list[dict]:
        """Get all artifacts (model + metrics) for a model.

        Results are cached per model_name for _CLIENT_CACHE_TTL seconds.
        Handles pagination automatically via nextPageToken.
        """
        if model_name in self._artifacts_cache and not self._is_cache_stale():
            return self._artifacts_cache[model_name]

        source = source_id or self.source_id
        encoded_source = quote(source, safe="")
        encoded_name = quote(model_name, safe="")
        all_items: list[dict] = []
        next_token = ""
        while True:
            token_param = f"&nextPageToken={next_token}" if next_token else ""
            data = self._get_json(
                f"/sources/{encoded_source}/models/{encoded_name}/artifacts"
                f"?pageSize={page_size}{token_param}"
            )
            all_items.extend(data.get("artifacts", data.get("items", [])))
            next_token = data.get("nextPageToken", "")
            if not next_token:
                break
        self._artifacts_cache[model_name] = all_items
        return all_items
