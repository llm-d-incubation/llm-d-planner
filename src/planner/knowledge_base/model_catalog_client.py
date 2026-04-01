"""HTTP client for the RHOAI Model Catalog API."""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, cast
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
        MODEL_CATALOG_CA_BUNDLE  -- path to CA cert bundle (optional)
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        source_id: str | None = None,
        verify_ssl: bool | None = None,
        ca_bundle: str | None = None,
    ) -> None:
        _default_url = "https://model-catalog.rhoai-model-registries.svc:8443"
        env_url = (os.getenv("MODEL_CATALOG_URL") or "").strip()
        _base = (
            base_url.strip()
            if base_url is not None and base_url.strip()
            else env_url or _default_url
        )
        self.base_url = _base.rstrip("/")

        self.token = token or os.getenv("MODEL_CATALOG_TOKEN") or self._read_sa_token()
        env_source = (os.getenv("MODEL_CATALOG_SOURCE_ID") or "").strip()
        self.source_id = (
            source_id.strip()
            if source_id is not None and source_id.strip()
            else env_source or "redhat_ai_validated_models"
        )

        verify_env = os.getenv("MODEL_CATALOG_VERIFY_SSL", "true").strip().lower()
        self.verify_ssl = (
            verify_ssl if verify_ssl is not None else verify_env not in {"0", "false", "no", "off"}
        )
        self._ca_bundle = ca_bundle or os.getenv("MODEL_CATALOG_CA_BUNDLE", "")
        self._api_base = f"{self.base_url}/api/model_catalog/v1alpha1"

        # Client-level cache
        self._models_cache: list[dict] | None = None
        self._models_loaded_at: float = 0
        self._artifacts_cache: dict[tuple[str, str], tuple[float, list[dict]]] = {}

        # Persistent HTTP client for connection pooling
        self._http: httpx.Client | None = None
        self._http_has_ca: bool = False
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()

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

    def _resolve_verify(self) -> bool | str:
        """Return the verify parameter for httpx: CA path if set, else bool."""
        if self._ca_bundle and Path(self._ca_bundle).exists():
            return self._ca_bundle
        return self.verify_ssl

    def _get_http(self) -> httpx.Client:
        """Return a persistent httpx.Client, creating it on first use.

        If a CA bundle is configured but wasn't available when the client was
        created (e.g., OpenShift service-ca not yet injected), the client is
        recreated once the CA file appears.
        """
        with self._lock:
            if (
                self._http is not None
                and not self._http.is_closed
                and not self._http_has_ca
                and self._ca_bundle
                and Path(self._ca_bundle).exists()
            ):
                self._http.close()
                self._http = None
            if self._http is None or self._http.is_closed:
                verify = self._resolve_verify()
                self._http_has_ca = isinstance(verify, str)
                self._http = httpx.Client(
                    verify=verify,
                    timeout=60.0,
                    headers=self._headers(),
                )
            return self._http

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http is not None and not self._http.is_closed:
            self._http.close()

    def _get_json(
        self, path: str, params: dict[str, Any] | None = None, *, max_retries: int = 3
    ) -> dict[str, Any]:
        """GET request returning parsed JSON with bounded retries."""
        url = f"{self._api_base}{path}"
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self._get_http().get(url, params=params)
                resp.raise_for_status()
                return cast(dict[str, Any], resp.json())
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
                last_exc = exc
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                last_exc = exc
            delay = (2**attempt) + random.uniform(0, 1)
            logger.warning(
                "Retrying %s (attempt %d/%d) after %.1fs", path, attempt + 1, max_retries, delay
            )
            time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _is_cache_stale(self, loaded_at: float) -> bool:
        return (time.time() - loaded_at) >= _CLIENT_CACHE_TTL

    def list_models(self, page_size: int = 200) -> list[dict]:
        """List all models from the catalog.

        Results are cached for _CLIENT_CACHE_TTL seconds.
        Handles pagination automatically via nextPageToken.
        """
        with self._cache_lock:
            if self._models_cache is not None and not self._is_cache_stale(self._models_loaded_at):
                return self._models_cache

            all_items: list[dict] = []
            next_token = ""
            while True:
                params: dict[str, str | int] = {"pageSize": page_size}
                if next_token:
                    params["nextPageToken"] = next_token
                data = self._get_json("/models", params=params)
                all_items.extend(data.get("models", data.get("items", [])))
                next_token = data.get("nextPageToken", "")
                if not next_token:
                    break
            logger.info("Fetched %d models from Model Catalog", len(all_items))
            self._models_cache = all_items
            self._models_loaded_at = time.time()
            return all_items

    def get_model_artifacts(
        self, model_name: str, source_id: str | None = None, page_size: int = 200
    ) -> list[dict]:
        """Get all artifacts (model + metrics) for a model.

        Results are cached per (model_name, source_id) for _CLIENT_CACHE_TTL seconds.
        Handles pagination automatically via nextPageToken.
        """
        source: str = source_id if source_id is not None else self.source_id
        cache_key: tuple[str, str] = (model_name, source)
        with self._cache_lock:
            cached = self._artifacts_cache.get(cache_key)
            if cached is not None and not self._is_cache_stale(cached[0]):
                return cached[1]

            encoded_source = quote(source, safe="")
            encoded_name = quote(model_name, safe="")
            all_items: list[dict] = []
            next_token = ""
            while True:
                params: dict[str, str | int] = {"pageSize": page_size}
                if next_token:
                    params["nextPageToken"] = next_token
                data = self._get_json(
                    f"/sources/{encoded_source}/models/{encoded_name}/artifacts",
                    params=params,
                )
                all_items.extend(data.get("artifacts", data.get("items", [])))
                next_token = data.get("nextPageToken", "")
                if not next_token:
                    break
            self._artifacts_cache[cache_key] = (time.time(), all_items)
            return all_items
