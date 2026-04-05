"""
Jina Search API implementation.
"""

import requests

from search.base import BaseSearch
from utils import load_config


class JinaSearch(BaseSearch):
    """Search backend powered by the Jina Search API."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        cfg = config["search"]
        limits = config["limits"]

        self.api_key: str = cfg["jina_api_key"]
        self.endpoint: str = cfg["jina_endpoint"]
        self.proxies: dict | None = cfg.get("proxies") if cfg.get("use_proxy") else None
        self._max_desc_len: int = limits.get("max_source_desc_len", 500)

    def visit_url(self, url: str) -> dict:
        """Fetch a URL directly via Jina Reader (r.jina.ai) and return as a single-source result."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            resp = requests.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                proxies=self.proxies,
                timeout=20,
            )
            resp.raise_for_status()
            content = resp.text.strip()
            if not content:
                return {"sources": {}, "error": f"Empty content from {url}"}

            # Extract title from first markdown heading
            title = url
            for line in content.splitlines()[:15]:
                if line.startswith("# "):
                    title = line[2:].strip()[:120]
                    break

            sources = {
                "source1": {
                    "url": url,
                    "title": title,
                    "description": content[: self._max_desc_len],
                }
            }
            return {"sources": sources, "error": None}
        except Exception as exc:
            return {"sources": {}, "error": f"{type(exc).__name__}: {exc}"}

    def search(self, query: str, max_results: int = 5) -> dict:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Respond-With": "no-content",
        }
        url = f"{self.endpoint}/?q={requests.utils.quote(query)}"

        try:
            resp = requests.get(
                url, headers=headers, proxies=self.proxies, timeout=12
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                return {"sources": {}, "error": f"No results for: {query}"}

            sources = {}
            for i, item in enumerate(data[:max_results], 1):
                title = (item.get("title") or "").strip()[:120]
                description = (item.get("description") or "").strip()[: self._max_desc_len]
                link = (item.get("url") or "").strip()
                if description and link:
                    sources[f"source{i}"] = {
                        "url": link,
                        "title": title,
                        "description": description,
                    }

            if sources:
                return {"sources": sources, "error": None}
            return {"sources": {}, "error": "No valid content in results"}

        except Exception as exc:
            return {"sources": {}, "error": f"{type(exc).__name__}: {exc}"}
