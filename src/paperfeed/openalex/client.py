"""OpenAlex REST API Client with polite pool, cursor pagination, and disk caching (Phase 1.1)."""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Generator, List, Optional
import httpx
from paperfeed.common.config import find_project_root, load_dotenv_file
from paperfeed.common.logger import get_logger

logger = get_logger("openalex.client")

DEFAULT_SELECT_FIELDS = [
    "id",
    "doi",
    "title",
    "publication_year",
    "publication_date",
    "type",
    "language",
    "authorships",
    "primary_location",
    "referenced_works",
    "counts_by_year",
    "abstract_inverted_index",
    "primary_topic",
    "topics",
]


class OpenAlexClient:
    """Robust OpenAlex API client supporting polite pool, API keys, rate limiting, and caching."""

    def __init__(
        self,
        base_url: str = "https://api.openalex.org",
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        raw_cache_dir: str | Path = "data/raw",
        requests_per_second: float = 10.0,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
    ) -> None:
        load_dotenv_file()
        self.base_url = base_url.rstrip("/")
        self.email = email or os.environ.get("OPENALEX_EMAIL", "researcher@example.edu")
        self.api_key = api_key or os.environ.get("OPENALEX_API_KEY")
        self.raw_cache_dir = Path(raw_cache_dir)
        if not self.raw_cache_dir.is_absolute():
            self.raw_cache_dir = find_project_root() / self.raw_cache_dir
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_delay = 1.0 / max(requests_per_second, 1.0)
        self.last_request_time = 0.0
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        headers = {
            "User-Agent": f"PaperFeed/0.1.0 (mailto:{self.email})",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["api_key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.session = httpx.Client(headers=headers, timeout=self.timeout_seconds)

    def _rate_limit(self) -> None:
        """Enforce rate limiting between outbound requests."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request_time = time.time()

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GET request with exponential backoff on transient errors."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if "mailto" not in params and self.email:
            params["mailto"] = self.email
        if self.api_key and "api_key" not in params:
            params["api_key"] = self.api_key

        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            self._rate_limit()
            try:
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    error_msg = ""
                    try:
                        error_data = response.json()
                        error_msg = f": {error_data.get('message', '')}"
                    except Exception:
                        pass
                    
                    if "Insufficient budget" in error_msg:
                        logger.error(
                            f"OpenAlex 429 Insufficient Budget{error_msg}. "
                            "Please set your free OpenAlex API key via OPENALEX_API_KEY environment variable "
                            "or in configs/default.yaml. Obtain one free at https://openalex.org"
                        )
                    else:
                        logger.warning(f"Rate limited (429) by OpenAlex{error_msg}. Backing off for {delay:.1f}s...")
                    
                    time.sleep(delay)
                    delay *= 2.0
                elif 500 <= response.status_code < 600:
                    logger.warning(f"OpenAlex server error ({response.status_code}). Retry {attempt}/{self.max_retries}...")
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    response.raise_for_status()
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt == self.max_retries:
                    logger.error(f"Failed request to {url} after {self.max_retries} attempts: {e}")
                    raise
                time.sleep(delay)
                delay *= 2.0

        raise RuntimeError(f"Exhausted retries for {url}")

    def fetch_work(self, work_id: str, select_fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single work by OpenAlex ID or DOI."""
        clean_id = work_id.replace("https://openalex.org/", "").strip()
        params = {}
        if select_fields:
            params["select"] = ",".join(select_fields)
        try:
            return self._get(f"works/{clean_id}", params)
        except Exception as e:
            logger.warning(f"Unable to fetch work {work_id}: {e}")
            return None

    def fetch_works_batch(
        self,
        work_ids: List[str],
        chunk_size: int = 50,
        select_fields: Optional[List[str]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Fetch multiple works in parallel batches using pipe-separated openalex_id filter."""
        clean_ids = [w.replace("https://openalex.org/", "").strip() for w in work_ids if w]
        clean_ids = list(dict.fromkeys(clean_ids))  # preserve order, unique
        for i in range(0, len(clean_ids), chunk_size):
            chunk = clean_ids[i : i + chunk_size]
            filter_str = f"openalex_id:{'|'.join(chunk)}"
            try:
                for work in self.paginate_works(
                    filter_str=filter_str,
                    select_fields=select_fields,
                    per_page=len(chunk),
                ):
                    yield work
            except Exception as e:
                logger.warning(f"Batch fetch error for chunk of {len(chunk)} works: {e}")

    def paginate_works(
        self,
        filter_str: str,
        select_fields: Optional[List[str]] = None,
        per_page: int = 200,
        max_results: Optional[int] = None,
        cache_prefix: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream works using cursor-based pagination.
        Yields work items one by one.
        """
        select = ",".join(select_fields or DEFAULT_SELECT_FIELDS)
        cursor = "*"
        fetched_total = 0

        while cursor:
            params = {
                "filter": filter_str,
                "select": select,
                "per-page": per_page,
                "cursor": cursor,
            }

            cache_file = None
            if cache_prefix:
                safe_cursor = cursor.replace("/", "_").replace("*", "start")
                cache_file = self.raw_cache_dir / f"{cache_prefix}_{safe_cursor}.json"

            if cache_file and cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = self._get("works", params)
                if cache_file:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f)

            results = data.get("results", [])
            if not results:
                break

            for work in results:
                yield work
                fetched_total += 1
                if max_results and fetched_total >= max_results:
                    return

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

    def close(self) -> None:
        """Close underlying HTTP client."""
        self.session.close()
