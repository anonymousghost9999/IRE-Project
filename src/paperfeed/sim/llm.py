"""Track 2 simulator: LLM ratings of papers for a simulated researcher (plan 9a / docs/design.md §13.2 spec points 1-3).

- ``GeminiClient``: one call to a pinned Gemini model through the AI Studio REST API (key from ``GEMINI_API_KEY``,
  never stored), JSON output, temperature and seed fixed by the caller. Rate-limit replies (HTTP 429 / 503) wait and
  retry; they do not count as malformed replies.
- ``parse_ratings``: strict parsing; every shown paper exactly once, integer ratings 1-5.
- ``RatingCache``: one rating per (persona, goal phase, paper, seed, model, prompt version), appended to a JSONL file
  that is released with the results. Only uncached papers are sent, so every arm sees the same rating for the same
  paper (common random numbers).
- ``Rater``: cache lookup, batches of at most 10 papers, up to 3 retries of a malformed reply, then ``RatingFailed``
  (the episode is marked failed, never imputed).
Only public OpenAlex metadata (titles, abstracts, subfield names) enters a prompt.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple, Union

from paperfeed.common.logger import get_logger

logger = get_logger("sim.llm")

PROMPT_VERSION = "track2-v1"
MAX_BATCH = 10
ABSTRACT_CHARS = 1200  # long abstracts are cut so a 10-paper call stays small; the cut is part of the prompt version

SYSTEM_INSTRUCTION = (
    "You are simulating a researcher who is browsing a feed of scientific papers. Judge each paper only by how useful "
    "it is for the researcher's current goal. Answer with JSON only."
)


@dataclass(frozen=True)
class PaperForRating:
    paper_id: str
    title: str
    abstract: str = ""


def build_prompt(persona_text: str, goal_text: str, papers: Sequence[PaperForRating]) -> str:
    """The rating prompt: persona, current goal, the papers, and the required JSON reply."""
    lines = [
        f"Researcher: {persona_text.strip()}",
        f"Current goal: {goal_text.strip()}",
        "",
        "Rate every paper below from 1 to 5 for this goal: 5 = must read now, 4 = clearly relevant, 3 = somewhat "
        "relevant, 2 = barely relevant, 1 = irrelevant.",
        "",
    ]
    for p in papers:
        abstract = (p.abstract or "").strip()
        if len(abstract) > ABSTRACT_CHARS:
            abstract = abstract[:ABSTRACT_CHARS].rsplit(" ", 1)[0] + " ..."
        lines += [f"[{p.paper_id}] {p.title.strip()}", abstract or "(no abstract)", ""]
    lines.append('Reply as {"ratings": [{"id": "<paper id>", "rating": <1-5>}, ...]} with every paper exactly once.')
    return "\n".join(lines)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class RatingParseError(ValueError):
    """The reply is not exactly one integer rating 1-5 for every shown paper."""


def parse_ratings(text: str, expected_ids: Sequence[str]) -> Dict[str, int]:
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RatingParseError(f"not JSON: {exc}") from exc
    items = data.get("ratings") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise RatingParseError('missing "ratings" list')
    out: Dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict) or "id" not in item or "rating" not in item:
            raise RatingParseError(f"bad item {item!r}")
        pid, rating = str(item["id"]).strip().strip("[]"), item["rating"]
        if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
            raise RatingParseError(f"rating for {pid!r} is not an integer 1-5: {rating!r}")
        if pid in out:
            raise RatingParseError(f"{pid!r} rated twice")
        out[pid] = rating
    if set(out) != set(expected_ids):
        missing, extra = set(expected_ids) - set(out), set(out) - set(expected_ids)
        raise RatingParseError(f"ids differ from the shown papers (missing {sorted(missing)[:3]}, extra {sorted(extra)[:3]})")
    return out


class LLMClient(Protocol):
    model: str

    def generate(self, prompt: str, temperature: float, seed: int) -> Tuple[str, Dict[str, Any]]:
        """(reply text, metadata such as the model version the provider reports)."""


class RateLimited(RuntimeError):
    def __init__(self, wait_seconds: float):
        super().__init__(f"rate limited; retry after {wait_seconds:.0f}s")
        self.wait_seconds = wait_seconds


class GeminiClient:
    """Gemini through the AI Studio REST API (``generateContent``), JSON output mode."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: str, api_key: Optional[str] = None, timeout: float = 60.0):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set the GEMINI_API_KEY environment variable (an AI Studio key); it is never stored.")
        self.model = model
        self._key = key
        self.timeout = timeout
        self.last_headers: Dict[str, str] = {}

    def list_models(self) -> List[str]:
        """Models this key can call with generateContent."""
        import httpx

        resp = httpx.get("https://generativelanguage.googleapis.com/v1beta/models", timeout=self.timeout,
                         headers={"x-goog-api-key": self._key}, params={"pageSize": 200})
        resp.raise_for_status()
        return sorted(m["name"].removeprefix("models/") for m in resp.json().get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", []))

    def generate(self, prompt: str, temperature: float, seed: int) -> Tuple[str, Dict[str, Any]]:
        import httpx

        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "seed": seed, "responseMimeType": "application/json"},
        }
        resp = httpx.post(self.URL.format(model=self.model), json=body, timeout=self.timeout,
                          headers={"x-goog-api-key": self._key})
        self.last_headers = rate_limit_headers(resp.headers)
        if resp.status_code in (429, 503):
            raise RateLimited(float(resp.headers.get("retry-after", 30)))
        resp.raise_for_status()
        data = resp.json()
        try:
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError) as exc:
            raise RatingParseError(f"no text in the reply: {str(data)[:200]}") from exc
        return text, {"model_version": data.get("modelVersion", self.model), "usage": data.get("usageMetadata", {})}


# OpenAI-format chat APIs (NVIDIA NIM, Mistral, Groq, OpenRouter, Cerebras).
OPENAI_COMPATIBLE: Dict[str, Tuple[str, str]] = {
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
}

RATE_LIMIT_HEADER_WORDS = ("ratelimit", "rate-limit", "remaining", "retry-after", "reset", "quota", "limit")


def rate_limit_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    """The reply headers that describe rate limits (names differ between providers, so match loosely)."""
    return {k: v for k, v in dict(headers).items() if any(w in k.lower() for w in RATE_LIMIT_HEADER_WORDS)}


class KeyPoolRateLimiter:
    """Thread-safe rate limiter managing a pool of API keys for concurrent requests.

    Features:
    - Enforces a per-key rate limit (e.g. 20 RPM per key -> interval of 60/20 = 3.0s between calls on the same key).
    - Concurrently distributes requests across all active keys (N keys @ 20 RPM -> N * 20 total RPM throughput).
    - Per-key cooldown on HTTP 429/503: cools down only the affected key while remaining keys continue serving requests.
    - Automatic load balancing: acquires the available key with the lowest in-flight usage.
    """

    def __init__(
        self,
        keys: Sequence[str],
        rpm_per_key: float = 20.0,
        min_interval: Optional[float] = None,
    ):
        clean_keys = [k.strip() for k in keys if k and k.strip()]
        if not clean_keys:
            raise RuntimeError(
                "No API keys provided. Please provide them via environment variable (e.g. NVIDIA_API_KEYS) "
                "or in data/artifacts/nvidia_keys.txt (one key per line)."
            )
        self.keys = clean_keys
        self.rpm = float(rpm_per_key)
        self.min_interval = min_interval if min_interval is not None else (
            60.0 / self.rpm if self.rpm > 0 else 0.0
        )
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._history: Dict[str, List[float]] = {k: [] for k in self.keys}
        self._cooldown_until: Dict[str, float] = {k: 0.0 for k in self.keys}
        self._in_flight: Dict[str, int] = {k: 0 for k in self.keys}
        self._disabled: Set[str] = set()

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def active_keys(self) -> List[str]:
        with self._lock:
            return [k for k in self.keys if k not in self._disabled]

    def has_ready_key(self) -> bool:
        """Check if any active key is currently outside its cooldown window."""
        now = time.time()
        with self._lock:
            for k in self.keys:
                if k not in self._disabled and self._cooldown_until[k] <= now:
                    return True
        return False

    def acquire_key(self, timeout: float = 300.0) -> str:
        """Acquire a key ready to serve a request under its rate limit. Blocks until one is ready."""
        deadline = time.time() + timeout
        with self._cond:
            while True:
                now = time.time()
                active = [k for k in self.keys if k not in self._disabled]
                if not active:
                    raise RuntimeError("All API keys in the pool have been disabled or exhausted.")

                # Prune history older than 60 seconds
                for k in active:
                    self._history[k] = [t for t in self._history[k] if now - t < 60.0]

                # Check for keys ready right now
                best_key: Optional[str] = None
                best_in_flight = float("inf")

                for k in active:
                    if self._cooldown_until[k] > now:
                        continue
                    if self.rpm > 0 and len(self._history[k]) >= self.rpm:
                        continue
                    if self.min_interval > 0 and self._history[k] and (now - self._history[k][-1] < self.min_interval):
                        continue
                    if self._in_flight[k] < best_in_flight:
                        best_in_flight = self._in_flight[k]
                        best_key = k

                if best_key is not None:
                    self._history[best_key].append(now)
                    self._in_flight[best_key] += 1
                    return best_key

                # Compute wait time until the earliest key becomes ready
                next_ready_times = []
                for k in active:
                    t_cool = self._cooldown_until[k]
                    t_interval = (self._history[k][-1] + self.min_interval) if (self.min_interval > 0 and self._history[k]) else now
                    t_window = (self._history[k][0] + 60.0) if (self.rpm > 0 and len(self._history[k]) >= self.rpm) else now
                    next_ready_times.append(max(now, t_cool, t_interval, t_window))

                wait_sec = max(0.02, min(next_ready_times) - now)
                if now + wait_sec > deadline:
                    raise TimeoutError(f"Timed out after {timeout}s waiting for an available API key.")
                self._cond.wait(timeout=wait_sec)

    def release_key(
        self,
        key: str,
        rate_limited_wait: Optional[float] = None,
        disabled: bool = False,
    ) -> None:
        """Release the in-flight hold on a key and update its rate-limit/cooldown status."""
        with self._cond:
            if key in self._in_flight:
                self._in_flight[key] = max(0, self._in_flight[key] - 1)
            if disabled:
                self._disabled.add(key)
                logger.error("API key %s... disabled from pool (unauthorized or quota exhausted).", key[:8])
            elif rate_limited_wait is not None and rate_limited_wait > 0:
                self._cooldown_until[key] = time.time() + rate_limited_wait
                logger.warning("API key %s... rate limited; cooling down for %.1fs.", key[:8], rate_limited_wait)
            self._cond.notify_all()


def load_api_keys(
    provider: str = "nvidia",
    explicit_keys: Optional[Union[str, Sequence[str]]] = None,
    keys_file: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Discover API keys for a provider from arguments, key files, or environment variables."""
    if explicit_keys:
        if isinstance(explicit_keys, str):
            parts = [k.strip() for k in re.split(r"[,;\s\n]+", explicit_keys) if k.strip()]
            if parts:
                return parts
        else:
            cleaned = [str(k).strip() for k in explicit_keys if k and str(k).strip()]
            if cleaned:
                return cleaned

    # Check key files
    paths_to_check: List[Path] = []
    if keys_file:
        paths_to_check.append(Path(keys_file))
    paths_to_check.extend([
        Path(f"data/artifacts/{provider}_keys.txt"),
        Path(f"{provider}_keys.txt"),
        Path(f".{provider}_keys"),
        Path("data/artifacts/nvidia_keys.txt"),
        Path("nvidia_keys.txt"),
    ])
    for p in paths_to_check:
        if p.exists() and p.is_file():
            try:
                lines = [
                    line.strip()
                    for line in p.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
                if lines:
                    logger.info("Loaded %d API key(s) for %s from %s", len(lines), provider, p)
                    return lines
            except Exception as exc:
                logger.warning("Failed reading keys from %s: %s", p, exc)

    # Check environment variables
    env_keys: List[str] = []
    # 1. Plural variable: e.g. NVIDIA_API_KEYS
    plural_var = f"{provider.upper()}_API_KEYS"
    val = os.environ.get(plural_var)
    if val:
        env_keys.extend([k.strip() for k in re.split(r"[,;\s\n]+", val) if k.strip()])

    # 2. Singular variable: e.g. NVIDIA_API_KEY
    sing_var = f"{provider.upper()}_API_KEY"
    val_sing = os.environ.get(sing_var)
    if val_sing:
        for k in re.split(r"[,;\s\n]+", val_sing):
            if k.strip() and k.strip() not in env_keys:
                env_keys.append(k.strip())

    # 3. Numbered variables: e.g. NVIDIA_API_KEY_1 .. NVIDIA_API_KEY_32
    for i in range(1, 33):
        k_val = os.environ.get(f"{provider.upper()}_API_KEY_{i}")
        if k_val and k_val.strip() and k_val.strip() not in env_keys:
            env_keys.append(k_val.strip())

    if env_keys:
        return env_keys

    # Check .env file
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, raw = line.split("=", 1)
            k, raw = k.strip(), raw.strip().strip("'\"")
            if not raw:
                continue
            if k == plural_var or k == sing_var:
                for item in re.split(r"[,;\s\n]+", raw):
                    if item.strip() and item.strip() not in env_keys:
                        env_keys.append(item.strip())
            elif k.startswith(f"{sing_var}_"):
                if raw not in env_keys:
                    env_keys.append(raw)
        if env_keys:
            return env_keys

    return []


class OpenAICompatibleClient:
    """``POST {base_url}/chat/completions`` in the OpenAI format (NVIDIA NIM, Mistral, Groq, OpenRouter, Cerebras).

    Supports multiple API keys with per-key rate limiting (KeyPoolRateLimiter), automatic key rotation,
    and individual key 429 backoff.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[Union[str, Sequence[str]]] = None,
        timeout: float = 90.0,
        json_mode: bool = True,
        rpm_per_key: float = 20.0,
        keys_file: Optional[Union[str, Path]] = None,
    ):
        if provider not in OPENAI_COMPATIBLE:
            raise ValueError(f"Unknown provider {provider!r}; expected one of {sorted(OPENAI_COMPATIBLE)} or 'gemini'.")
        self.provider = provider
        self.base_url, env = OPENAI_COMPATIBLE[provider]
        keys = load_api_keys(provider, explicit_keys=api_key, keys_file=keys_file)
        if not keys:
            raise RuntimeError(
                f"Set the {env} environment variable, provide {provider.upper()}_API_KEYS, "
                f"or save keys in data/artifacts/{provider}_keys.txt (one per line); keys are never stored in git."
            )
        self.model = model
        self.timeout = timeout
        self.json_mode = json_mode
        self.rpm_per_key = rpm_per_key
        self.pool = KeyPoolRateLimiter(keys, rpm_per_key=rpm_per_key)
        self.last_headers: Dict[str, str] = {}

    @property
    def num_keys(self) -> int:
        return len(self.pool)

    def _headers(self, key: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {key}"}

    def generate(self, prompt: str, temperature: float, seed: int) -> Tuple[str, Dict[str, Any]]:
        import httpx

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": prompt}],
            "temperature": temperature,
            "seed": seed,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}

        max_attempts = max(3, len(self.pool))
        last_exc: Optional[Exception] = None

        for attempt in range(max_attempts):
            key = self.pool.acquire_key()
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=body,
                    timeout=self.timeout,
                    headers=self._headers(key),
                )
                self.last_headers = rate_limit_headers(resp.headers)

                if resp.status_code in (429, 503):
                    wait = float(resp.headers.get("retry-after", 15))
                    self.pool.release_key(key, rate_limited_wait=wait)
                    last_exc = RateLimited(wait)
                    if self.pool.has_ready_key() and attempt < max_attempts - 1:
                        continue
                    raise last_exc

                if resp.status_code in (401, 403):
                    self.pool.release_key(key, disabled=True)
                    if self.pool.has_ready_key() and attempt < max_attempts - 1:
                        logger.warning("Key returned %d; retrying with next key in pool.", resp.status_code)
                        continue
                    resp.raise_for_status()

                # If provider rejects response_format, fallback to prompt-guided JSON
                if resp.status_code == 400 and self.json_mode and "response_format" in resp.text:
                    logger.warning("Provider rejected response_format; falling back to prompt-guided JSON.")
                    self.json_mode = False
                    body.pop("response_format", None)
                    self.pool.release_key(key)
                    continue

                resp.raise_for_status()
                self.pool.release_key(key)

                data = resp.json()
                try:
                    text = data["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError) as exc:
                    raise RatingParseError(f"no text in the reply: {str(data)[:200]}") from exc

                return _strip_code_fence(text), {
                    "model_version": data.get("model", self.model),
                    "usage": data.get("usage", {}),
                }

            except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                self.pool.release_key(key, rate_limited_wait=5.0)
                last_exc = net_err
                if attempt < max_attempts - 1:
                    continue
                raise

        if last_exc:
            raise last_exc
        raise RuntimeError("Failed to generate response after exhausting available keys.")

    def list_models(self) -> List[str]:
        import httpx

        key = self.pool.acquire_key()
        try:
            resp = httpx.get(f"{self.base_url}/models", timeout=self.timeout, headers=self._headers(key))
            resp.raise_for_status()
            return sorted(m["id"] for m in resp.json().get("data", []))
        finally:
            self.pool.release_key(key)


class NvidiaClient(OpenAICompatibleClient):
    """Convenience subclass for NVIDIA NIM with multiple key support and 20 RPM per key."""

    def __init__(
        self,
        model: str = "meta/llama-3.3-70b-instruct",
        api_keys: Optional[Union[str, Sequence[str]]] = None,
        rpm_per_key: float = 20.0,
        timeout: float = 90.0,
        keys_file: Optional[Union[str, Path]] = None,
    ):
        super().__init__(
            provider="nvidia",
            model=model,
            api_key=api_keys,
            timeout=timeout,
            rpm_per_key=rpm_per_key,
            keys_file=keys_file,
        )


def _strip_code_fence(text: str) -> str:
    """Some models wrap JSON in a ```json fence even in JSON mode; the content inside is what gets parsed."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()


def make_client(
    provider: str = "nvidia",
    model: Optional[str] = None,
    api_keys: Optional[Union[str, Sequence[str]]] = None,
    rpm_per_key: float = 20.0,
    keys_file: Optional[Union[str, Path]] = None,
) -> LLMClient:
    """The simulator client for ``configs/default.yaml`` ``simulation.provider`` / ``simulation.model``."""
    if provider == "gemini":
        key = api_keys[0] if isinstance(api_keys, (list, tuple)) and api_keys else (api_keys if isinstance(api_keys, str) else None)
        return GeminiClient(model or "gemini-2.5-flash-lite", api_key=key)
    m = model or ("meta/llama-3.3-70b-instruct" if provider == "nvidia" else "mistral-large-latest")
    return OpenAICompatibleClient(provider, m, api_key=api_keys, rpm_per_key=rpm_per_key, keys_file=keys_file)


@dataclass
class RatingCache:
    """Ratings keyed by (persona, phase, paper, seed, model, prompt version), persisted as JSONL (append-only).
    Thread-safe across multiple concurrent workers.
    """

    path: Optional[Path] = None
    _ratings: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.path is not None and Path(self.path).exists():
            for line in Path(self.path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._ratings[self.key(row["persona"], row["phase"], row["paper_id"], row["seed"], row["model"],
                                           row["prompt_version"])] = int(row["rating"])

    @staticmethod
    def key(persona: str, phase: str, paper_id: str, seed: int, model: str, prompt_version: str) -> Tuple[str, ...]:
        return (persona, phase, paper_id, str(seed), model, prompt_version)

    def get(self, *key_parts) -> Optional[int]:
        with self._lock:
            return self._ratings.get(self.key(*key_parts))

    def put_many(self, rows: Sequence[Mapping[str, Any]]) -> None:
        with self._lock:
            for row in rows:
                self._ratings[self.key(row["persona"], row["phase"], row["paper_id"], row["seed"], row["model"],
                                       row["prompt_version"])] = int(row["rating"])
            if self.path is not None and rows:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")

    def __len__(self) -> int:
        with self._lock:
            return len(self._ratings)


class RatingFailed(RuntimeError):
    """No valid reply after the allowed retries: the episode is marked failed (never imputed)."""


@dataclass
class Rater:
    client: LLMClient
    cache: RatingCache
    temperature: float = 0.7
    max_retries: int = 3
    max_rate_limit_waits: int = 20
    max_workers: int = 1
    sleep: Callable[[float], None] = time.sleep
    calls: int = 0  # LLM requests sent, retries included (cache hits send nothing)
    _call_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def rate(self, persona_id: str, persona_text: str, phase: str, goal_text: str,
             papers: Sequence[PaperForRating], seed: int) -> Dict[str, int]:
        """Rating 1-5 of every paper for this persona, goal phase and seed; cached ratings are reused."""
        model = self.client.model
        out: Dict[str, int] = {}
        todo: List[PaperForRating] = []
        for p in papers:
            cached = self.cache.get(persona_id, phase, p.paper_id, seed, model, PROMPT_VERSION)
            if cached is None:
                todo.append(p)
            else:
                out[p.paper_id] = cached

        batches = [todo[i : i + MAX_BATCH] for i in range(0, len(todo), MAX_BATCH)]

        def _process_batch(batch: Sequence[PaperForRating]) -> Dict[str, int]:
            ratings, meta, phash = self._call(persona_text, goal_text, batch, seed)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.cache.put_many([
                {"persona": persona_id, "phase": phase, "paper_id": p.paper_id, "seed": seed, "model": model,
                 "prompt_version": PROMPT_VERSION, "rating": ratings[p.paper_id], "prompt_sha": phash,
                 "model_version": meta.get("model_version"), "rated_at": stamp}
                for p in batch
            ])
            return ratings

        if batches:
            if self.max_workers > 1 and len(batches) > 1:
                with ThreadPoolExecutor(max_workers=min(len(batches), self.max_workers)) as executor:
                    futures = [executor.submit(_process_batch, b) for b in batches]
                    for fut in as_completed(futures):
                        out.update(fut.result())
            else:
                for b in batches:
                    out.update(_process_batch(b))

        return {p.paper_id: out[p.paper_id] for p in papers}

    def _call(self, persona_text: str, goal_text: str, batch: Sequence[PaperForRating], seed: int):
        prompt = build_prompt(persona_text, goal_text, batch)
        ids = [p.paper_id for p in batch]
        malformed, waits = 0, 0
        while True:
            try:
                with self._call_lock:
                    self.calls += 1
                text, meta = self.client.generate(prompt, self.temperature, seed)
                return parse_ratings(text, ids), meta, prompt_hash(prompt)
            except RateLimited as exc:
                waits += 1
                if waits > self.max_rate_limit_waits:
                    raise RatingFailed(f"still rate limited after {waits - 1} waits") from exc
                logger.info("Rate limited; waiting %.0fs.", exc.wait_seconds)
                self.sleep(exc.wait_seconds)
            except RatingParseError as exc:
                malformed += 1
                if malformed > self.max_retries:
                    raise RatingFailed(f"malformed reply after {self.max_retries} retries: {exc}") from exc
                logger.warning("Malformed reply (%s); retry %d of %d.", exc, malformed, self.max_retries)
                self.sleep(2.0 ** malformed)
