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

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

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


# OpenAI-format chat APIs (one client for all of them). Keys are read from the named environment variable only.
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


class OpenAICompatibleClient:
    """``POST {base_url}/chat/completions`` in the OpenAI format (NVIDIA NIM, Mistral, Groq, OpenRouter, Cerebras)."""

    def __init__(self, provider: str, model: str, api_key: Optional[str] = None, timeout: float = 90.0,
                 json_mode: bool = True):
        if provider not in OPENAI_COMPATIBLE:
            raise ValueError(f"Unknown provider {provider!r}; expected one of {sorted(OPENAI_COMPATIBLE)} or 'gemini'.")
        self.provider = provider
        self.base_url, env = OPENAI_COMPATIBLE[provider]
        key = api_key or os.environ.get(env)
        if not key:
            raise RuntimeError(f"Set the {env} environment variable; the key is never stored.")
        self.model = model
        self._key = key
        self.timeout = timeout
        self.json_mode = json_mode
        self.last_headers: Dict[str, str] = {}

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._key}"}

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
        resp = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout, headers=self._headers())
        self.last_headers = rate_limit_headers(resp.headers)
        if resp.status_code in (429, 503):
            raise RateLimited(float(resp.headers.get("retry-after", 30)))
        resp.raise_for_status()
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RatingParseError(f"no text in the reply: {str(data)[:200]}") from exc
        return _strip_code_fence(text), {"model_version": data.get("model", self.model), "usage": data.get("usage", {})}

    def list_models(self) -> List[str]:
        import httpx

        resp = httpx.get(f"{self.base_url}/models", timeout=self.timeout, headers=self._headers())
        resp.raise_for_status()
        return sorted(m["id"] for m in resp.json().get("data", []))


def _strip_code_fence(text: str) -> str:
    """Some models wrap JSON in a ```json fence even in JSON mode; the content inside is what gets parsed."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()


def make_client(provider: str, model: str):
    """The simulator client for ``configs/default.yaml`` ``simulation.provider`` / ``simulation.model``."""
    return GeminiClient(model) if provider == "gemini" else OpenAICompatibleClient(provider, model)


@dataclass
class RatingCache:
    """Ratings keyed by (persona, phase, paper, seed, model, prompt version), persisted as JSONL (append-only)."""

    path: Optional[Path] = None
    _ratings: Dict[Tuple[str, ...], int] = field(default_factory=dict)

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
        return self._ratings.get(self.key(*key_parts))

    def put_many(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            self._ratings[self.key(row["persona"], row["phase"], row["paper_id"], row["seed"], row["model"],
                                   row["prompt_version"])] = int(row["rating"])
        if self.path is not None and rows:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

    def __len__(self) -> int:
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
    sleep: Callable[[float], None] = time.sleep
    calls: int = 0  # LLM requests sent, retries included (cache hits send nothing)

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
        for i in range(0, len(todo), MAX_BATCH):
            batch = todo[i : i + MAX_BATCH]
            ratings, meta, phash = self._call(persona_text, goal_text, batch, seed)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.cache.put_many([
                {"persona": persona_id, "phase": phase, "paper_id": p.paper_id, "seed": seed, "model": model,
                 "prompt_version": PROMPT_VERSION, "rating": ratings[p.paper_id], "prompt_sha": phash,
                 "model_version": meta.get("model_version"), "rated_at": stamp}
                for p in batch
            ])
            out.update(ratings)
        return {p.paper_id: out[p.paper_id] for p in papers}

    def _call(self, persona_text: str, goal_text: str, batch: Sequence[PaperForRating], seed: int):
        prompt = build_prompt(persona_text, goal_text, batch)
        ids = [p.paper_id for p in batch]
        malformed, waits = 0, 0
        while True:
            try:
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
