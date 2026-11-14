"""Track 2 simulator client (plan 9a): prompt, strict parsing, cache, retries. No network: fake clients only."""

import json

import pytest

from paperfeed.sim.llm import (
    ABSTRACT_CHARS,
    PROMPT_VERSION,
    GeminiClient,
    PaperForRating,
    RateLimited,
    Rater,
    RatingCache,
    RatingFailed,
    RatingParseError,
    build_prompt,
    parse_ratings,
)


def _papers(n, prefix="W"):
    return [PaperForRating(f"{prefix}{i}", f"Title {i}", f"Abstract {i}") for i in range(n)]


def test_prompt_lists_every_paper_and_cuts_long_abstracts():
    long = PaperForRating("W9", "Long one", "word " * 1000)
    prompt = build_prompt("Wrote about retrieval.", "Find retrieval papers.", _papers(2) + [long])
    assert "Researcher: Wrote about retrieval." in prompt and "Current goal: Find retrieval papers." in prompt
    assert "[W0] Title 0" in prompt and "[W1] Title 1" in prompt and "[W9] Long one" in prompt
    assert prompt.count("word") < ABSTRACT_CHARS / 4 and " ..." in prompt
    assert '"ratings"' in prompt


@pytest.mark.parametrize("reply, problem", [
    ("not json", "not JSON"),
    ('{"scores": []}', "ratings"),
    ('{"ratings": [{"id": "W0", "rating": 4}]}', "missing"),
    ('{"ratings": [{"id": "W0", "rating": 4}, {"id": "W1", "rating": 2}, {"id": "W7", "rating": 2}]}', "extra"),
    ('{"ratings": [{"id": "W0", "rating": 4}, {"id": "W0", "rating": 3}]}', "twice"),
    ('{"ratings": [{"id": "W0", "rating": 4.5}, {"id": "W1", "rating": 2}]}', "integer"),
    ('{"ratings": [{"id": "W0", "rating": true}, {"id": "W1", "rating": 2}]}', "integer"),
    ('{"ratings": [{"id": "W0", "rating": 6}, {"id": "W1", "rating": 2}]}', "integer"),
])
def test_parsing_is_strict(reply, problem):
    with pytest.raises(RatingParseError, match=problem):
        parse_ratings(reply, ["W0", "W1"])


def test_parsing_accepts_bracketed_ids():
    assert parse_ratings('{"ratings": [{"id": "[W0]", "rating": 5}, {"id": "W1", "rating": 1}]}', ["W0", "W1"]) == {"W0": 5, "W1": 1}


class ScriptedClient:
    """Rates paper k as (k mod 5) + 1, or plays back scripted failures first."""

    model = "fake-gemini"

    def __init__(self, failures=()):
        self.failures = list(failures)
        self.prompts = []

    def generate(self, prompt, temperature, seed):
        self.prompts.append((prompt, temperature, seed))
        if self.failures:
            f = self.failures.pop(0)
            if f == "429":
                raise RateLimited(7)
            return f, {}
        ids = [line[1 : line.index("]")] for line in prompt.splitlines() if line.startswith("[")]
        return json.dumps({"ratings": [{"id": i, "rating": int(i[1:]) % 5 + 1} for i in ids]}), {"model_version": "fake-001"}


def _rater(client, tmp_path, **kw):
    return Rater(client, RatingCache(tmp_path / "cache.jsonl"), sleep=lambda s: None, **kw)


def test_rater_batches_caches_and_reuses_across_arms(tmp_path):
    client = ScriptedClient()
    rater = _rater(client, tmp_path)
    r = rater.rate("P1", "persona", "A", "goal A", _papers(25), seed=3)
    assert len(client.prompts) == 3  # 10 + 10 + 5
    assert r == {f"W{i}": i % 5 + 1 for i in range(25)} and list(r) == [f"W{i}" for i in range(25)]
    assert all(t == 0.7 and s == 3 for _, t, s in client.prompts)

    # another arm shows an overlapping set: only the new papers are sent
    r2 = rater.rate("P1", "persona", "A", "goal A", _papers(3) + [PaperForRating("W40", "T", "A")], seed=3)
    assert len(client.prompts) == 4 and "[W40]" in client.prompts[-1][0] and "[W0]" not in client.prompts[-1][0]
    assert r2["W0"] == 1 and r2["W40"] == 1

    # another goal phase or seed is a different rating
    rater.rate("P1", "persona", "B", "goal B", _papers(1), seed=3)
    rater.rate("P1", "persona", "A", "goal A", _papers(1), seed=4)
    assert len(client.prompts) == 6


def test_cache_persists_and_reloads(tmp_path):
    rater = _rater(ScriptedClient(), tmp_path)
    rater.rate("P1", "persona", "A", "goal", _papers(2), seed=1)
    rows = [json.loads(l) for l in (tmp_path / "cache.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["prompt_version"] == PROMPT_VERSION and rows[0]["model_version"] == "fake-001"
    assert rows[0]["prompt_sha"] and rows[0]["model"] == "fake-gemini"
    again = ScriptedClient()
    fresh = Rater(again, RatingCache(tmp_path / "cache.jsonl"), sleep=lambda s: None)
    assert fresh.rate("P1", "persona", "A", "goal", _papers(2), seed=1) == {"W0": 1, "W1": 2}
    assert again.prompts == []  # everything came from the released cache


def test_malformed_replies_are_retried_then_fail(tmp_path):
    client = ScriptedClient(failures=["oops", '{"ratings": []}'])
    rater = _rater(client, tmp_path)
    assert rater.rate("P1", "p", "A", "g", _papers(2), seed=0) == {"W0": 1, "W1": 2}
    assert rater.calls == 3

    bad = ScriptedClient(failures=["x"] * 10)
    with pytest.raises(RatingFailed, match="after 3 retries"):
        _rater(bad, tmp_path / "b").rate("P1", "p", "A", "g", _papers(2), seed=0)
    assert len(bad.prompts) == 4  # first try + 3 retries
    assert len(RatingCache(tmp_path / "b" / "cache.jsonl")) == 0  # nothing imputed


def test_rate_limits_wait_without_counting_as_malformed(tmp_path):
    waits = []
    client = ScriptedClient(failures=["429", "429", "oops", "429"])
    rater = Rater(client, RatingCache(), sleep=waits.append)
    assert rater.rate("P1", "p", "A", "g", _papers(1), seed=0) == {"W0": 1}
    assert waits.count(7) == 3  # rate-limit waits
    with pytest.raises(RatingFailed, match="rate limited"):
        Rater(ScriptedClient(failures=["429"] * 5), RatingCache(), sleep=lambda s: None,
              max_rate_limit_waits=2).rate("P1", "p", "A", "g", _papers(1), seed=0)


class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload or {}, headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_gemini_client_request_and_reply(monkeypatch):
    import httpx

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiClient("gemini-x")
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    seen = {}

    def fake_post(url, json, timeout, headers):
        seen.update(url=url, body=json, headers=headers)
        return _Resp(200, {"candidates": [{"content": {"parts": [{"text": '{"ratings": []}'}]}}],
                           "modelVersion": "gemini-x-001", "usageMetadata": {"totalTokenCount": 9}})

    monkeypatch.setattr(httpx, "post", fake_post)
    client = GeminiClient("gemini-x")
    text, meta = client.generate("prompt", 0.7, 5)
    assert text == '{"ratings": []}' and meta["model_version"] == "gemini-x-001"
    assert "models/gemini-x:generateContent" in seen["url"] and "secret" not in seen["url"]  # key in a header only
    assert seen["headers"]["x-goog-api-key"] == "secret"
    cfg = seen["body"]["generationConfig"]
    assert cfg == {"temperature": 0.7, "seed": 5, "responseMimeType": "application/json"}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp(429, headers={"retry-after": "12"}))
    with pytest.raises(RateLimited) as exc:
        client.generate("prompt", 0.7, 5)
    assert exc.value.wait_seconds == 12
