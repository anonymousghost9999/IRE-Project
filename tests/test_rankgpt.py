"""RankGPT-style sliding-window reranking (plan 11d) with a scripted model: no download, no GPU."""

import re

from paperfeed.eval.rankgpt import (
    RerankStats,
    candidate_text,
    parse_permutation,
    sliding_window_rerank,
    window_messages,
)


def test_candidate_text_and_prompt():
    assert candidate_text("Dense retrieval", "We study it.") == "Dense retrieval. We study it."
    assert candidate_text("Title only", None) == "Title only"
    long = candidate_text("T", "word " * 200)
    assert long.endswith(" ...") and len(long) < 320
    msgs = window_messages("graph neural networks", ["a", "b", "c"])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "[1] a\n[2] b\n[3] c" in msgs[1]["content"] and "3 papers" in msgs[1]["content"]


def test_parse_permutation_follows_rankgpt():
    assert parse_permutation("[3] > [1] > [2]", 3) == ([2, 0, 1], True)
    assert parse_permutation("[2] > [2] > [9] > [1]", 3) == ([1, 0, 2], True)  # duplicate and out-of-range dropped
    assert parse_permutation("[3]", 4) == ([2, 0, 1, 3], True)  # the rest keep their input order
    assert parse_permutation("I cannot rank these.", 3) == ([0, 1, 2], False)


def _model_by_score(score):
    """A scripted model that ranks a window by ``score(text)``, highest first."""
    def generate(messages):
        texts = re.findall(r"^\[(\d+)\] (.*)$", messages[1]["content"], flags=re.M)
        ranked = sorted(texts, key=lambda t: -score(t[1]))
        return " > ".join(f"[{i}]" for i, _ in ranked)
    return generate


def test_sliding_window_bubbles_relevant_papers_to_the_top():
    # 50 candidates; the relevant ones (score 1) start at the bottom of the input list
    items = [(f"P{i}", f"paper {i} {'relevant' if i >= 45 else 'other'}") for i in range(50)]
    stats = RerankStats()
    out = sliding_window_rerank("q", items, _model_by_score(lambda t: 1 if "relevant" in t else 0), stats=stats)
    assert sorted(out) == sorted(p for p, _ in items) and len(out) == 50  # a permutation of the pool
    assert set(out[:5]) == {f"P{i}" for i in range(45, 50)}  # window 20 / step 10 carries them to the top
    assert stats.windows == 4 and stats.unparsable == 0  # windows [30,50) [20,40) [10,30) [0,20)


def test_unparsable_windows_keep_their_order_and_are_counted():
    items = [(f"P{i}", f"t{i}") for i in range(25)]
    stats = RerankStats()
    assert sliding_window_rerank("q", items, lambda m: "no idea", stats=stats) == [p for p, _ in items]
    assert stats.windows == 2 and stats.unparsable == 2 and stats.failures == ["no idea", "no idea"]
    assert sliding_window_rerank("q", items[:1], lambda m: "x") == ["P0"]  # one candidate: nothing to rank
