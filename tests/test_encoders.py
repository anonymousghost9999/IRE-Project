"""V1 encoder interface (plan 10.1a): text formats, pooling and normalisation with tiny stand-in models."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from paperfeed.index.encoders import (
    ENCODER_KEYS,
    ENCODER_SPECS,
    HFEncoder,
    format_document,
    format_query,
    make_encoder,
    pool,
)


def test_document_formats_follow_the_specification():
    scincl, bge, qwen = ENCODER_SPECS["scincl"], ENCODER_SPECS["bge"], ENCODER_SPECS["qwen3"]
    assert format_document(scincl, "Dense retrieval", "We study it.", sep_token="[SEP]") == "Dense retrieval [SEP] We study it."
    assert format_document(bge, "Dense retrieval", "We study it.") == "Dense retrieval. We study it."
    assert format_document(bge, "Dense retrieval.", "We study it.") == "Dense retrieval. We study it."  # no ".."
    assert format_document(qwen, " T ", None) == "T"  # title only without an abstract
    assert format_document(bge, "", "Only abstract") == "Only abstract"


def test_query_formats_follow_the_specification():
    assert format_query(ENCODER_SPECS["scincl"], " graph neural nets ") == "graph neural nets"
    assert format_query(ENCODER_SPECS["bge"], "q").startswith("Represent this sentence for searching relevant passages: ")
    assert format_query(ENCODER_SPECS["qwen3"], "q") == (
        "Instruct: Given a research question, retrieve scientific papers that answer it\nQuery: q"
    )


def test_pooling_cls_and_last_token_with_either_padding_side():
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    right = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    assert torch.equal(pool(hidden, right, "cls"), hidden[:, 0])
    assert torch.equal(pool(hidden, right, "last_token"), torch.stack([hidden[0, 3], hidden[1, 1]]))
    left = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
    assert torch.equal(pool(hidden, left, "last_token"), hidden[:, -1])
    with pytest.raises(ValueError):
        pool(hidden, right, "mean")


class _Tok:
    """Whitespace tokenizer stand-in: token id = word length; pads on the right."""

    sep_token = "[SEP]"

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        ids = [[len(w) for w in t.split()][:max_length] or [0] for t in texts]
        width = max(len(i) for i in ids)
        input_ids = torch.tensor([i + [0] * (width - len(i)) for i in ids])
        mask = torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i in ids])

        class Batch(dict):
            def to(self, device):
                return self

        return Batch(input_ids=input_ids, attention_mask=mask)


class _Model(torch.nn.Module):
    """hidden state of a token = (id, 1, 0); records the inputs it saw."""

    config = type("C", (), {"hidden_size": 3})()

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, input_ids, attention_mask):
        self.seen.append(input_ids.tolist())
        h = torch.stack([input_ids.float(), torch.ones_like(input_ids, dtype=torch.float32),
                         torch.zeros_like(input_ids, dtype=torch.float32)], dim=-1)
        return type("O", (), {"last_hidden_state": h})()


def test_hf_encoder_formats_pools_normalises_and_keeps_order():
    model = _Model()
    enc = HFEncoder(ENCODER_SPECS["scincl"], device="cpu", use_fp16=False, model=model, tokenizer=_Tok())
    papers = [{"title": "aaaa bb", "abstract": "c"}, {"title": "a", "abstract": None}]
    vecs = enc.embed_papers(papers, batch_size=1)
    assert vecs.shape == (2, 3) and vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-6)
    # CLS = first token: "aaaa" (id 4) for paper 0, "a" (id 1) for paper 1, in the input order despite length sorting
    np.testing.assert_allclose(vecs[0], np.array([4, 1, 0]) / np.sqrt(17), rtol=1e-6)
    np.testing.assert_allclose(vecs[1], np.array([1, 1, 0]) / np.sqrt(2), rtol=1e-6)
    rows = [row for batch in model.seen for row in batch]
    assert [4, 2, 5, 1] in rows  # "aaaa bb [SEP] c": title, the tokenizer's sep token, abstract
    assert [1] in rows  # title only
    assert enc.embed_query("x").shape == (3,)


def test_make_encoder_keeps_post_t_models_out_of_track1():
    assert ENCODER_KEYS == ("specter2", "scincl", "bge", "qwen3")
    assert make_encoder("bge", device="cpu").model_name == "BAAI/bge-base-en-v1.5"  # lazy: nothing downloaded
    assert make_encoder("qwen3", device="cpu").spec.track1_eligible is False
    with pytest.raises(ValueError, match="after T"):
        make_encoder("qwen3", device="cpu", track1=True)
    make_encoder("scincl", device="cpu", track1=True)
    with pytest.raises(ValueError, match="Unknown encoder"):
        make_encoder("e5")
