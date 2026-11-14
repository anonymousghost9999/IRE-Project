"""Alternative dense encoders for the V1 encoder study (Extended; docs/design.md §13.8 "V1 specification", plan 10.1a).

Every encoder exposes the interface the pipeline already uses for SPECTER2 (``index.embeddings.SPECTER2Encoder``):

    embed_papers(papers, batch_size, show_progress) -> (N, d) float32, L2-normalised
    embed_queries(queries, batch_size)              -> (N, d)
    embed_query(query)                              -> (d,)

The text formats are fixed by design §13.8 V1 point 2 (each model's documented usage, 512 tokens):

    encoder  document text                pooling      text query
    scincl   title [SEP] abstract         [CLS]        the plain query
    bge      title. abstract              [CLS]        "Represent this sentence for searching relevant passages: " + query
    qwen3    title. abstract              last token   "Instruct: Given a research question, retrieve scientific papers
                                                        that answer it\\nQuery: " + query

Paper-to-paper uses (CSFCube, persona centroids, feed mode) only ever embed documents. All outputs are L2-normalised,
because the pipeline compares vectors by cosine / inner product, which is how each of these models is meant to be used.
Qwen3-Embedding-0.6B is released after T and is for the retrieval benchmarks only (never a Track 1 decision).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from paperfeed.common.logger import get_logger

logger = get_logger("index.encoders")


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    model_name: str
    pooling: str  # "cls", "mean" or "last_token"
    doc_separator: str  # between title and abstract; "[SEP]" means the tokenizer's own sep token
    query_prefix: str = ""
    track1_eligible: bool = True  # False: released after T, retrieval benchmarks only (design §13.8 V1 point 1)
    max_length: int = 512


ENCODER_SPECS: Dict[str, EncoderSpec] = {
    "scincl": EncoderSpec("scincl", "malteos/scincl", "cls", "[SEP]"),
    "bge": EncoderSpec("bge", "BAAI/bge-base-en-v1.5", "cls", ". ",
                       query_prefix="Represent this sentence for searching relevant passages: "),
    # Phase 11 comparison only (Scholar Inbox-style classifier, design §13.9 Phase 11 spec point 4); 2023, base usage:
    # mean pooling over "title. abstract"
    "gte": EncoderSpec("gte", "thenlper/gte-large", "mean", ". "),
    "qwen3": EncoderSpec("qwen3", "Qwen/Qwen3-Embedding-0.6B", "last_token", ". ",
                         query_prefix="Instruct: Given a research question, retrieve scientific papers that answer it\nQuery: ",
                         track1_eligible=False),
}

ENCODER_KEYS = ("specter2",) + tuple(ENCODER_SPECS)


def format_document(spec: EncoderSpec, title: Optional[str], abstract: Optional[str], sep_token: str = "[SEP]") -> str:
    """The document text for ``spec``; title only when there is no abstract."""
    title, abstract = (title or "").strip(), (abstract or "").strip()
    if not abstract:
        return title
    if spec.doc_separator == "[SEP]":
        return f"{title} {sep_token} {abstract}"
    return f"{title.rstrip('.')}{spec.doc_separator}{abstract}" if title else abstract


def format_query(spec: EncoderSpec, query: str) -> str:
    return f"{spec.query_prefix}{(query or '').strip()}"


def pool(hidden, attention_mask, how: str):
    """[CLS] (first position), mean over real tokens, or last-token pooling; the mask marks real tokens, so it works
    with left or right padding."""
    import torch

    if how == "cls":
        return hidden[:, 0]
    if how == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    if how != "last_token":
        raise ValueError(f"Unknown pooling {how!r}.")
    if bool((attention_mask[:, -1] == 1).all()):  # left padding: the last position is always a real token
        return hidden[:, -1]
    last = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class HFEncoder:
    """A Hugging Face encoder with the pooling and text formats of its ``EncoderSpec``. Loaded lazily."""

    def __init__(self, spec: EncoderSpec, device: Optional[str] = None, use_fp16: bool = True,
                 model: Any = None, tokenizer: Any = None) -> None:
        self.spec = spec
        self.name = spec.key
        self.model_name = spec.model_name
        self._device = device
        self.use_fp16 = use_fp16
        self._model, self._tokenizer = model, tokenizer  # injectable for tests

    def _ensure_loaded(self) -> None:
        import torch

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, padding_side="left" if self.spec.pooling == "last_token" else "right"
            )
        if self._model is None:
            from transformers import AutoModel

            dtype = torch.float16 if (self.use_fp16 and self._device == "cuda") else torch.float32
            self._model = AutoModel.from_pretrained(self.model_name, torch_dtype=dtype)
            logger.info("Loaded %s on %s (%s).", self.model_name, self._device, dtype)
        self._model.to(self._device)
        self._model.eval()

    @property
    def sep_token(self) -> str:
        tok = self._tokenizer
        return getattr(tok, "sep_token", None) or "[SEP]"

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        import torch

        self._ensure_loaded()
        if not texts:
            return np.empty((0, int(self._model.config.hidden_size)), dtype=np.float32)
        order = np.argsort([len(t) for t in texts], kind="stable")  # length-sorted batches, order restored below
        sorted_texts = [texts[i] for i in order]
        chunks: List[np.ndarray] = []
        starts = range(0, len(sorted_texts), batch_size)
        if show_progress:
            from tqdm import tqdm

            starts = tqdm(starts, desc=f"Embedding ({self.name})", total=(len(sorted_texts) + batch_size - 1) // batch_size)
        with torch.no_grad():
            for i in starts:
                chunks.append(self._embed_batch(list(sorted_texts[i : i + batch_size])))
        stacked = np.vstack(chunks)
        result = np.empty_like(stacked)
        result[order] = stacked
        return result

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """One batch; on CUDA out-of-memory it is split in halves (down to single texts), so a large model on a small
        GPU slows down instead of failing. Splitting changes the vectors only by floating-point noise (padding is
        masked)."""
        import torch

        batch = out = None
        try:
            batch = self._tokenizer(texts, padding=True, truncation=True, max_length=self.spec.max_length,
                                    return_tensors="pt").to(self._device)
            out = self._model(**batch)
            vec = pool(out.last_hidden_state, batch["attention_mask"], self.spec.pooling)
            return torch.nn.functional.normalize(vec.float(), p=2, dim=1).cpu().numpy().astype(np.float32)
        except torch.OutOfMemoryError:
            if len(texts) == 1:
                raise
            batch = out = None  # release the failed batch before retrying
            torch.cuda.empty_cache()
            half = len(texts) // 2
            logger.warning("%s: CUDA out of memory on %d texts; splitting the batch.", self.name, len(texts))
            return np.vstack([self._embed_batch(texts[:half]), self._embed_batch(texts[half:])])

    def embed_papers(self, papers: Sequence[Mapping[str, Any]], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        self._ensure_loaded()
        texts = [format_document(self.spec, p.get("title"), p.get("abstract"), self.sep_token) for p in papers]
        return self.embed_texts(texts, batch_size, show_progress)

    def embed_queries(self, queries: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self.embed_texts([format_query(self.spec, q) for q in queries], batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_queries([query], batch_size=1)[0]


def make_encoder(key: str, device: Optional[str] = None, use_fp16: bool = True, track1: bool = False):
    """The encoder for ``key`` (one of ``ENCODER_KEYS``). ``track1=True`` refuses encoders released after T."""
    if key == "specter2":
        from paperfeed.index.embeddings import SPECTER2Encoder

        return SPECTER2Encoder(device=device, use_fp16=use_fp16)
    if key not in ENCODER_SPECS:
        raise ValueError(f"Unknown encoder {key!r}; expected one of {ENCODER_KEYS}.")
    spec = ENCODER_SPECS[key]
    if track1 and not spec.track1_eligible:
        raise ValueError(f"{spec.model_name} was released after T: retrieval benchmarks only (design §13.8 V1 point 1).")
    return HFEncoder(spec, device=device, use_fp16=use_fp16)
