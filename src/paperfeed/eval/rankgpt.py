"""RankGPT-style zero-shot listwise reranking (plan 11d; docs/design.md §13.9 Phase 11 spec point 4).

The Core pool's 100 candidates are reranked by a small open LLM against the feed pseudo-query (§8.1). As in RankGPT
(Sun et al., 2023), a window of 20 candidates slides from the bottom of the list to the top with step 10; each window
is reordered by the model's answer "[3] > [1] > [2] ...". Parsing follows RankGPT: numbers are read in order,
duplicates and out-of-range numbers are dropped, and candidates the model left out keep their input order after the
ones it ranked. A window with no usable number keeps its input order and is counted as unparsable.

Model: Qwen/Qwen2.5-3B-Instruct in 4-bit (released September 2024, so it cannot encode citations made after T),
greedy decoding. If 4-bit loading is unavailable, Qwen/Qwen2.5-1.5B-Instruct in fp16 is used and reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from paperfeed.common.logger import get_logger

logger = get_logger("eval.rankgpt")

WINDOW, STEP = 20, 10
ABSTRACT_CHARS = 300
PRIMARY_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM = ("You are RankGPT, an intelligent assistant that can rank scientific papers based on their relevance to a "
          "researcher's interests.")


def candidate_text(title: str, abstract: Optional[str]) -> str:
    """Title and the first 300 characters of the abstract."""
    abstract = (abstract or "").strip()
    if len(abstract) > ABSTRACT_CHARS:
        abstract = abstract[:ABSTRACT_CHARS].rsplit(" ", 1)[0] + " ..."
    title = (title or "").strip()
    if not abstract:
        return title
    return f"{title.rstrip('.')}. {abstract}" if title else abstract


def window_messages(query: str, texts: Sequence[str]) -> List[Dict[str, str]]:
    """The RankGPT prompt for one window, as chat messages."""
    n = len(texts)
    passages = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts, start=1))
    user = (
        f"I will provide you with {n} papers, each indicated by a number identifier []. Rank them based on their "
        f"relevance to the researcher's interests: {query.strip()}\n\n{passages}\n\n"
        f"Rank the {n} papers above based on their relevance to the researcher's interests: {query.strip()}. "
        "All the papers should be included and listed using identifiers, in descending order of relevance. The output "
        "format should be [] > [], e.g., [2] > [1]. Only respond with the ranking results, do not say any word or explain."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_permutation(text: str, n: int) -> Tuple[List[int], bool]:
    """0-based order of a window from the model's answer; (input order, False) when no usable number is found."""
    seen: List[int] = []
    for m in re.findall(r"\d+", text or ""):
        i = int(m) - 1
        if 0 <= i < n and i not in seen:
            seen.append(i)
    if not seen:
        return list(range(n)), False
    return seen + [i for i in range(n) if i not in seen], True


@dataclass
class RerankStats:
    windows: int = 0
    unparsable: int = 0
    failures: List[str] = field(default_factory=list)


def sliding_window_rerank(query: str, items: Sequence[Tuple[str, str]], generate: Callable[[List[Dict[str, str]]], str],
                          window: int = WINDOW, step: int = STEP, stats: Optional[RerankStats] = None) -> List[str]:
    """``items``: (paper_id, text) in the input order (the Core pool). Returns the reranked paper IDs."""
    order = list(items)
    stats = stats if stats is not None else RerankStats()
    end = len(order)
    while True:
        start = max(0, end - window)
        chunk = order[start:end]
        if len(chunk) > 1:
            answer = generate(window_messages(query, [t for _, t in chunk]))
            perm, ok = parse_permutation(answer, len(chunk))
            stats.windows += 1
            if not ok:
                stats.unparsable += 1
                stats.failures.append((answer or "")[:80])
            order[start:end] = [chunk[i] for i in perm]
        if start == 0:
            break
        end -= step
    return [pid for pid, _ in order]


class LocalRanker:
    """Greedy generation with a local Hugging Face chat model (4-bit when possible)."""

    def __init__(self, model_name: str = PRIMARY_MODEL, four_bit: bool = True, max_new_tokens: int = 160):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        loaded = None
        if four_bit:
            import sys
            if sys.platform == "win32":
                logger.warning("4-bit loading on Windows is unsupported/unstable; using %s in fp16 (pre-stated fallback).",
                               FALLBACK_MODEL)
            else:
                try:
                    from transformers import BitsAndBytesConfig

                    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                               bnb_4bit_quant_type="nf4")
                    loaded = (model_name, AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant,
                                                                              device_map="auto"), "4-bit nf4")
                except Exception as exc:  # bitsandbytes missing or unsupported here: the pre-stated fallback
                    logger.warning("4-bit loading of %s failed (%s); using %s in fp16.", model_name, exc, FALLBACK_MODEL)
        if loaded is None:
            name = FALLBACK_MODEL if four_bit else model_name
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to("cuda")
            else:
                model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
            loaded = (name, model, str(dtype))
        self.model_name, self.model, self.precision = loaded
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
