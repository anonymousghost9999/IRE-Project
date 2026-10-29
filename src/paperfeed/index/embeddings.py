"""SPECTER2 Paper & Query Embeddings Module (Phase 2, Step 2.1 & 2.2).

Governed by docs/prob.md §6D and docs/design.md §5, §6.1:
- Base model: allenai/specter2_base
- Paper adapter: allenai/specter2 (proximity)
- Query adapter: allenai/specter2_adhoc_query (short text / search / seed topics)
- Embedding dimension: 768-d, L2-normalized
- Input format: title + [SEP] + abstract (or title-only if abstract is missing)
- Max token length: 512
- Hardware acceleration: fp16 on CUDA if available
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

from paperfeed.common.logger import get_logger

logger = get_logger("index.embeddings")


def clean_id(uri_or_id: str) -> str:
    """Normalize OpenAlex URI to bare ID."""
    return uri_or_id.replace("https://openalex.org/", "").strip()


class SPECTER2Encoder:
    """
    Encoder for scientific papers and search queries using SPECTER2 with task adapters.
    """

    def __init__(
        self,
        base_model_name: str = "allenai/specter2_base",
        dense_adapter_name: str = "allenai/specter2",
        query_adapter_name: str = "allenai/specter2_adhoc_query",
        device: Optional[str] = None,
        use_fp16: bool = True,
    ) -> None:
        self.base_model_name = base_model_name
        self.dense_adapter_name = dense_adapter_name
        self.query_adapter_name = query_adapter_name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_fp16 = use_fp16 and (self.device.type == "cuda")
        self._tokenizer = None
        self._model = None
        self._active_adapter = None

    def _ensure_loaded(self) -> None:
        """Lazy loader for tokenizer and model."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel

        logger.info(f"Loading SPECTER2 tokenizer and base model: {self.base_model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self._model = AutoAdapterModel.from_pretrained(
            self.base_model_name,
            torch_dtype=dtype,
        )

        logger.info(f"Loading proximity adapter: {self.dense_adapter_name}")
        self._model.load_adapter(
            self.dense_adapter_name,
            source="hf",
            load_as="proximity",
            set_active=True,
        )

        logger.info(f"Loading adhoc_query adapter: {self.query_adapter_name}")
        self._model.load_adapter(
            self.query_adapter_name,
            source="hf",
            load_as="adhoc_query",
            set_active=False,
        )

        self._model.to(device=self.device, dtype=dtype)
        self._model.eval()
        self._active_adapter = "proximity"
        self._model.set_active_adapters("proximity")
        logger.info(f"SPECTER2 loaded on {self.device} (fp16={self.use_fp16})")

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    def set_adapter(self, adapter_name: str) -> None:
        """Set active adapter ('proximity' or 'adhoc_query')."""
        self._ensure_loaded()
        if adapter_name not in ("proximity", "adhoc_query"):
            raise ValueError(f"Unknown adapter '{adapter_name}'. Must be 'proximity' or 'adhoc_query'.")
        if self._active_adapter != adapter_name:
            self._model.set_active_adapters(adapter_name)
            self._active_adapter = adapter_name

    def format_paper_text(self, title: str, abstract: Optional[str] = None) -> str:
        """Format paper as title + [SEP] + abstract (or title-only)."""
        sep = self._tokenizer.sep_token if getattr(self, "_tokenizer", None) is not None else "[SEP]"
        title = (title or "").strip()
        abstract = (abstract or "").strip()
        if abstract:
            return f"{title} {sep} {abstract}"
        return title

    def embed_texts(
        self,
        texts: List[str],
        adapter_name: str = "proximity",
        batch_size: int = 32,
        max_length: int = 512,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of pre-formatted texts with the chosen adapter.
        Returns L2-normalized float32 numpy array of shape (N, 768).
        """
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        self._ensure_loaded()
        self.set_adapter(adapter_name)

        # Length-sorted batching keeps padding small; the original order is restored at the end.
        order = np.argsort([len(t) for t in texts], kind="stable")
        sorted_texts = [texts[i] for i in order]

        embeddings_list = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"Embedding ({adapter_name})", total=(len(texts) + batch_size - 1) // batch_size)

        with torch.no_grad():
            for i in iterator:
                batch_texts = sorted_texts[i : i + batch_size]
                inputs = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self._model(**inputs)
                # Take CLS token embedding (index 0)
                cls_rep = outputs.last_hidden_state[:, 0, :]
                # L2 normalize
                cls_norm = torch.nn.functional.normalize(cls_rep, p=2, dim=1)
                embeddings_list.append(cls_norm.cpu().to(torch.float32).numpy())

        sorted_embeddings = np.vstack(embeddings_list)
        embeddings = np.empty_like(sorted_embeddings)
        embeddings[order] = sorted_embeddings
        return embeddings

    def embed_papers(
        self,
        papers: List[Dict[str, Any]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed candidate or authored papers using title + abstract and 'proximity' adapter.
        """
        texts = [self.format_paper_text(p.get("title", ""), p.get("abstract")) for p in papers]
        return self.embed_texts(texts, adapter_name="proximity", batch_size=batch_size, show_progress=show_progress)

    def embed_queries(
        self,
        queries: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Embed search queries or short facet seeds using 'adhoc_query' adapter.
        """
        return self.embed_texts(queries, adapter_name="adhoc_query", batch_size=batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query into a 1D vector (768,)."""
        vectors = self.embed_queries([query], batch_size=1)
        return vectors[0]
