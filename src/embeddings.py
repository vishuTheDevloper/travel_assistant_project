"""Shared local embeddings for indexing and searching the Singapore knowledge base.

Run from the project folder: python src/embeddings.py
This checks every saved chunk and creates two example vectors. It does not
build the vector database. Both indexing and retrieval must use get_embeddings().

Model documentation: https://huggingface.co/BAAI/bge-small-en-v1.5
The pinned revision keeps the downloaded model version consistent.
If the model or embedding settings change, rebuild the vector database.
"""

import hashlib
import json
import math
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MAX_INPUT_TOKENS = 512
EMBEDDING_DIMENSION = 384
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def embedding_configuration() -> dict:
    """Settings to record alongside the index and check when reopening it."""
    return {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "device": "cpu",
        "dimension": EMBEDDING_DIMENSION,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "normalize_embeddings": True,
        "document_instruction": "",
        "query_instruction": QUERY_INSTRUCTION,
        "newline_handling": "replace_with_space",
    }


@lru_cache(maxsize=1)
def get_tokenizer():
    """Use the tokenizer from exactly the same model revision."""
    return AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )


def check_token_lengths(texts: list[str], *, is_query: bool = False) -> list[int]:
    """Reject oversized input instead of allowing silent text truncation.

    Counts include the model's special tokens and, for questions, the
    model author's recommended retrieval instruction.
    """
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding input must contain non-empty text strings.")

    prefix = QUERY_INSTRUCTION if is_query else ""
    # LangChain replaces newlines with spaces before passing text to the model.
    prepared = [prefix + text.replace("\n", " ") for text in texts]
    encoded = get_tokenizer()(
        prepared,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    lengths = [len(ids) for ids in encoded["input_ids"]]
    oversized = [(i, n) for i, n in enumerate(lengths) if n > MAX_INPUT_TOKENS]
    if oversized:
        examples = ", ".join(f"item {i + 1}: {n} tokens" for i, n in oversized[:5])
        action = "Shorten the search question." if is_query else "Split the oversized chunks into smaller pieces."
        raise ValueError(
            f"Input exceeds the {MAX_INPUT_TOKENS}-token model limit ({examples}). "
            f"{action} No text was truncated."
        )
    return lengths


class TravelEmbeddings(HuggingFaceEmbeddings):
    """LangChain embeddings with a length check for documents and questions."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        check_token_lengths(texts)
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        check_token_lengths([text], is_query=True)
        return super().embed_query(text)


# Shared embedding configuration
# BGE converts text into normalized 384-dimensional vectors on the CPU.
# Indexing and search reuse this factory so document and question vectors are comparable.
@lru_cache(maxsize=1)
def get_embeddings() -> TravelEmbeddings:
    """Load once per process; reuse this factory for indexing and retrieval."""
    return TravelEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu",
            "revision": MODEL_REVISION,
            "trust_remote_code": False,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 16,
            "prompt": "",
        },
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": QUERY_INSTRUCTION,
        },
        show_progress=False,
    )


def check_vector(vector: list[float]) -> None:
    """Check the expected size, valid numbers, and unit-length normalization."""
    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(f"Expected {EMBEDDING_DIMENSION} numbers per embedding.")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("An embedding contains a non-finite number.")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isclose(norm, 1.0, abs_tol=1e-4):
        raise ValueError("The embedding is not normalized to unit length.")


def main() -> None:
    chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks.json"
    if not chunks_path.is_file():
        raise FileNotFoundError("Run python src/chunk_documents.py first.")

    chunks_bytes = chunks_path.read_bytes()
    records = json.loads(chunks_bytes)
    if not isinstance(records, list) or not records:
        raise ValueError("chunks.json must contain a non-empty list of chunks.")
    texts = [record["page_content"] for record in records]

    print(f"Embedding model: {MODEL_NAME}", flush=True)
    print("Device: CPU", flush=True)
    print("Checking chunk lengths. The tokenizer downloads on first use...", flush=True)
    lengths = check_token_lengths(texts)
    print(f"Chunks checked: {len(texts)}", flush=True)
    print(f"Largest chunk: {max(lengths)} tokens", flush=True)
    print(f"Model input limit: {MAX_INPUT_TOKENS} tokens", flush=True)
    print("Token-limit check passed. No chunks need truncation.", flush=True)

    print("Loading model. The first download can take a few minutes...", flush=True)
    embeddings = get_embeddings()
    document_vector = embeddings.embed_documents([texts[0]])[0]
    query_vector = embeddings.embed_query("Which indoor attractions can I visit in Singapore?")
    check_vector(document_vector)
    check_vector(query_vector)

    report = {
        "status": "passed",
        "embedding_configuration": embedding_configuration(),
        "chunks_sha256": hashlib.sha256(chunks_bytes).hexdigest(),
        "chunks_checked": len(texts),
        "smallest_chunk_tokens": min(lengths),
        "largest_chunk_tokens": max(lengths),
        "special_tokens_included": True,
        "all_chunks_fit_without_truncation": True,
        "sample_document_dimension": len(document_vector),
        "sample_query_dimension": len(query_vector),
        "sample_vectors_finite_and_normalized": True,
        "vector_database_built": False,
        "package_versions": {
            name: version(name)
            for name in ["torch", "sentence-transformers", "langchain-huggingface", "transformers", "langchain-chroma"]
        },
    }
    report_path = chunks_path.parent / "embedding_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Document embedding: {len(document_vector)} numbers")
    print(f"Question embedding: {len(query_vector)} numbers")
    print("Embedding checks passed.")
    print("Saved data/processed/embedding_check.json")
    print("Next step: build the vector database.")


if __name__ == "__main__":
    main()
