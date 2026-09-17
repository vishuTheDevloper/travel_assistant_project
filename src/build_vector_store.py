"""Build a persistent Chroma database for the Singapore knowledge base.

Run from the project folder: python src/build_vector_store.py
This uses LangChain and the shared model from embeddings.py. No Gemini call
or paid database service is used. Chroma saves the database automatically.

Identical inputs reuse the same collection and can resume an interrupted build.
Changed chunks or model settings use a new collection. Previous collections
are retained; index_manifest.json identifies the current, verified collection.
Future retrieval code should call load_vector_store() from this module.
"""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Support both direct execution and imports such as src.build_vector_store.
if __package__:
    from .embeddings import check_token_lengths, check_vector, embedding_configuration, get_embeddings
else:
    from embeddings import check_token_lengths, check_vector, embedding_configuration, get_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_DIRECTORY = PROJECT_ROOT / "data" / "vector_store"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
MANIFEST_PATH = DATABASE_DIRECTORY / "index_manifest.json"
INDEX_SCHEMA_VERSION = 1
BATCH_SIZE = 32
DISTANCE_METRIC = "cosine"


def read_chunks() -> tuple[list[Document], str]:
    """Read text and citation metadata, retaining the deterministic chunk IDs."""
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError("Run python src/chunk_documents.py first.")
    raw = CHUNKS_PATH.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list) or not records:
        raise ValueError("chunks.json must contain a non-empty list.")

    documents = []
    seen_ids = set()
    required = ["chunk_id", "source_id", "title", "url", "section_path"]
    for number, record in enumerate(records, start=1):
        text = record.get("page_content")
        metadata = record.get("metadata", {})
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk {number} has no usable text.")
        if any(not isinstance(metadata.get(key), str) or not metadata[key].strip() for key in required):
            raise ValueError(f"Chunk {number} is missing its ID or citation metadata.")
        if any(not isinstance(value, (str, int, float, bool)) for value in metadata.values()):
            raise ValueError(f"Chunk {number} contains unsupported nested metadata.")
        if metadata["chunk_id"] in seen_ids:
            raise ValueError(f"Duplicate chunk ID: {metadata['chunk_id']}")
        seen_ids.add(metadata["chunk_id"])
        documents.append(Document(page_content=text, metadata=metadata))

    return documents, hashlib.sha256(raw).hexdigest()


def index_identity(chunks_sha256: str) -> dict:
    """Keep different document snapshots and embedding configurations apart."""
    identity = {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "chunks_sha256": chunks_sha256,
        "embedding_configuration": embedding_configuration(),
        "distance_metric": DISTANCE_METRIC,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        **identity,
        "index_fingerprint": fingerprint,
        "collection_name": f"singapore_travel_{fingerprint[:24]}",
    }


def check_collection_settings(collection, identity: dict) -> None:
    """Check the stored configuration before inserting or retrieving vectors."""
    if (collection.metadata or {}).get("index_fingerprint") != identity["index_fingerprint"]:
        raise ValueError("Stored collection settings do not match this index.")
    if collection.configuration.get("hnsw", {}).get("space") != DISTANCE_METRIC:
        raise ValueError("The collection must use cosine distance.")


def verify_stored_chunks(collection, documents: list[Document]) -> set[str]:
    """Verify all stored text, metadata, and vectors, including partial builds."""
    expected = {doc.metadata["chunk_id"]: doc for doc in documents}
    stored = collection.get(include=["documents", "metadatas", "embeddings"])
    stored_ids = set(stored["ids"])
    if not stored_ids.issubset(expected):
        raise ValueError("This collection contains chunks from different inputs.")
    if not stored_ids:
        return set()

    for chunk_id, text, metadata, vector in zip(
        stored["ids"], stored["documents"], stored["metadatas"], stored["embeddings"], strict=True
    ):
        doc = expected[chunk_id]
        if text != doc.page_content or metadata != doc.metadata:
            raise ValueError(f"Stored text or citation metadata differs for {chunk_id}.")
        check_vector(vector)
    return stored_ids


# Persistent vector index
# Chroma stores chunk text, metadata and embeddings for reuse between sessions.
# Check the saved manifest before searching to reject missing or incompatible indexes.
def load_vector_store() -> Chroma:
    """Open the completed index for retrieval using the same embedding model.

    Refuse a missing, incomplete, or outdated index instead of silently creating
    an empty collection or searching with incompatible embeddings.
    """
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Run python src/build_vector_store.py first.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    documents, chunks_sha256 = read_chunks()
    identity = index_identity(chunks_sha256)
    if manifest.get("status") != "ready" or any(manifest.get(k) != v for k, v in identity.items()):
        raise ValueError("The index is outdated. Run python src/build_vector_store.py again.")
    if not (DATABASE_DIRECTORY / "chroma.sqlite3").is_file():
        raise FileNotFoundError("The saved database is missing. Run python src/build_vector_store.py.")

    client = chromadb.PersistentClient(path=str(DATABASE_DIRECTORY))
    collection = client.get_collection(identity["collection_name"], embedding_function=None)
    check_collection_settings(collection, identity)
    if collection.count() != len(documents) or manifest.get("total_chunks") != len(documents):
        raise ValueError("The stored chunk count is incorrect. Run python src/build_vector_store.py again.")
    return Chroma(
        client=client,
        collection_name=identity["collection_name"],
        embedding_function=get_embeddings(),
        create_collection_if_not_exists=False,
    )


def main() -> None:
    documents, chunks_sha256 = read_chunks()
    identity = index_identity(chunks_sha256)
    source_counts = Counter(doc.metadata["source_id"] for doc in documents)
    print(f"Source documents: {len(source_counts)}", flush=True)
    print(f"Chunks to store: {len(documents)}", flush=True)
    print("Checking all chunks against the embedding model's token limit...", flush=True)
    check_token_lengths([doc.page_content for doc in documents])

    DATABASE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DATABASE_DIRECTORY))
    collection = client.get_or_create_collection(
        name=identity["collection_name"],
        embedding_function=None,
        metadata={"index_fingerprint": identity["index_fingerprint"]},
        configuration={"hnsw": {"space": DISTANCE_METRIC}},
    )
    check_collection_settings(collection, identity)
    existing_ids = verify_stored_chunks(collection, documents)
    missing = [doc for doc in documents if doc.metadata["chunk_id"] not in existing_ids]

    if missing:
        print(f"Creating {len(missing)} embeddings on CPU...", flush=True)
        store = Chroma(
            client=client,
            collection_name=identity["collection_name"],
            embedding_function=get_embeddings(),
            create_collection_if_not_exists=False,
        )
        for start in range(0, len(missing), BATCH_SIZE):
            batch = missing[start : start + BATCH_SIZE]
            store.add_documents(batch, ids=[doc.metadata["chunk_id"] for doc in batch])
            completed = len(existing_ids) + min(start + BATCH_SIZE, len(missing))
            print(f"Stored {completed}/{len(documents)} chunks", flush=True)
    else:
        print("All chunks are already stored. No duplicate entries added.", flush=True)

    stored_ids = verify_stored_chunks(collection, documents)
    if len(stored_ids) != len(documents) or collection.count() != len(documents):
        raise ValueError("Some chunks were not stored. Run this script again to resume.")

    manifest = {
        **identity,
        "status": "ready",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_documents": len(source_counts),
        "total_chunks": len(documents),
        "chunks_per_source": dict(source_counts),
        "text_metadata_and_vectors_verified": True,
        "package_versions": {
            name: version(name)
            for name in ["chromadb", "langchain-chroma", "langchain-huggingface", "sentence-transformers", "torch"]
        },
    }
    # Publish the current index only after every chunk has been verified.
    temporary_path = MANIFEST_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(MANIFEST_PATH)

    print(f"\nStored chunks verified: {len(stored_ids)}")
    print("Text, source details, and embeddings verified.")
    print("Vector database saved in data/vector_store/")
    print("Saved data/vector_store/index_manifest.json")
    print("Next step: test semantic retrieval.")


if __name__ == "__main__":
    main()
