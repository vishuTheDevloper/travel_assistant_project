"""Split prepared travel documents into sections and smaller retrieval chunks.

Run from the project folder: python src/chunk_documents.py
Input: data/processed/documents.json
Outputs: chunks.json, chunks_preview.md, and chunking_summary.json.

The size settings below are CHARACTER counts, not embedding-model tokens.
The embedding step must also check its model's token limit before indexing.
"""

import hashlib
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_CHUNK_CHARACTERS = 1000
TARGET_OVERLAP_CHARACTERS = 150
CHUNKING_VERSION = "1"
HEADERS = [("#" * level, f"heading_{level}") for level in range(1, 7)]


# Section-aware chunking
# Split documents by heading, then create chunks of up to 1,000 characters.
# Target 150 characters of overlap within sections while retaining citation metadata.
def create_chunks(records):
    """Preserve source metadata and the full heading path on every chunk."""
    if not isinstance(records, list) or len(records) < 3:
        raise ValueError("Expected at least three source documents in documents.json.")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS,
        strip_headers=True,
    )
    chunks = []
    seen_sources = set()

    for record in records:
        source_metadata = record["metadata"].copy()
        source_id = source_metadata["source_id"]
        if source_id in seen_sources:
            raise ValueError(f"Duplicate source document: {source_id}")
        seen_sources.add(source_id)
        if not source_metadata.get("title") or not source_metadata.get("source"):
            raise ValueError(f"Missing citation details for {source_id}.")

        sections = header_splitter.split_text(record["page_content"])
        source_chunk_index = 0

        for section_index, section in enumerate(sections):
            if not section.page_content.strip():
                continue
            heading_values = [
                section.metadata[key]
                for _, key in HEADERS
                if section.metadata.get(key)
            ]
            section_path = " > ".join(heading_values) or "Overview"
            destination = source_metadata.get("destination", "Singapore")
            context = f"Destination: {destination}\nSection: {section_path}\n\n"

            # Reserve room for the heading context repeated on EVERY chunk.
            # Thus the complete page_content stays within the configured size.
            body_budget = MAX_CHUNK_CHARACTERS - len(context)
            if body_budget <= TARGET_OVERLAP_CHARACTERS:
                raise ValueError(f"Section heading is too long: {section_path}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=body_budget,
                chunk_overlap=TARGET_OVERLAP_CHARACTERS,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
                is_separator_regex=False,
            )
            pieces = text_splitter.split_documents([section])

            for piece_index, piece in enumerate(pieces):
                body = piece.page_content.strip()
                if not body:
                    continue
                content = context + body
                if len(content) > MAX_CHUNK_CHARACTERS:
                    raise ValueError(f"Chunk size limit exceeded for {source_id}.")

                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
                chunk_id = (
                    f"{source_id}-{section_index:03d}-{piece_index:03d}-{digest}"
                )
                metadata = {
                    **source_metadata,
                    **section.metadata,
                    "section_path": section_path,
                    "section_index": section_index,
                    "chunk_index": source_chunk_index,
                    "piece_index": piece_index,
                    "chunk_id": chunk_id,
                    "character_count": len(content),
                    "chunking_version": CHUNKING_VERSION,
                }
                chunks.append(Document(page_content=content, metadata=metadata))
                source_chunk_index += 1

        if source_chunk_index == 0:
            raise ValueError(f"No usable chunks were created for {source_id}.")

    if len({chunk.metadata["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs were generated.")
    return chunks


def save_preview(chunks, path):
    """Save up to three examples per source for a quick human review."""
    lines = [
        "# Travel knowledge-base chunk preview",
        "",
        "These are samples. The complete set is stored in chunks.json.",
        "",
    ]
    included = Counter()
    for chunk in chunks:
        metadata = chunk.metadata
        source_id = metadata["source_id"]
        if included[source_id] >= 3:
            continue
        included[source_id] += 1
        lines.extend([
            f"## {source_id} — example {included[source_id]}",
            "",
            f"- Source title: {metadata['title']}",
            f"- Source URL: {metadata['source']}",
            f"- Section: {metadata['section_path']}",
            f"- Characters: {metadata['character_count']}",
            f"- Chunk ID: {metadata['chunk_id']}",
            "",
            chunk.page_content,
            "",
            "---",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    output_dir = PROJECT_ROOT / "data" / "processed"
    input_path = output_dir / "documents.json"
    if not input_path.is_file():
        raise FileNotFoundError("Run python src/document_loader.py first.")
    input_bytes = input_path.read_bytes()
    records = json.loads(input_bytes.decode("utf-8-sig"))
    chunks = create_chunks(records)

    serialized = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]
    (output_dir / "chunks.json").write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_preview(chunks, output_dir / "chunks_preview.md")

    counts = Counter(chunk.metadata["source_id"] for chunk in chunks)
    lengths = [len(chunk.page_content) for chunk in chunks]
    summary = {
        "source_documents": len(records),
        "total_chunks": len(chunks),
        "chunks_per_source": dict(counts),
        "maximum_chunk_characters": MAX_CHUNK_CHARACTERS,
        "target_overlap_characters": TARGET_OVERLAP_CHARACTERS,
        "overlap_scope": "Within long sections only; actual overlap may be smaller.",
        "context_included_in_size_limit": True,
        "smallest_chunk_characters": min(lengths),
        "largest_chunk_characters": max(lengths),
        "input_documents_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "chunking_version": CHUNKING_VERSION,
        "langchain_text_splitters_version": version("langchain-text-splitters"),
        "embedding_token_limits_checked": False,
    }
    (output_dir / "chunking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for source_id, count in counts.items():
        print(f"{source_id}: {count} chunks")
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Largest chunk: {max(lengths)} characters")
    print("Source details and section headings retained on every chunk.")
    print("Saved chunks.json, chunks_preview.md and chunking_summary.json.")


if __name__ == "__main__":
    main()
