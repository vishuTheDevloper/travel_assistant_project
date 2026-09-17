"""Retrieve and rerank Singapore passages while preserving their original sources.

Run from the project folder: python src/retriever.py
This demonstrates retrieval only. It does not call Gemini or generate answers.
The questions cover the PDF's six knowledge-base topics, with separate indoor
and outdoor examples, plus an unsupported question to inspect failure behaviour.

Use up to 24 Chroma semantic matches and 24 TF-IDF keyword matches, remove
duplicate chunk IDs, then rank their relevance with a small local CrossEncoder.
The existing BGE embedding model and vector database remain compatible.

Day headings are expanded for ranking only (Day 1 -> Day 1 (first day)). This
expresses the same ordinal information; returned source text is never changed.
No question-specific answers, destinations, or chunk IDs are used for ranking.

Scores are not confidence percentages or proof that an answer is supported.
The answering step must still reject irrelevant evidence and unsupported facts.
"""

import json
import math
import re
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

if __package__:
    from .build_vector_store import MANIFEST_PATH, load_vector_store, read_chunks
else:
    from build_vector_store import MANIFEST_PATH, load_vector_store, read_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_K = 4
CANDIDATES_PER_METHOD = 24
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
RERANK_TOKEN_LIMIT = 512
RETRIEVAL_VERSION = "3"
# Candidate selection only: this does not prove that an entire venue is indoors.
# The answer must cite the exact excerpt describing the chosen activity.
INDOOR_DESCRIPTOR = re.compile(r"\bindoors?\b|\bsheltered\b|\bair[\s-]+conditioned\b", re.I)
DAY_ORDINALS = (
    "", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth", "twenty-first", "twenty-second", "twenty-third",
    "twenty-fourth", "twenty-fifth", "twenty-sixth", "twenty-seventh",
    "twenty-eighth", "twenty-ninth", "thirtieth", "thirty-first",
)
TEST_CASES = [
    {
        "topic": "Major attractions and neighbourhoods",
        "question": "Which neighbourhoods in Singapore can I explore for culture and sightseeing?",
        "review_for": "Named Singapore neighbourhoods and descriptions of their sights or cultural experiences.",
    },
    {
        "topic": "Local transportation",
        "question": "How can I get around Singapore using the MRT and buses?",
        "review_for": "Useful information about Singapore's MRT or bus services.",
    },
    {
        "topic": "Cultural and practical travel tips",
        "question": "What should visitors know about dress and shoes when visiting temples in Singapore?",
        "review_for": "Visitor etiquette concerning clothing or removing shoes at places of worship.",
    },
    {
        "topic": "Food and local experiences",
        "question": "Which local dishes should I try at Singapore hawker centres?",
        "review_for": "Named local dishes and descriptions. Hawker-centre locations alone are insufficient.",
    },
    {
        "topic": "Sample itineraries",
        "question": "What could I do in the morning and evening on my first day in Singapore?",
        "review_for": "An itinerary with identifiable activities and their day or time-of-day context.",
    },
    {
        "topic": "Indoor activity suggestions",
        "question": "What indoor attractions in Singapore can I visit on a rainy day?",
        "review_for": "Activities explicitly described as indoors or sheltered; do not assume a whole attraction is indoors.",
    },
    {
        "topic": "Outdoor activity suggestions",
        "question": "Where can I enjoy nature walks or cycling outdoors in Singapore?",
        "review_for": "Named parks, nature areas, walks, or cycling experiences.",
    },
    {
        "topic": "Unsupported destination question",
        "question": "Which ski resorts should I visit in Switzerland?",
        "review_for": "This Singapore knowledge base does not support Switzerland ski recommendations. Returned neighbours must not be treated as an answer.",
    },
]


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """A local question-passage scorer; this does not replace the BGE embeddings."""
    return CrossEncoder(
        RERANK_MODEL,
        revision=RERANK_REVISION,
        device="cpu",
        max_length=RERANK_TOKEN_LIMIT,
        trust_remote_code=False,
        activation_fn=torch.nn.Identity(),
    )


@lru_cache(maxsize=1)
def get_keyword_index(chunks_sha256: str):
    """Cache a lightweight keyword index for this exact document snapshot."""
    documents, actual_sha256 = read_chunks()
    if actual_sha256 != chunks_sha256:
        raise ValueError("Chunks changed during retrieval. Rebuild the vector store.")
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    )
    matrix = vectorizer.fit_transform([doc.page_content for doc in documents])
    return documents, vectorizer, matrix


def ranking_text(document) -> str:
    """Express numbered days in words without modifying the returned evidence."""
    section = document.metadata["section_path"]

    def expand_day(match):
        day = int(match.group(1))
        return f"Day {day} ({DAY_ORDINALS[day]} day)" if 0 < day < len(DAY_ORDINALS) else match.group(0)

    expanded = re.sub(r"\bDay (\d+)\b", expand_day, section)
    return document.page_content.replace(f"Section: {section}", f"Section: {expanded}", 1)


# Hybrid retrieval
# Combine vector similarity with keyword matches, then rerank the candidate passages.
# These local searches select evidence for Gemini without spending a Gemini request.
def retrieve_passages(question: str, k: int = DEFAULT_K, *, indoor_evidence_only: bool = False) -> list[dict]:
    """Combine semantic and keyword candidates, then rerank the best k passages.

    No fixed relevance threshold is assumed. A returned passage is a candidate
    for answering, not proof that the question is answerable. Current weather
    and currency conversions must use our MCP tools in the completed assistant.

    For an indoor planning facet, also scan the small local corpus for explicit
    indoor descriptors before reranking. This prevents a short venue description
    from disappearing behind broad museum/itinerary matches. Source text and
    metadata are unchanged; descriptors are candidates, not verified claims.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a non-empty search question.")
    if type(k) is not int or not 1 <= k <= 10:
        raise ValueError("k must be an integer between 1 and 10.")
    if type(indoor_evidence_only) is not bool:
        raise ValueError("indoor_evidence_only must be a boolean.")

    question = question.strip()
    # The loader validates the index configuration and reuses our shared BGE model.
    store = load_vector_store()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    documents, vectorizer, matrix = get_keyword_index(manifest["chunks_sha256"])
    dense_matches = store.similarity_search_with_score(question, k=CANDIDATES_PER_METHOD)
    candidates = {}
    for document, distance in dense_matches:
        if not math.isfinite(float(distance)):
            raise ValueError("The database returned a non-finite distance.")
        candidates[document.metadata["chunk_id"]] = {
            "document": document,
            "cosine_distance": float(distance),
            "keyword_similarity": 0.0,
            "candidate_methods": ["semantic"],
        }

    keyword_scores = (matrix @ vectorizer.transform([question]).T).toarray().ravel()
    for i in keyword_scores.argsort()[::-1][:CANDIDATES_PER_METHOD]:
        if keyword_scores[i] <= 0:
            continue
        document = documents[i]
        candidate = candidates.setdefault(document.metadata["chunk_id"], {
            "document": document,
            "cosine_distance": None,
            "keyword_similarity": 0.0,
            "candidate_methods": [],
        })
        candidate["keyword_similarity"] = float(keyword_scores[i])
        candidate["candidate_methods"].append("keyword")

    if indoor_evidence_only:
        eligible = {}
        for document in documents:
            # Ignore the generated heading prefix: support must occur in the body.
            body = document.page_content.split("\n\n", 1)[-1]
            if not INDOOR_DESCRIPTOR.search(body):
                continue
            chunk_id = document.metadata["chunk_id"]
            candidate = candidates.get(chunk_id, {
                "document": document,
                "cosine_distance": None,
                "keyword_similarity": 0.0,
                "candidate_methods": [],
            })
            candidate["candidate_methods"].append("indoor_descriptor")
            eligible[chunk_id] = candidate
        candidates = eligible

    if not candidates:
        return []
    items = list(candidates.values())
    reranker = get_reranker()
    pairs = [(question, ranking_text(item["document"])) for item in items]
    tokenized = reranker.tokenizer(
        [pair[0] for pair in pairs],
        text_pair=[pair[1] for pair in pairs],
        add_special_tokens=True, truncation=False, padding=False,
    )
    if any(len(ids) > RERANK_TOKEN_LIMIT for ids in tokenized["input_ids"]):
        raise ValueError("The question and a passage exceed the reranker's input limit. Use a shorter search question; no text was truncated.")
    scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
    for item, score in zip(items, scores, strict=True):
        if not math.isfinite(float(score)):
            raise ValueError("The reranker returned a non-finite score.")
        item["rerank_score"] = float(score)
    items.sort(key=lambda item: (-item["rerank_score"], item["document"].metadata["chunk_id"]))

    passages = []
    seen_sections = set()
    for item in items:
        document = item["document"]
        metadata = document.metadata
        section_key = (metadata["source_id"], metadata["section_path"])
        if indoor_evidence_only and section_key in seen_sections:
            continue
        seen_sections.add(section_key)
        required = ["chunk_id", "source_id", "title", "url", "section_path"]
        if any(not isinstance(metadata.get(key), str) or not metadata[key].strip() for key in required):
            raise ValueError("A retrieved passage is missing its source information.")
        if not document.page_content.strip():
            raise ValueError("The database returned empty source text.")
        passages.append({
            "rank": len(passages) + 1,
            "page_content": document.page_content,
            "metadata": dict(metadata),
            "cosine_distance": item["cosine_distance"],
            "keyword_similarity": item["keyword_similarity"],
            "rerank_score": item["rerank_score"],
            "candidate_methods": item["candidate_methods"],
        })
        if len(passages) == k:
            break
    return passages


def write_review_report(cases: list[dict]) -> None:
    """Save complete evidence for human review instead of claiming accuracy."""
    output_directory = PROJECT_ROOT / "data" / "processed"
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "index_fingerprint": manifest["index_fingerprint"],
        "retrieval_method": "semantic_and_keyword_search_with_cross_encoder_reranking",
        "retrieval_version": RETRIEVAL_VERSION,
        "candidates_per_method": CANDIDATES_PER_METHOD,
        "reranking_model": RERANK_MODEL,
        "reranking_revision": RERANK_REVISION,
        "ranking_text_preprocessing": "Numbered itinerary days expanded to ordinal words; source text retained unchanged.",
        "k": DEFAULT_K,
        "score_interpretation": "Rerank score: higher is more relevant to the question, not a confidence percentage or proof of answerability. Cosine distance can be null for keyword-only candidates.",
        "relevance_review_status": "pending_human_review",
        "generated_answers": False,
        "package_versions": {
            name: version(name) for name in ["sentence-transformers", "transformers", "scikit-learn", "torch", "langchain-chroma"]
        },
        "test_cases": cases,
    }
    # Keep the first semantic-only report for a fair before/after review.
    results_path = output_directory / "retrieval_results.json"
    baseline_path = output_directory / "retrieval_results_baseline.json"
    if results_path.is_file() and not baseline_path.exists():
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        if previous.get("retrieval_method") == "cosine_similarity_search":
            shutil.copyfile(results_path, baseline_path)
    results_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Semantic Retrieval Review",
        "",
        "These are retrieved source passages, not generated answers.",
        "Up to 24 semantic matches and 24 keyword matches are combined, deduplicated, and reranked to select four passages.",
        "Rerank scores are higher for stronger matches, but are not accuracy or confidence percentages.",
        "Scores alone do not prove that a passage supports the requested answer.",
        "For ranking, Day 1 is also expressed as first day, and similarly for other numbered days. Original source text below is unchanged.",
        "The semantic-only report, when present, is preserved in retrieval_results_baseline.json.",
        "",
        "The seven Singapore questions cover six PDF topics: indoor and outdoor activities are tested separately.",
        "The final question demonstrates that an unsupported question can still retrieve passages.",
        "",
        "Review the actual text against each question. Do not mark a case successful just because results exist.",
        "Live weather and exchange rates will be handled by the custom MCP tools.",
        "",
    ]
    for number, case in enumerate(cases, start=1):
        lines.extend([
            f"## {number}. {case['topic']}",
            "",
            f"**Question:** {case['question']}",
            "",
            f"**What to check:** {case['review_for']}",
            "",
            "**Review:** Pending. Mark useful / partly useful / unsupported, and explain why.",
            "",
        ])
        if not case["results"]:
            lines.extend(["No passages returned.", ""])
        for passage in case["results"]:
            metadata = passage["metadata"]
            lines.extend([
                f"### Passage {passage['rank']}",
                "",
                f"**Source:** [{metadata['title']}]({metadata['url']})",
                "",
                f"**Section:** {metadata['section_path']}",
                "",
                f"**Rerank score:** {passage['rerank_score']:.4f}",
                "",
                f"**Found by:** {', '.join(passage['candidate_methods'])}",
                "",
                f"**Chunk ID:** `{metadata['chunk_id']}`",
                "",
                *["> " + line for line in passage["page_content"].splitlines()],
                "",
            ])
    (output_directory / "retrieval_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("Opening the saved database with the shared embedding model...", flush=True)
    print("The local reranking model downloads on first use. No API key is required.", flush=True)
    cases = []
    for number, case in enumerate(TEST_CASES, start=1):
        print(f"\n{number}. {case['topic']}", flush=True)
        print(f"Question: {case['question']}", flush=True)
        results = retrieve_passages(case["question"])
        cases.append({**case, "results": results})
        for passage in results:
            print(
                f"  {passage['rank']}. {passage['metadata']['source_id']} | "
                f"{passage['metadata']['section_path']} | rerank score {passage['rerank_score']:.4f}",
                flush=True,
            )
        if not results:
            print("  No passages returned.", flush=True)

    write_review_report(cases)
    print(f"\nRetrieval checks completed for {len(TEST_CASES)} questions.")
    print("Saved data/processed/retrieval_results.json")
    print("Saved data/processed/retrieval_review.md")
    print("Next step: review whether the retrieved passages answer each question.")


if __name__ == "__main__":
    main()
