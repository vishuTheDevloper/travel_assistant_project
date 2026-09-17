"""Read the four saved travel pages and prepare documents for RAG.

Run from the project folder: python src/document_loader.py
This step reads local files; it does not call Gemini or create embeddings.
"""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def clean_text(value):
    """Remove HTML markup from a text field and tidy its whitespace."""
    fragment = BeautifulSoup(str(value or ""), "html.parser")
    for unwanted in fragment.select("script, style, noscript"):
        unwanted.decompose()
    return " ".join(fragment.get_text(" ", strip=True).split())


def embedded_text(value, heading_level=2):
    """Read meaningful text recursively from Visit Singapore's page data."""
    parts = []
    if isinstance(value, list):
        for item in value:
            parts.extend(embedded_text(item, heading_level))
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                parts.extend(embedded_text(item, heading_level))
                continue

            name = key.lower()
            # The site's translated text fields end in _t. Ignore image
            # descriptions, button labels, style settings and tracking data.
            relevant = name.endswith("_t") and any(
                word in name for word in ("title", "header", "description", "text")
            )
            if not relevant or any(word in name for word in ("alt", "cta", "button")):
                continue
            text = clean_text(item)
            if text:
                if "title" in name or "header" in name:
                    text = "#" * heading_level + " " + text
                parts.append(text)
    return parts


def itinerary_text(data):
    """Keep each attraction connected to its day and time of day."""
    parts = []
    day = clean_text(data.get("header_t"))
    if day:
        parts.append("## " + day)
    description = clean_text(data.get("description_t"))
    if description:
        parts.append(description)

    labels = {
        pill.get("pillCategory"): clean_text(pill.get("pillLabel_t"))
        for pill in data.get("pills", [])
    }
    previous_category = None
    for tile in data.get("tiles", []):
        category = tile.get("tilePillCategory")
        if category != previous_category:
            label = labels.get(category) or clean_text(category)
            if label:
                parts.append("### " + label)
            previous_category = category
        parts.extend(embedded_text(tile, heading_level=4))
    return parts


def extract_visit_singapore(soup):
    """Read visible headings and the text stored in aem-data attributes."""
    root = soup.find("main") or soup.body
    if root is None:
        raise ValueError("The saved Visit Singapore page has no main content.")

    for unwanted in root.select(
        "script, style, nav, header, footer, noscript, "
        "stb-main-navigation, stb-footer-container, stb-breadcrumbs, "
        "stb-newsletter, stb-social-media, stb-explore-more, stb-title-with-links"
    ):
        unwanted.decompose()

    parts = []
    for element in root.find_all(True):
        if element.has_attr("aem-data"):
            try:
                data = json.loads(element["aem-data"])
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Cannot read embedded content in {element.name}."
                ) from error
            if element.name == "stb-things-to-do" and isinstance(data, dict):
                parts.extend(itinerary_text(data))
            else:
                parts.extend(embedded_text(data))
        elif element.name in ("h1", "h2", "h3", "h4"):
            text = clean_text(element)
            if text:
                parts.append("#" * int(element.name[1]) + " " + text)
    return "\n\n".join(parts)


def extract_wikivoyage(soup):
    """Read the article body, excluding navigation and editing controls."""
    root = (
        soup.select_one("#mw-content-text .mw-parser-output")
        or soup.select_one("#mw-content-text")
        or soup.select_one(".mw-parser-output")
    )
    if root is None:
        raise ValueError("The saved Wikivoyage page has no article body.")

    for unwanted in root.select(
        "script, style, nav, noscript, .mw-editsection, .toc, "
        ".noprint, .navbox, .metadata, .infobox, .sistersitebox"
    ):
        unwanted.decompose()

    parts = []
    for element in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        # A parent list item already contains the text of its nested children.
        if element.find_parent(["li", "p"]) is not None:
            continue
        text = " ".join(element.get_text(" ", strip=True).split())
        if not text:
            continue
        if element.name.startswith("h"):
            text = "#" * int(element.name[1]) + " " + text
        elif element.name == "li":
            text = "- " + text
        parts.append(text)
    return "\n\n".join(parts)


# Source ingestion
# Read the four saved Singapore pages and extract their travel content.
# Retain each title, URL and source identifier so later answers can cite the original page.
def load_documents(project_root=PROJECT_ROOT):
    """Load sources.json and return LangChain Documents with source metadata."""
    project_root = Path(project_root)
    manifest_path = project_root / "data" / "sources.json"
    sources = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(sources, list) or len(sources) < 3:
        raise ValueError("sources.json must list at least three public resources.")

    documents = []
    seen_ids, seen_urls = set(), set()
    for source in sources:
        source_id = source["source_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
            raise ValueError(f"Invalid source_id: {source_id}")
        if source_id in seen_ids or source["url"] in seen_urls:
            raise ValueError(f"Duplicate source entry: {source_id}")
        seen_ids.add(source_id)
        seen_urls.add(source["url"])

        file_path = project_root / source["local_path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing saved page: {file_path}")
        raw_bytes = file_path.read_bytes()
        soup = BeautifulSoup(raw_bytes, "html.parser")
        hostname = urlparse(source["url"]).hostname
        if hostname == "en.wikivoyage.org":
            content = extract_wikivoyage(soup)
        elif hostname in ("www.visitsingapore.com", "visitsingapore.com"):
            content = extract_visit_singapore(soup)
        else:
            raise ValueError(f"Add an extraction rule for this source: {hostname}")

        # A basic empty-page check; topic coverage is reviewed separately.
        if len(content.strip()) < 200:
            raise ValueError(f"Too little travel text extracted from {source_id}.")

        metadata = {
            "source_id": source_id,
            "title": source["title"],
            "source": source["url"],
            "url": source["url"],
            "local_path": source["local_path"],
            "accessed_on": source["accessed_on"],
            "destination": "Singapore",
            "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def main():
    # Read and validate every source before writing the processed documents.
    documents = load_documents()
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for document in documents:
        source_id = document.metadata["source_id"]
        review_text = (
            f"# {document.metadata['title']}\n\n"
            f"Source: {document.metadata['source']}\n"
            f"Accessed: {document.metadata['accessed_on']}\n\n"
            f"{document.page_content}\n"
        )
        (output_dir / f"{source_id}.md").write_text(review_text, encoding="utf-8")
        records.append({
            "page_content": document.page_content,
            "metadata": document.metadata,
        })
        print(f"{source_id}: {len(document.page_content):,} characters extracted")

    (output_dir / "documents.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nPrepared {len(documents)} source documents.")
    print("Readable copies and documents.json saved in data/processed.")


if __name__ == "__main__":
    main()
