"""Check the repaired project without Gemini or external provider requests."""
import argparse
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_data():
    from src.document_loader import load_documents
    from src.chunk_documents import create_chunks
    sources = json.loads((ROOT / 'data/sources.json').read_text(encoding='utf-8-sig'))
    documents = load_documents(ROOT)
    records = [{'page_content': d.page_content, 'metadata': d.metadata} for d in documents]
    chunks = create_chunks(records)
    saved = json.loads((ROOT / 'data/processed/chunks.json').read_text(encoding='utf-8-sig'))
    regenerated = [{'page_content': d.page_content, 'metadata': d.metadata} for d in chunks]
    if regenerated != saved:
        raise ValueError('Raw source extraction/chunking differs from chunks.json; rebuild the index before using it.')
    manifest = json.loads((ROOT / 'data/vector_store/index_manifest.json').read_text(encoding='utf-8-sig'))
    checksum = hashlib.sha256((ROOT / 'data/processed/chunks.json').read_bytes()).hexdigest()
    if manifest.get('chunks_sha256') != checksum:
        raise ValueError('The index manifest does not match chunks.json.')
    print(f'Source resources: {len(sources)}. Reproduced chunks: {len(chunks)}. Manifest matches.', flush=True)


# Offline verification
# Reproduce the saved chunks and run regression, MCP integration and interface tests.
# Mocked model responses and recorded tool data keep these checks free of live API calls.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrieval', action='store_true', help='Also check real CPU retrieval using already cached models; no downloads.')
    args = parser.parse_args()
    from src.assistant import ASSISTANT_VERSION
    print('Assistant version:', ASSISTANT_VERSION, flush=True)
    print('Gemini requests: 0. External weather/currency requests: 0.', flush=True)
    check_data()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py', top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful(): return 1
    if args.retrieval:
        os.environ['HF_HUB_OFFLINE'] = '1'
        from src.build_vector_store import load_vector_store, read_chunks, verify_stored_chunks
        from src.assistant import retrieve_context, DEFAULT_QUESTION
        from src.conversation_memory import ConversationMemory
        store = load_vector_store()
        documents, _ = read_chunks()
        count = len(verify_stored_chunks(store._collection, documents))
        context = retrieve_context(DEFAULT_QUESTION, ConversationMemory())
        if not context['passages']: raise ValueError('No passages retrieved.')
        print(f'Verified {count} stored vectors and retrieved {len(context["passages"])} source passages.', flush=True)
    print(f'PASS: {result.testsRun} offline tests. These are not a fresh Gemini answer.', flush=True)
    return 0

if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as error:
        print(f'Check failed: {type(error).__name__}: {error}')
        raise SystemExit(1) from None
