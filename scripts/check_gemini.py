"""Check the selected Gemini model using the shared project configuration.

python scripts/check_gemini.py --config-only  # No Gemini request.
python scripts/check_gemini.py --list-models  # Model metadata, no generation.
python scripts/check_gemini.py                # One connection-test request.
"""

import argparse
import sys
from pathlib import Path

from dotenv import dotenv_values
from google.genai.errors import APIError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Direct execution starts in scripts/, so add our project for the src import.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_config import create_gemini_model


def diagnostic_error(error: Exception) -> str:
    """Show Google's message and status, excluding request headers and the key."""
    current = error
    seen = set()
    detail = f"{type(error).__name__}: {error}"
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, APIError):
            detail = (
                f"Google HTTP {current.code} {current.status}: "
                f"{current.message}"
            )
            break
        current = current.__cause__ or current.__context__
    key = (
        dotenv_values(PROJECT_ROOT / ".env", encoding="utf-8-sig")
        .get("GOOGLE_API_KEY") or ""
    ).strip()
    if key:
        detail = detail.replace(key, "[REDACTED]")
    return detail[:4000]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--config-only", action="store_true")
    mode.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    model = None
    try:
        model = create_gemini_model(max_tokens=512, timeout=30)
        print("Configured Gemini model:", model.model)
        print("API key found in the project configuration.")
        print("Automatic retries: disabled")
        if args.config_only:
            print("Configuration check passed. No Gemini API request was made.")
            return 0
        if args.list_models:
            print("Models advertised by the API with generateContent support:")
            # Keep the LangChain model alive; its cleanup owns the SDK client.
            names = sorted(
                item.name for item in model.client.models.list()
                if item.name and "generateContent" in (item.supported_actions or [])
            )
            for name in names:
                print(name)
            print("Listing does not verify free quota, access to every feature, or generation success.")
            print("No generation request was made.")
            return 0
        print("Testing one plain-text request through LangChain (no tools or RAG)...")
        response = model.invoke("Reply with only these words: Gemini connection successful.")
        if not response.text.strip():
            raise RuntimeError("Gemini returned no answer text.")
        print("Gemini replied:")
        print(response.text)
        return 0
    except Exception as error:
        print("Connection check failed:", diagnostic_error(error))
        print("No automatic retry was made.")
        return 1
    finally:
        if model is not None:
            model.client.close()


if __name__ == "__main__":
    raise SystemExit(main())
