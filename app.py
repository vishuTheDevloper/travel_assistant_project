"""Streamlit chat interface for the Singapore travel assistant.

Run from the project root, with its virtual environment active:
    python -m streamlit run app.py --server.fileWatcherType none

Opening the app, inspecting sources and downloading previous answers do not
call Gemini. Only a new, explicitly submitted chat message runs answer_turn.
Each browser session owns its memory; the app does not cache model answers,
weather or exchange rates as if they were current information.
"""

import asyncio
import inspect
import json
import sys
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant import answer_turn, safe_error, save_report
from src.conversation_memory import ConversationMemory


STYLES = """
<style>
    .block-container { max-width: 1040px; padding-top: 2rem; }
    .travel-hero {
        background: linear-gradient(120deg, #102e38, #145a57);
        border-radius: 22px; padding: 30px 32px; margin-bottom: 22px;
        color: #ffffff; box-shadow: 0 12px 30px rgba(14, 59, 63, .12);
    }
    .travel-eyebrow {
        font-size: .75rem; letter-spacing: .16em; font-weight: 700;
        color: #a4e6d8; margin-bottom: 12px;
    }
    .travel-hero h1 {
        color: #ffffff; font-size: clamp(1.8rem, 4vw, 2.7rem);
        line-height: 1.15; font-weight: 700; margin: 0 0 12px; padding: 0;
    }
    .travel-hero p { color: #d4ebe7; margin: 0; max-width: 610px; }
    .travel-tags { margin-top: 22px; display: flex; gap: 8px; flex-wrap: wrap; }
    .travel-tags span {
        border: 1px solid rgba(205, 239, 230, .3); border-radius: 30px;
        padding: 5px 12px; color: #e5f5f0; font-size: .8rem;
    }
    [data-testid="stChatMessage"] { border-radius: 16px; padding: 18px; }
    [data-testid="stChatMessage"] h3 { font-size: 1.1rem; padding-top: .8rem; }
    @media (max-width: 640px) {
        .travel-hero { padding: 24px 20px; border-radius: 16px; }
        .block-container { padding-top: 1rem; }
    }
</style>
"""


def initialize_session() -> None:
    defaults = {
        "travel_memory": ConversationMemory(),
        "travel_items": [],
        "travel_busy": False,
        "travel_attempts": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def new_chat() -> None:
    st.session_state.travel_memory.clear()
    st.session_state.travel_items = []
    st.session_state.pop("travel_pending", None)
    st.session_state.travel_busy = False
    # Keep the session-wide usage counter: starting over does not reset quota.


def enqueue_message() -> None:
    """Consume a submit event once; ordinary widget reruns never resubmit it."""
    question = st.session_state.get("travel_prompt", "")
    if isinstance(question, str) and question.strip() and not st.session_state.travel_busy:
        st.session_state.travel_pending = question.strip()
        st.session_state.travel_busy = True


def configuration_problems() -> list[str]:
    """Local file/config checks only; never instantiate a model here."""
    problems = []
    config = dotenv_values(PROJECT_ROOT / ".env", encoding="utf-8-sig")
    if not (config.get("GOOGLE_API_KEY") or "").strip() or not (config.get("GEMINI_MODEL") or "").strip():
        problems.append("Add GOOGLE_API_KEY and GEMINI_MODEL to the project's .env file.")
    if not (PROJECT_ROOT / "data" / "vector_store" / "index_manifest.json").is_file():
        problems.append("The local knowledge-base index is missing. Complete the vector-store setup first.")
    if any(not (PROJECT_ROOT / "mcp_server" / name).is_file() for name in ("server.py", "weather.py", "currency.py")):
        problems.append("The custom MCP server files are missing from the mcp_server folder.")
    return problems


def run_assistant(question: str, memory: ConversationMemory, progress):
    """Use a fresh subprocess-capable loop in Streamlit's script thread.

    Windows Streamlit/Tornado may select a Selector loop, which cannot start
    the MCP subprocess. Use Proactor for this turn without changing the global
    event-loop policy used by the Streamlit web server.
    """
    def loop_factory():
        loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

    try:
        with asyncio.Runner(loop_factory=loop_factory) as runner:
            return runner.run(answer_turn(question, memory, progress=progress))
    finally:
        asyncio.set_event_loop(None)


def progress_label(message: str) -> str | None:
    if message.startswith("Retrieving"):
        return "Finding relevant travel guidance…"
    if "correcting validation problems" in message:
        return "Checking and correcting the answer…"
    if message.startswith("Gemini request 1"):
        return "Reviewing your question and preferences…"
    if message.startswith("Gemini request 2"):
        return "Putting the answer together…"
    if "Using MCP tool: get_weather" in message:
        return "Checking the weather forecast…"
    if "Using MCP tool: convert_currency" in message:
        return "Checking the reference exchange rate…"
    return None


def render_item(item: dict, index: int) -> None:
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant", avatar="🧭"):
        report = item.get("report")
        if report is None:
            st.error(item["error"])
            st.caption("No automatic retry was made. You can edit your question and send it again when ready.")
            return
        if report["status"] == "partial":
            st.caption("Some information is still missing; the supported part is shown below.")
        st.markdown(report["answer_markdown"])
        if item.get("save_warning"):
            st.warning(item["save_warning"])
        with st.expander("Sources and tool activity"):
            st.caption("Passages below are retrieval candidates. Citations in the answer identify the passages actually used. Quote checks do not prove every interpretation is correct.")
            sources = {}
            for passage in report.get("retrieved_passages", []):
                meta = passage["metadata"]
                sources[meta["source_id"]] = {"Source": meta["title"], "Link": meta["url"]}
            if sources:
                st.dataframe(list(sources.values()), hide_index=True, width="stretch")
            records = report.get("tool_calls", [])
            if records:
                for record in records:
                    st.write(f"**{record['tool']} — {record['status']}**")
                    st.json({"inputs": record["arguments"], "result": record.get("result"), "error": record.get("error")}, expanded=False)
            else:
                st.caption("No live tool was called for this answer.")
            st.caption(f"Model requests for this answer: {report['model_calls']}. Previous conversation turns supplied: {report['previous_turn_count']}.")
        left, right = st.columns(2)
        with left:
            st.download_button(
                "Download answer", data=f"# Question\n\n{item['question']}\n\n{report['answer_markdown']}\n",
                file_name=f"singapore_answer_{index + 1}.md", mime="text/markdown",
                key=f"travel_md_{index}", on_click="ignore", disabled=st.session_state.travel_busy,
            )
        with right:
            st.download_button(
                "Download evidence JSON", data=json.dumps(report, ensure_ascii=False, indent=2),
                file_name=item.get("report_filename", f"singapore_evidence_{index + 1}.json"), mime="application/json",
                key=f"travel_json_{index}", on_click="ignore", disabled=st.session_state.travel_busy,
            )


# Chat interface
# Streamlit presents the conversation, source references and tool results.
# Session state retains chat history, and only a submitted message starts a request.
def main() -> None:
    st.set_page_config(page_title="Singapore Travel Companion", page_icon="🧭", layout="centered")
    initialize_session()
    memory = st.session_state.travel_memory
    busy = st.session_state.travel_busy
    problems = configuration_problems()
    st.markdown(STYLES, unsafe_allow_html=True)
    with st.sidebar:
        st.title("🧭 Travel Companion")
        st.caption("YOUR SINGAPORE TRIP")
        st.button("New chat", on_click=new_chat, disabled=busy, width="stretch", type="primary")
        st.caption("A new chat clears the conversation and its preferences. Download any answers you want to keep first.")
        st.divider()
        st.markdown("**Make the plan yours**")
        st.write("Share your dates, budget, interests, dietary needs, or who is travelling with you. Refine the plan in a follow-up message.")
        st.divider()
        st.caption(f"Completed turns in this chat: {memory.turn_count} / {memory.max_turns}")
        st.caption(f"Model request attempts this browser session: {st.session_state.travel_attempts}")
        st.caption("This is a local attempt counter, not your provider's remaining quota. It resets if the browser session is lost.")
        with st.expander("How answers are built"):
            st.write("Saved travel guides support destination facts. Our custom MCP server supplies weather and currency results. Suggested activities are labelled separately, with source references.")
            st.caption("Chat context is held in this browser session. Checked answers and their evidence are also saved locally in the project.")
    st.markdown("""
        <div class="travel-hero">
            <div class="travel-eyebrow">SINGAPORE · TRAVEL COMPANION</div>
            <h1>A little planning.<br>A better trip.</h1>
            <p>Explore ideas, account for the weather, and shape a trip around what matters to you.</p>
            <div class="travel-tags"><span>Travel guides</span><span>Weather forecasts</span><span>Currency conversion</span></div>
        </div>
    """, unsafe_allow_html=True)
    if not st.session_state.travel_items:
        st.markdown("**What would you like to explore?**")
        examples = [
            ("Plan a trip", "Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast."),
            ("Discover local food", "Which local dishes should I try at Singapore hawker centres?"),
            ("Check a budget", "Convert 10000 INR to SGD using the latest available reference rate."),
        ]
        for column, (title, text) in zip(st.columns(3), examples):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(text)
        st.caption("Type a question below to start. Mention your preferences so follow-up suggestions can take them into account.")
    for index, item in enumerate(st.session_state.travel_items):
        render_item(item, index)
    for problem in problems:
        st.error(problem)
    full = memory.turn_count >= memory.max_turns
    if full:
        st.info("This chat has reached its context limit. Start a new chat and restate the preferences you want to keep.")
    st.caption("A message uses up to three Gemini requests, including one correction if needed. Opening this screen or downloading an answer uses none. Failed API and tool calls are not automatically retried.")
    input_options = {}
    if "submit_mode" in inspect.signature(st.chat_input).parameters:
        input_options["submit_mode"] = "disable"
    st.chat_input(
        "Ask about your Singapore trip…", key="travel_prompt", max_chars=memory.max_question_chars,
        disabled=busy or full or bool(problems), on_submit=enqueue_message, **input_options,
    )
    pending = st.session_state.pop("travel_pending", None)
    if not pending:
        return
    item = {"question": pending}
    with st.chat_message("user"):
        st.markdown(pending)
    with st.chat_message("assistant", avatar="🧭"):
        with st.status("Working on your question…", expanded=False) as status:
            def progress(message):
                if message.startswith("Gemini request "):
                    st.session_state.travel_attempts += 1
                label = progress_label(message)
                if label:
                    status.update(label=label)
            try:
                if problems:
                    raise ValueError("Complete the local setup shown above before sending a question.")
                report = run_assistant(pending, memory, progress)
                item["report"] = report
                try:
                    item["report_filename"] = save_report(report).name
                except OSError:
                    item["save_warning"] = "The answer is available, but its local copy could not be saved. Use the download buttons to keep it."
                status.update(label="Answer ready", state="complete")
            except Exception as error:
                item["error"] = safe_error(error)
                status.update(label="The request could not be completed", state="error")
            finally:
                st.session_state.travel_busy = False
    st.session_state.travel_items.append(item)
    # Redraw from saved session data. The pending input was consumed above, so
    # this rerun cannot repeat the model request or tool execution.
    st.rerun()


if __name__ == "__main__":
    main()
