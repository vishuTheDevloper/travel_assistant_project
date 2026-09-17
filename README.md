# AI Travel Planning Assistant - Singapore

A Singapore travel assistant built with **LangChain, local RAG, a custom MCP server and client, Gemini, and a Streamlit chat interface**. Saved documents supply destination facts; MCP tools supply current weather and reference-rate estimates.

## Git Hub Repo link below
https://github.com/vishuTheDevloper/travel_assistant_project

## Start the project

Use **Windows Command Prompt**. In VS Code, open the project folder containing `app.py`, then choose **Terminal > New Terminal > Command Prompt**. All commands below run from that folder. In a separate Command Prompt, use `cd /d "C:\path\to\travel_assistant_project"`, replacing the example with your actual folder path.

### Already installed: start here

If this folder already has its working `.venv`, private `.env`, and matching vector database, only run:

```bat
.venv\Scripts\activate.bat
python -m streamlit run app.py --server.fileWatcherType none
```

Open the **Local URL printed in the terminal**. Keep that terminal running. Stop the app with **Ctrl+C**. Restart it manually after changing Python files because file watching is disabled. The app starts its own MCP subprocess when tools are needed; no separate server terminal is required.

### First-time setup: install Python packages

Prerequisites: **64-bit Python 3.13** with the Windows `py` launcher, and internet access for package/model downloads, Gemini and live tools. The tested laptop uses Windows 11 and Python 3.13.15; embeddings and reranking run on CPU. Extract the ZIP before starting. Create the virtual environment in the project's final location rather than copying another computer's environment.

Run these commands one at a time:

```bat
py -3.13 --version
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
python -m pip install -r requirements.txt
python -m pip check
```

Stop if an installation command fails. Continue after `pip check` reports **No broken requirements found**. The prompt should show `(.venv)`. If Python is missing, install Python 3.13 and reopen Command Prompt. If a pinned package is unavailable for your platform, resolve that specific installation error before continuing; arbitrary version changes are not a verified reproduction.

**One requirements file, two installation commands:** every package version, including `torch==2.14.0+cpu`, is now listed in `requirements.txt`. The first install command obtains the CPU PyTorch build from its official CPU package index. The second installs the remaining pins from PyPI and accepts the already installed matching torch build. Keeping the CPU index on its own command avoids redirecting every project dependency to that index. The package versions are unchanged. No CUDA, torchvision or torchaudio is required. See [PyTorch installation](https://pytorch.org/get-started/locally/) and [pip installation behaviour](https://pip.pypa.io/en/stable/cli/pip_install/).

### Configure the API key

Create `.env` only if it does not already exist, then open it:

```bat
if not exist .env copy .env.example .env
notepad .env
```

Set these two values, save and close the editor:

```dotenv
GOOGLE_API_KEY=your_own_google_ai_studio_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

The model name above is the configuration used in the recorded project runs; your API project must have access to it. `src/model_config.py` reads the project-root `.env` for all answer-model entry points. It does not control the local embedding model. Keep the real API key private and close this file before screen sharing or recording.

### Prepare the knowledge base when needed

This ZIP includes saved source data and the vector database. If `data/processed/chunks.json`, `data/vector_store/chroma.sqlite3`, and `data/vector_store/index_manifest.json` are present and consistent, continue to the checks below.

For a clean rebuild, first ensure all four saved HTML pages listed in `data/sources.json` exist under `data/raw/`; section 6 explains their acquisition. Then run:

```bat
python src/document_loader.py
python src/chunk_documents.py
python src/embeddings.py
python src/build_vector_store.py
```

These steps extract, chunk, embed and index the documents. They make no Gemini requests, but first-time local model downloads require internet access. Review the extracted Markdown and `chunks_preview.md`. After changing sources or embedding settings, rebuild the matching index and restart the app.

### Check the project and launch

```bat
python scripts/check_project.py
python -m streamlit run app.py --server.fileWatcherType none
```

The project check runs locally with mocked model responses and recorded tool fixtures; it makes no Gemini or external weather/currency requests. The supplied implementation has **41 offline tests**. The first real retrieval may download/load the local embedding and reranker models; let it finish. The Streamlit browser shows chat, citations, tool activity and answer/evidence downloads. Ask follow-ups in the same chat to retain preferences; **New chat** clears them.

### Updating an existing working copy

Stop Streamlit, extract this ZIP separately, and copy the project contents into the existing folder containing `app.py`, replacing matching files. Retain that folder's private `.env` and working `.venv`. This revision adds explanatory comments and reorganises documentation/package lists; application behaviour and package versions are unchanged, so an existing working installation needs no reinstall.

Then use the two commands under **Already installed: start here**. Keep historical answer exports labelled with their original dates. Git delivery and the demonstration recording remain submission steps; the code comments describe implementation responsibilities only.

---

Documentation baseline: assistant version **13**. Setup and explanatory comments updated on **17 September 2026**. The application supports one-to-seven-day itineraries; this is a project limit, not a limit specified by the assignment. The remaining sections describe the implementation against the assignment requirements.

## 1. Background

Travel questions mix relatively stable information, such as attractions and transport guidance, with changing information, such as a forecast or exchange rate. This project keeps those sources separate so an old travel document or previous chat answer is not treated as current weather or a current rate.

## 2. Problem statement and solution

The application accepts a travel question, retrieves relevant Singapore passages, and provides their source titles and links. When the question requires current information, Gemini selects the appropriate tool exposed by the project's MCP server. The application combines the returned data with the retrieved passages to produce a structured answer.

For example, a weather-aware itinerary uses RAG for activity ideas and MCP for the forecast on the actual trip dates.

## 3. Application scope

- One destination: **Singapore**.
- Free-form questions about attractions, neighbourhoods, culture, practical guidance, food, transport and itineraries.
- Weather-aware plans, currency conversion, and follow-up changes to earlier preferences.
- No flight or hotel booking, payment processing, reservation service, or turn-by-turn navigation.
- Questions outside the available knowledge or tool coverage receive a limitation or clarification instead of invented details.

## 4. Core features

### 4.1 Destination Knowledge Assistant using RAG

#### Knowledge-base topic coverage

| Required topic | Main source coverage |
|---|---|
| Major attractions and neighbourhoods | Wikivoyage; Visit Singapore Things to Do |
| Local transportation guidance | Wikivoyage; Essential Travel Information |
| Cultural and practical travel tips | Wikivoyage; Essential Travel Information |
| Food and local experiences | Wikivoyage; sample itinerary |
| Sample itineraries | Visit Singapore 4 Days in Singapore; Wikivoyage |
| Indoor and outdoor activities | Things to Do; Wikivoyage; sample itinerary |

#### Loading public content

`src/document_loader.py` reads the saved HTML files listed in `data/sources.json`. It extracts the Wikivoyage article body and Visit Singapore content, including text inside the site's embedded `aem-data` attributes. Navigation and other unrelated markup are removed. The output is LangChain document content saved as `data/processed/documents.json`, with readable Markdown copies for inspection.

#### Meaningful chunking

`src/chunk_documents.py` preserves document headings and section paths. It splits large sections into chunks with a maximum target of **1,000 characters** and **150 characters of target overlap** within sections. Actual overlap can vary at text boundaries. Source titles, URLs, identifiers and section metadata remain attached.

The original four-page snapshot produced **454 chunks**: 300 Wikivoyage, 24 essential-information, 34 itinerary, and 96 things-to-do chunks. Fresh website downloads may produce different counts.

#### Embeddings

`src/embeddings.py` uses `BAAI/bge-small-en-v1.5` on CPU, producing normalized **384-dimensional** vectors. Its revision is pinned in code. Indexing and query retrieval use the same embedding factory and configuration; queries additionally use the model's retrieval instruction prefix.

The embedding model accepts up to 512 tokens. The original chunks were checked before indexing; the longest contained 340 tokens. Changing the embedding model or its settings requires rebuilding the index.

#### Vector store

`src/build_vector_store.py` persists embeddings in **Chroma** under `data/vector_store/`, using cosine distance. The index manifest records its identity and embedding settings. Text, metadata and vectors are checked before the manifest is marked ready. An unchanged build can resume without adding duplicate chunks.

#### Retrieval

`src/retriever.py` combines semantic candidates with TF-IDF keyword candidates, then reranks them locally with `cross-encoder/ms-marco-MiniLM-L6-v2`. These operations do not call Gemini.

The combined assistant uses additional local searches for itinerary topics, transport and dietary preferences. It reserves more indoor candidates for longer trips while keeping the total context within **14 passages**. This changes the evidence available to the model; it does not assign hard-coded attractions to particular days.

#### Grounded answer generation and citations

Gemini receives the question, conversation context and retrieved source excerpts through LangChain. It selects excerpt identifiers such as `P2-E1`. Python resolves those identifiers to exact source text and checks citation integrity. Knowledge-base facts are displayed as source excerpts. The model selects activities, their days and roles; their factual descriptions use the complete selected source excerpts and source-section labels. General suggestions also use all selected exact source excerpts. Weather adjustments are generated locally from the returned daily forecast policy and the retained plan. Duplicate indoor alternatives are removed locally. In a new unconstrained itinerary, a spare, already selected main activity can fill an empty day; no new venue is invented. Fixed schedules and follow-ups are not rebalanced.

The final answer shows source titles and URLs. Valid quotation matching does **not** prove that every generated sentence follows from its quotation. Human review of factual meaning remains necessary.

#### Missing knowledge

The assistant can return `partial`, `insufficient_information`, or `clarification_needed`. It must explain unsupported requests. It does not obtain additional destination facts through MCP or silently invent a missing attraction, price or opening hour.

### 4.2 Current Travel Information using MCP

#### Custom server and client

The evaluator clarified that the assignment requires a **custom MCP server and a client that consumes it**. This project implements both:

- `mcp_server/server.py`: our `singapore_travel` server, built using the MCP SDK's FastMCP support.
- `src/mcp_client.py`: starts that server as a Python subprocess, initializes a **stdio MCP session**, discovers its tools and exposes them through LangChain adapters.

The client uses the active Python interpreter to start the server. The Streamlit app manages this connection; a second manually started server terminal is not required. The SDK and adapter are protocol libraries, not ready-made travel servers.

#### MCP Tool 1: Weather information

`get_weather` is implemented in `mcp_server/weather.py` and exposed by our server.

- Inputs: Singapore city name and optional inclusive `start_date` / `end_date` in `YYYY-MM-DD` form.
- External provider: Open-Meteo.
- Outputs: dated daily conditions, temperature ranges, rain probabilities and other available fields, plus a separate timestamped current model estimate.
- The implementation accepts dates from today through today + 15 days in Singapore time. Provider availability is still checked.
- Missing values remain unavailable rather than becoming zero.
- Location, provider, units and retrieval timestamps are retained.

A daily rain probability does not identify the hours of rain or mean it rains all day. The current model estimate is separate from future daily forecasts.

#### MCP Tool 2: Currency conversion

`convert_currency` is implemented in `mcp_server/currency.py` and exposed by our server.

- Inputs: amount, source currency and target currency.
- External provider: Frankfurter, requesting ECB reference-rate data.
- Supported codes: INR, SGD, USD, EUR, GBP, AUD, CAD, CHF and JPY.
- Decimal arithmetic is used for amounts and conversion; the result includes the rate and publication date.
- This is an estimate excluding bank fees, commissions and provider margins. It does not exchange money.
- Invalid amounts, unsupported currencies and rates outside the application's freshness policy produce an error.

#### Tool availability, selection and inputs

LangChain makes both discovered MCP tools available to Gemini. The model selects tools from the user's intent and supplies actual input arguments. For a budget already stated in the conversation, the model can use that context; essential missing inputs should trigger clarification.

Destination-only questions are intended to use RAG without a weather or currency call. Tool selection is visible in the exported evidence JSON.

#### Returned information, provenance and errors

Successful tool results include the custom server and tool identity. The UI renders weather figures directly from those results and identifies their provider and timestamps. Failed or unavailable tools are reported without fabricated replacement values.

### 4.3 Combined RAG and MCP response

#### Required assignment scenario

> Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast.

The assistant retrieves attraction, indoor/outdoor, itinerary and transport evidence. It interprets next week using the Singapore calendar, states any assumed start date, and requests the corresponding weather dates through MCP.

#### Day-wise weather adaptation

The application applies a conservative planning rule: days with a precipitation probability of at least 60%, precipitation of at least 5 mm, or a precipitation weather code prioritise sourced sheltered main activities. Outdoor options are conditional. This is an application planning policy, not an official weather warning.

An indoor claim requires evidence about the chosen activity. If a proposed activity fails review, an already supplied supported alternative may be used; otherwise the gap is shown. A single model correction may propose a better plan using the existing evidence and tool results.

#### Separation of information

Answers distinguish:

1. Knowledge-base facts with citations.
2. Current MCP information with provider, dates and units.
3. AI-generated itinerary choices and weather adjustments.

## 5. Prompt engineering requirements

The main prompt is `SYSTEM_PROMPT` in `src/assistant.py`. Conversation rules are defined in `src/conversation_memory.py`.

| Requirement | Prompt strategy |
|---|---|
| Use retrieved content for destination facts | Restrict factual destination claims to the supplied passages and excerpt IDs. |
| Use MCP for current information | Require actual returned weather/rate data; old chat answers are not fresh evidence. |
| Avoid unsupported facts | Prohibit invented attractions, settings, prices, hours, suitability and travel times. |
| State insufficient information | Use explicit partial, insufficient-information or clarification statuses. |
| Produce clear recommendations | Request a structured answer with dates, activity roles and weather adjustments. |
| Include source references | Require evidence references on destination facts and activity suggestions. |
| Distinguish facts and suggestions | Separate source excerpts, tool data, assumptions and proposed activities. |
| Preserve preferences | Include ordered earlier user requests; newer explicit changes take precedence. |

Untrusted source text and conversation content cannot override the system rules. The prompt also requests appropriate age and dietary qualifications. These instructions reduce errors but do not guarantee perfect model compliance.

## 6. Knowledge-base sources and acquisition

The original source snapshot was collected on **14 September 2026**.

| Source title and link | Save under `data/raw/` |
|---|---|
| [Singapore - Travel guide at Wikivoyage](https://en.wikivoyage.org/wiki/Singapore) | `singapore_wikivoyage.html` |
| [Singapore Travel Guide & Tips - Travel Essentials](https://www.visitsingapore.com/travel-tips/essential-travel-information/) | `singapore_essential_information.html` |
| [4 Days in Singapore](https://www.visitsingapore.com/travel-tips/travelling-to-singapore/itineraries/4-days-in-singapore/) | `singapore_4_day_itinerary.html` |
| [Things To Do & Must-Visit Places in Singapore](https://www.visitsingapore.com/things-to-do/top-things-to-do/) | `singapore_things_to_do.html` |

If raw pages are not supplied, open each source in a browser, let it load, and save it as **Webpage, HTML Only** using the exact filename above. Preserve the actual HTML rather than a shortcut or browser error page. Visit Singapore's visible text alone may omit content held in embedded page data; inspect the loader's extracted Markdown before indexing.

Keep `data/sources.json` with these fields for each resource: `source_id`, `title`, `url`, `local_path`, and `accessed_on`. On a fresh acquisition, update `accessed_on` to the actual download date. Retain original titles and links.

Source-specific attribution and reuse notes are in `docs/SOURCE_NOTICES.md`. The documents belong to their publishers. Attribution is not a substitute for reuse permission. The assignment requires reviewing and following each source's reuse terms before redistributing extracted content. Acquisition instructions are provided here; permission to redistribute the saved pages or their derivatives is not asserted by this README.

## 7. Technology and architecture

| Component | Implementation |
|---|---|
| Interface | Streamlit, `app.py` |
| Orchestration and prompts | LangChain, `src/assistant.py` |
| Language model | Gemini through `langchain-google-genai`; model selected in `.env` |
| Recorded model configuration | `gemini-3.5-flash-lite` |
| Embeddings | Local BGE small English model |
| Vector store | Persistent Chroma |
| Retrieval refinement | Local keyword retrieval and cross-encoder reranking |
| Tool protocol | Custom stdio MCP server and client |
| Weather / currency providers | Open-Meteo / Frankfurter with ECB data |
| Answer validation | Pydantic schemas, source excerpt checks and local review rules |
| Memory | Bounded in-process conversation storage in Streamlit session state |

The UI passes the current question and session memory to the LangChain assistant. Local retrieval supplies evidence. Gemini either submits a destination answer or requests MCP tools. The client invokes our server, which contacts the external provider. Returned tool messages and source excerpts then support the final model answer. Local validation and rendering happen before a reviewed answer is stored in memory and displayed.

### Conversation context

All user requests are retained in order within a maximum **12-turn** conversation. Up to three recent assistant exchanges are included in the model context, with size limits. Preferences do not require an extra summarisation model call.

A new explicit preference replaces an older conflicting preference. New run reports retain the conversation context used for that answer, so offline replay can check those same preferences. Earlier reports without context remain explicitly limited to their saved question. Old assistant answers are context, not factual evidence. **New chat** clears the current conversation. Memory is not a durable account profile and is not restored by uploading an evidence JSON. A reviewed partial answer may enter memory; a request that stops with an error does not add a rejected answer.

### Request budget

A combined-assistant message uses at most **three Gemini request attempts**: ordinarily one for a destination-only answer or two when live tools are needed, plus at most one validation correction. Each live tool can be executed at most once per message. Correction reuses recorded results and does not refetch them.

Automatic HTTP/provider retries are disabled. Opening the UI and downloading an answer do not generate another Gemini request. Provider quotas and model availability still apply; unlimited usage is not promised.

## 8. Acceptance criteria and current evidence

| Criterion | Implementation or evidence |
|---|---|
| At least three resources | Four saved source documents; original snapshot indexed as 454 chunks. |
| Semantic retrieval | BGE embeddings and Chroma, with local retrieval review outputs. |
| Answers with references | Source excerpt IDs resolve to saved text; titles and links are rendered. Semantic review remains necessary. |
| Weather through MCP | Custom `get_weather` calls recorded with provider and dates. |
| Currency through MCP | Custom `convert_currency` tested directly and through model-selected tool use. |
| Combined RAG + MCP | Three-day and four-day run records were reviewed. |
| Multi-turn context | A real follow-up retained the four-day dates, vegetarian preference and public transport, and replaced a children's attraction after an adults-only update. |
| Intent-based tools | Recorded currency/weather selections and separate RAG checks; inspect each demonstration's tool trace. |
| Missing knowledge and failures | Out-of-scope destination response, provider errors and local validation checks were observed. |
| Simple interface | Streamlit chat, progress, new-chat control and answer/evidence downloads. |

The Windows/Python 3.13 installation passed all **41 offline tests**. Actual UI reports were also reviewed for source-only food answers, currency conversion, a three-day weather-aware plan, and a same-chat extension to four days retaining preferences. The food run used one Gemini request and no MCP calls; each reviewed currency/combined run used two Gemini requests and one appropriate MCP call. These are recorded examples with their original dates, not new forecasts or exchange-rate quotations. The final display wording was checked offline. See `docs/LIVE_REVIEW.md` and `docs/REVIEW_AND_FIXES.md` for evidence and limitations.

## 9. Deliverables

The assignment requires all of the following:

- Source code in a Git repository.
- A working application.
- Knowledge-base documents or clear acquisition instructions.
- This README covering architecture, sources, RAG, MCP, prompts, context and setup.
- Sample questions with actual application responses.
- A short demonstration showing RAG, MCP, a combined answer and conversational context.

A requirement-by-requirement map is in `docs/ASSIGNMENT_CHECKLIST.md`; saved response examples are indexed in `examples/README.md`; the demo steps are in `docs/DEMO_GUIDE.md`.

A ZIP is a packaging format; it does not remove the Git-repository requirement. Selected successful sample responses are included. Git submission details, source redistribution review and the demonstration recording remain separate completion checks. This README does not claim that those items have already been submitted.

## 10. Additional verification and terminal commands

The complete installation and browser startup sequence is at the [start of this README](#start-the-project). These commands are optional; they are not required every time the application starts.

| Purpose | Command | External requests |
| --- | --- | --- |
| Offline project checks | `python scripts/check_project.py` | None; model/provider responses use fixtures |
| Also verify stored vectors and actual retrieval | `python scripts/check_project.py --retrieval` | None; requires already cached local models |
| Prepare combined context | `python src/assistant.py --prepare` | No Gemini/provider calls; first-use local model downloads may occur |
| Check a Gemini connection | `python scripts/check_gemini.py` | One Gemini request |
| Check the custom MCP connection | `python src/mcp_client.py` | Weather/currency requests; no Gemini generation |
| Review local retrieval | `python src/retriever.py` | No Gemini calls; may download local models on first use |
| Terminal conversation | `python src/assistant.py --chat` | Gemini and any selected tools |

For a single combined request:

```bat
python src/assistant.py --ask "Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast."
```

To recheck a saved response without model/provider requests:

```bat
python src/assistant.py --replay-report "data/processed/assistant_runs/assistant_20260914T185019_941207Z.json"
```

A replay uses the original evidence and timestamps, writes a labelled report under `data/processed/assistant_replays/`, and does not update chat memory. Offline checks and saved replays are not fresh generated answers. Real run reports are saved under `data/processed/assistant_runs/`.

## 11. Sample questions and demonstration procedure

The prompts below are a test procedure, not a claim that every response is already approved. Preserve actual downloaded JSON and Markdown alongside the final selected examples.

1. **RAG:** `Which local dishes should I try at Singapore hawker centres?` Show the retrieved source references and check that destination-only information did not require weather or currency.
2. **MCP currency:** `Convert 10000 INR to SGD using the latest available reference rate.` Show tool arguments, returned rate, publication date and the fee exclusion.
3. **Required combined scenario:** `Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast.` Show source passages, MCP weather dates and the proposed activities.
4. **Context:** In that same chat, state a preference, then send a follow-up that depends on it without repeating it. Confirm that the latest preference is applied and dates remain consistent.
5. **Missing knowledge:** `Which ski resorts should I visit in Switzerland?` Confirm that the Singapore knowledge limitation is explained.

A historical currency test returned **10,000 INR = 132.70 SGD**, at a reference rate of **0.01327 dated 11 September 2026**. This is a recorded example, not a current exchange-rate quotation. Future runs can differ.

For the video, show both the user/bot interaction and enough source/tool evidence to explain the workflow. Replays must be labelled as recorded data with their original timestamps. Avoid exposing `.env` or API keys on screen.

## 12. Known limitations and troubleshooting

### Answer quality still requires review

Version 13 limits clear information questions to the local final-answer schema, with no live tools exposed and no MCP connection opened. It rejects unrequested itineraries and assumed dates, and excludes unrelated earlier trip details from retrieval expansion while retaining conversation preferences. The model still chooses tools for planning, live and ambiguous requests. Scope detection is conservative and supports common English phrasings; it is not a complete multilingual intent classifier. The new flow passed eight additional offline regressions (41 total), including a food question after earlier currency/trip conversations. The user subsequently confirmed all 41 tests passed on Windows/Python 3.13. The live version 13 food answer succeeded with one Gemini request, zero MCP calls and no itinerary or assumed date; all nine selected references matched the saved passages. See `docs/LIVE_REVIEW.md`. A final display-only adjustment uses “Practical tips from the sources” for non-itinerary suggestions; it was checked against that same saved answer without another model request.

Version 12 removes overlapping source quotations in the display, keeps source gaps visible as `[…]`, hides an identical same-day backup, labels supported indoor options correctly, and consolidates general transport advice. The saved draft, selected evidence, day allocation and tool results remain unchanged by this display step. All 33 offline tests passed. Version 11 live three-day and four-day follow-up reports were reviewed successfully; the later currency request recorded a provider HTTP 522 and an honest partial answer. See `docs/LIVE_REVIEW.md`.

Version 11 fixes the saved empty-day failure, uncited optional placeholders, unexplained partial-status crashes, duplicated backups, and unsupported generated wording in general suggestions. A new unconstrained plan can move a spare model-selected activity into an empty day; a follow-up or fixed schedule cannot. Restrictions for child-only activities require a confirmed eligible child. Source selection and overall suitability still need review. No local test can guarantee future Gemini availability or perfect interpretation of every travel request.

The local checks validate schema, citation identity, quotation presence, dates, selected dietary conflicts and explicit indoor wording. They are not a complete semantic, age-eligibility, accessibility, nutrition or route validator. A JSON status of `answered` does not replace human review.

Opening hours, prices, availability and detailed transfers are not verified live. The supported seven-day length is an application limit; actual source coverage can still be insufficient for a useful plan.

### Common problems

| Problem | Action |
|---|---|
| PowerShell blocks activation | Use Command Prompt and `.venv\Scripts\activate.bat`. |
| Package import error | Activate the project environment, install its requirements and run `python -m pip check`. |
| Missing source page or too little extracted text | Check the saved HTML and `data/sources.json`; review embedded Visit Singapore content and loader output. |
| Missing or mismatched vector store | Build it from the matching chunks and embedding settings; restart the app. |
| First retrieval is slow | Local models may be downloading/loading on CPU. Watch terminal progress. |
| Hugging Face unauthenticated/symlink warning | These warnings did not prevent the original CPU setup from working. Investigate separately if an actual download error occurs. |
| Streamlit watcher/model import errors | Use the documented `--server.fileWatcherType none` launch command. |
| Browser cannot connect | Use the Local URL and port shown in the running terminal; inspect startup errors. |
| Gemini 429 | Inspect project/model quota and usage. Repeating the same request immediately may continue to fail. |
| Gemini 503 | Provider capacity was unavailable; the app does not automatically repeat the request. |
| Gemini 404/model unavailable | Check access to the configured model and update `GEMINI_MODEL` to an accessible model. |
| Unsupported forecast date or currency | Correct the input or accept the reported tool limitation; the app must not invent replacement data. |
| A day is partial or missing | Inspect the source evidence and review notes. Never label the run complete solely because the request returned JSON. |

### Preparing the final submission

Keep the working environment on your own computer. Create a separate submission copy with one outer `travel_assistant_project` folder. Include code, requirements, README, source manifest/acquisition instructions, permitted data, selected examples and the demonstration. Exclude private `.env`, `.venv`, `.venv_old`, caches and irrelevant diagnostics from the shared copy.

Retain the required Git repository and provide the agreed repository submission separately if the ZIP excludes its history. Review sample responses, reuse terms and the final package contents before submission. Do not present recorded weather as live, or local synthetic checks as model-generated examples.
