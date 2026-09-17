# Assignment coverage and remaining submission steps

This map follows the supplied **AI Travel Planning Assistant — Developer Assignment Brief** and the evaluator's clarification that the MCP server/client must be implemented by the candidate. Implementation coverage does not mean the video or repository has already been submitted.

## 1. Background

Stable Singapore destination facts come from local reference documents. Current weather and exchange rates come from the two custom MCP tools. The final response labels both separately.

## 2. Problem statement

`src/assistant.py` combines LangChain prompts, retrieval, model-selected MCP tools, context and answer validation; `app.py` provides a conversational interface.

## 3. Application scope

One destination: Singapore. Booking, payments, reservations and route navigation are outside scope. One-to-seven-day itineraries are an application limit, not a PDF requirement.

## 4.1 Destination Knowledge Assistant using RAG

### Major attractions and neighbourhoods

Wikivoyage districts/attractions and Visit Singapore Things to Do; see `data/sources.json` and `data/processed/documents.json`.

### Local transportation guidance

Wikivoyage Get around and Visit Singapore Public Transport/Accessibility sections.

### Cultural and practical travel tips

Wikivoyage culture/practical guidance and Visit Singapore Essential Travel Information.

### Food and local experiences

Wikivoyage Eat, local delicacies and dietary restrictions; itinerary/Things to Do sources provide additional experiences.

### Sample itineraries

Wikivoyage itinerary references and the dedicated Visit Singapore four-day itinerary resource.

### Indoor and outdoor activity suggestions

Source-described indoor attractions and outdoor parks, neighbourhoods and nature activities. Indoor settings require explicit evidence; a museum name alone does not establish it.

### RAG requirement 1: Load public documents/web pages

`src/document_loader.py`: BeautifulSoup plus embedded Visit Singapore page data. Original source title, URL, acquisition date and content hash retained.

### RAG requirement 2: Meaningful chunks

`src/chunk_documents.py`: heading-aware text splitting, target 1,000 characters and overlap 150; source/section metadata on every chunk.

### RAG requirement 3: Embeddings

`src/embeddings.py`: local `BAAI/bge-small-en-v1.5`, pinned revision, normalised 384-dimensional vectors. The same model/configuration is used for indexing and queries.

### RAG requirement 4: Vector store

`src/build_vector_store.py`: persistent Chroma, cosine distance and an index manifest; checks for stale or mismatched data.

### RAG requirement 5: Semantic retrieval

`src/retriever.py`: semantic candidates plus keyword candidates, refined by a local cross-encoder. The combined assistant reserves evidence for multiple itinerary topics.

### RAG requirement 6: Grounded answers

`src/assistant_response.py`: excerpt IDs resolve to actual retrieved source text. Unsupported data remains missing; exact quote matching is not a complete semantic proof.

### RAG requirement 7: Source titles/links

Rendered citations use the source metadata. Examples and downloadable evidence preserve the cited passages.

## 4.2 Current Travel Information using MCP

### MCP Tool 1: Weather

`mcp_server/weather.py` obtains Open-Meteo current estimates/forecasts. The custom `get_weather` handler returns dates, units, provider, timestamps and MCP provenance. Out-of-range dates and missing values are handled explicitly.

### MCP Tool 2: Currency conversion

`mcp_server/currency.py` obtains ECB reference rates through Frankfurter and calculates the amount with Decimal arithmetic. The response states rate date and excludes fees/margins. Equal currencies use an explicitly labelled identity conversion.

### MCP requirement 8: At least two tools

`mcp_server/server.py` exposes `get_weather` and `convert_currency` using FastMCP.

### MCP requirement 9: Tools available to the AI

`src/mcp_client.py` starts the candidate's server, initialises a session and converts discovered tools with LangChain MCP adapters.

### MCP requirement 10: Select by intent

Gemini receives the discovered schemas and chooses tool calls. Destination-only facts use RAG; hypothetical rainy-day suggestions do not inherently require a current forecast.

### MCP requirement 11: Required inputs

The model supplies actual dates, amount and currency codes, using stated conversation context. Server schemas/provider validation reject invalid values. Missing essential information should result in clarification.

### MCP requirement 12: Use tool results

`src/assistant_response.py` renders the checked numbers from successful MCP records; model arithmetic is not used for displayed conversions.

### MCP requirement 13: Identify MCP information

The response has a separate MCP information section, provider links and timestamps.

### MCP requirement 14: Failures without fabrication

Tool errors remain errors; no invented replacement weather/rate is supplied. Tested through real MCP transport with invalid dates and through combined-workflow fixtures.

### Evaluator clarification: Candidate's own server and client

Both implementations are included. SDKs/adapters support the protocol; no third-party ready-made weather/currency MCP server replaces this code.

## 4.3 Combined RAG and MCP response

### Required three-day Singapore scenario

Use the exact prompt: “Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast.” The model retrieves attractions, indoor/outdoor options, itinerary and transport passages; selects weather; then proposes dated daily activities.

The user's fresh version 11 three-day run and same-chat four-day follow-up were reviewed successfully, including matching dates, tool figures, citations and retained preferences. Version 12 display improvements were checked offline against those saved reports. A later currency UI attempt initially returned provider HTTP 522; a subsequent version 12 UI attempt succeeded with the correct amount, rate date and source. The food-only check then exposed unnecessary weather/itinerary work, addressed by version 13 scope checks; the fresh version 13 food rerun passed with one Gemini request, zero MCP calls and no added itinerary/date. See `LIVE_REVIEW.md`.

### Day-wise weather adaptation and indoor alternatives

A local planning preference prioritises sheltered main activities for rain probability at least 60%, precipitation at least 5 mm, or precipitation weather codes. Other settings can be conditional options. This is application policy, not a provider warning. A distinct indoor alternative is supplied when useful; a sheltered main activity need not repeat itself as a backup.

### Separate facts, tool results and recommendations

The UI distinguishes source excerpts, MCP figures and proposed day allocation. Recorded replays are labelled historical and do not enter chat memory.

## 5. Prompt engineering

### Retrieved content for destination facts

The system prompt limits destination facts to current retrieved excerpts.

### MCP responses for current information

The model must use actual tools for current weather/conversions, not prior chat answers.

### Avoid unsupported facts

Source references are required; unsupported indoor descriptions and invalid IDs cannot establish support.

### State insufficient information

Partial, insufficient-information and clarification statuses preserve explicit limitations.

### Structured recommendations

Pydantic output defines facts, suggestions, assumptions, dates, activities and limitations.

### Include references

Excerpt IDs map to source titles/URLs in the renderer.

### Distinguish facts and AI suggestions

Factual descriptions use selected excerpts. Activity/day selection remains a suggestion; weather/rate values are separately rendered from tools.

### Preserve conversation preferences

`src/conversation_memory.py` keeps ordered user requests and recent exchanges. Later explicit changes win; new chat resets the context.

## 6. Suggested knowledge base

Four public sources are supplied and listed in README section 6. Acquisition instructions are included. See `SOURCE_NOTICES.md` before redistribution; source content is not licensed as project code.

## 7. Technology requirements

LangChain, BGE embeddings, Chroma, Gemini, custom MCP client/server and Streamlit are implemented. All local inference runs on CPU; Gemini is selected through `.env`.

## 8. Minimum acceptance criteria

Code and saved evidence cover all ten listed capabilities: three-plus resources, semantic retrieval, citations, weather, currency, combined answers, memory, intent-based tools, error handling and a UI. Offline tests cover the repaired workflow. The user’s actual combined, same-chat context, currency and food outputs have now been reviewed. All 41 version 13 offline tests passed on the user’s Windows installation. These observed scenarios establish the demonstrated capabilities, not a guarantee for every possible prompt or provider response.

## 9. Deliverables

### Deliverable 15: Git repository

The uploaded project had Git initialised but no commits. Keep your original repository and commit the final reviewed code; provide the repository in the submission format requested by the evaluator. This repair ZIP does not overwrite your Git history.

### Deliverable 16: Working application

The Windows UI, real retrieval and 41 offline tests pass. Actual combined/context, currency and source-only food responses were reviewed successfully. Use `DEMO_GUIDE.md` to record the required demonstration; repeating checks solely for acceptance is unnecessary.

### Deliverable 17: Knowledge-base documents/acquisition

Supplied working snapshot plus exact acquisition instructions in README section 6. Review source-specific redistribution terms.

### Deliverable 18: README

Included: architecture, sources, RAG, MCP, prompt/context strategy, installation, execution, testing and limitations.

### Deliverable 19: Sample questions and responses

`examples/README.md` selects the successful food, currency, three-day and four-day follow-up examples. Original live JSON reports and clearly labelled display replays are included. Historical failures remain labelled as failures.

### Deliverable 20: Short demonstration

`DEMO_GUIDE.md` provides the recording sequence. The actual video remains to be recorded by the user.
