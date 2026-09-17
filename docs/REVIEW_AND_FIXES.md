# Project review and coordinated fixes

Baseline: the two identical project ZIPs supplied on 14 September 2026. Current assistant version: **13**.

## Root causes and changes

| Finding | Change | Verification |
| --- | --- | --- |
| Three-day draft put four eligible sheltered activities on Days 1-2 and an unsupported indoor activity on Day 3 | A new unconstrained plan may move a spare, already selected main activity to an empty day. The donor retains a main activity. No venue names or fixed three-day schedule are coded into this rule. | Exact saved three-day failure replays with three complete days. |
| Rescheduling could conflict with a user's fixed plan | Balancing is disabled for follow-ups, existing preferences, bookings, day-specific wording, and time-qualified source excerpts. | Fixed-day and follow-up regressions retain the unresolved gap rather than moving activities. |
| The corrected model draft said `partial` with no explanation and was discarded | Preserve the supported response as partial with an explicit limitation; request a concrete explanation through the one allowed correction. Never silently upgrade it to complete. | Real saved correction is renderable; wrong citations still fail. |
| General suggestions could add details absent from cited excerpts | Facts, activity descriptions and general suggestions use all selected exact source excerpts. The original model wording remains in diagnostic evidence. | Unsupported dairy/egg wording disappears without editing the source corpus. |
| Wrong or stale weather prose could survive an activity change | The displayed adjustment follows the retained plan and actual per-date weather policy. Live figures are rendered directly from MCP results. | Weather timestamps, numbers and missing values remain unchanged. |
| Child-only attractions could be used without eligible children | Preserve quoted restrictions; require a confirmed eligible child for explicit child-age limits. Ambiguous ages remain unconfirmed. | Adult-only updates and age boundary checks. This is a limited English-language check, not a complete eligibility engine. |
| Repeated backup/main activities | Omit redundant backups; detect a day that only repeats earlier main activities unless the user requests repetition. | Real four-day output and duplicate checks. |
| Follow-up duration was not checked against history | Enforce an unambiguous latest trip length when an itinerary is returned. Retain conversation context in new reports for replay. | Three-to-four-day update and one-to-seven-day synthetic cases. |
| Setup template incomplete; CPU wheel on a separate index | Completed `.env.example`, documented the separate CPU-index installation step, preserved package versions and marked `pywin32` Windows-only. | Dependency checks plus the user’s fresh Windows environment, passing offline checks and real retrieval. |
| Streamlit file watcher imported unrelated model modules | Added project-level watcher configuration. | Streamlit landing, submit, rerun, follow-up, failure and reset tests. |
| Replay repair command only handled older diagnostic structure | Repair can consume the selected draft in a normal run report too. Original tool timestamps remain historical. | Saved-report parsing and replay tests. |

## Validation scope

**Result: all 41 offline tests passed; four resources reproduced exactly 454 chunks, all 454 stored vectors were verified, and actual CPU retrieval returned 14 passages. Dependency resolution and `pip check` passed.** See `validation_result.json`.

Run `python scripts/check_project.py` for offline checks. Add `--retrieval` to verify the actual saved vector database and CPU retrieval with cached models and downloads disabled.

- Actual raw HTML is loaded and chunked; the regenerated chunks must equal the supplied `chunks.json`.
- The real 454-vector Chroma collection was checked against saved text, metadata and embedding settings.
- Saved three-day and four-day responses are replayed without changing the original JSON files or recorded tool results.
- The integration harness uses actual LangChain/Gemini schema conversion and real stdio MCP communication. Gemini generation is mocked and weather is a recorded fixture. A separate production-server check uses same-currency conversion and invalid future dates, which need no provider HTTP request.
- The repaired saved three-day draft takes two mocked model attempts and one recorded-weather tool call. An invalid citation exercises exactly one additional correction, without repeating tools.
- If both drafts fail, the UI receives checked tool data and an honest partial response instead of invented itinerary content.
- Streamlit tests confirm reruns/download controls do not submit another model request; failed requests do not enter memory; New chat clears preferences.

The review ran on Linux/Python 3.12 with the key application package versions from the uploaded Windows environment. It did **not** run a fresh Gemini generation, perform a clean Windows/Python 3.13 install, or record the demonstration video. The user's private API key was neither supplied nor required.

## Windows text-encoding correction

The user reported 25 passing tests and one failure where the saved expected weather unit was `Â°C` and the MCP result was `°C`. The test parent used an implicit locale encoding, while the MCP subprocess explicitly used UTF-8. The underlying saved weather file was correct.

Saved JSON reads in the integration test, its MCP fixture, and the manifest check now specify UTF-8 explicitly (accepting a UTF-8 BOM). Temporary JSON test writes also specify UTF-8. No weather data or equality assertion was changed.

The exact reported failure was reproduced on Linux by making unspecified `Path.read_text` calls use Windows cp1252. After the correction, all 26 offline tests and the four-resource/454-chunk manifest check passed under the same simulated default encoding. This check made zero Gemini or external weather/currency requests. The user subsequently confirmed all 26 original tests and real retrieval passed on Windows. The user also confirmed all 33 version 12 tests passed on Windows. The user subsequently confirmed all 41 version 13 tests passed on Windows/Python 3.13.

## Version 12 display corrections

- Merge overlapping selected source spans within the same passage so repeated sentences appear once. Keep every selected qualification. Mark gaps between separate spans with `[…]` instead of implying continuous quotations.
- Hide an indoor backup only when its full evidence is already displayed among that same day's activities. Preserve backups containing additional qualifications.
- Label a source-supported indoor optional activity as an optional indoor activity. Outdoor, mixed and unknown options retain their weather condition.
- Put general transport-guide excerpts under Getting around once, instead of displaying repeated weather-dependent attractions. Mixed transport/venue evidence and main activities are not moved by this display rule.
- Keep original structured drafts, selected evidence, day allocation, weather, currency and tool error data unchanged. No provider fallback, extra model request or automatic retry was added.

The seven new regression tests include both fresh live itinerary reports and the failed currency report, along with qualification-preservation and mixed-evidence cases. All 33 offline tests passed. Display replays used zero Gemini and zero live provider calls. See `LIVE_REVIEW.md` for the distinction between the user's live evidence and the local replay checks.

## Version 13 request-scope correction

The live food question `Which local dishes should I try at Singapore hawker centres?` produced relevant source facts, but also an unsolicited one-day itinerary and weather call. The earlier conversation contained only a currency request. This was a scope failure even though the generated report said `answered`.

Clear information requests now expose only the local final-answer function and do not open an MCP connection. Unrequested itineraries or dates fail validation. A malformed request for an unavailable live tool is blocked before execution and may use the existing single correction, without calling the provider. The correction instructions for information questions no longer request itinerary activities. Unrelated earlier trip requests no longer expand retrieval into itinerary facets; explicit dietary preferences remain available.

This is a conservative guard for common English information questions. Planning, live and ambiguous requests retain model-selected MCP tools. The full question and relevant memory remain available to the model. No hard-coded food answer or new model-based router was added.

Eight additional regressions verify normal one-request reference answers, blocked weather calls, one correction for an invented itinerary, missing destination information, and unaffected live/planning requests. Their SDK responses are fixtures, not fresh model generations. All 41 offline tests passed. The original faulty food report is preserved; strict replay rejects it instead of relabelling it as successful.

## Version 13 live acceptance and final display wording

The user’s live report `assistant_20260914T185019_941207Z.json` answered the hawker-food question in one Gemini request, with zero MCP calls, no correction, no itinerary and no assumed date. All nine selected references matched six retrieved passages, which also matched the stored source chunks. The uploaded Markdown matched the report. This was an actual user-run Gemini answer; the review itself used no provider calls.

The remaining generic “Suggested plan” heading and day/weather introduction were display-template wording. Non-itinerary suggestions now use “Practical tips from the sources”; itinerary wording refers to weather data only when a successful weather record exists. The saved food answer was rendered offline and checked to differ only in this heading/introduction. Original report bytes, selected sources and structured data are preserved. The assistant remains version 13 because request handling and schemas did not change.

## Remaining limits

Source excerpts can be exact yet irrelevant or unsuitable for a particular traveller. The local checks do not prove full semantic grounding, age eligibility, accessibility, route feasibility, or current opening hours. Specific user constraints outside the recognised checks rely on the model and human review. A source shortage remains a partial answer, especially for longer trips that need many distinct indoor venues. Model/provider 429, 503 and availability failures remain external limitations.

## Technical references

The project uses LangChain's tool-binding integration: [ChatGoogleGenerativeAI](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai). The CPU installation follows the [PyTorch installation guide](https://pytorch.org/get-started/locally/). Model output selection is still performed by the LLM; local rules validate and format the selected evidence.
