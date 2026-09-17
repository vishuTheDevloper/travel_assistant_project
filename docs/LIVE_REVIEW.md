# Review of the user's live Windows reports

Reviewed saved evidence from 14 September 2026. The application used Gemini 3.5 Flash Lite and the candidate's custom MCP server/client.

| Scenario | Original report | Result | Recorded model/tool requests |
| --- | --- | --- | --- |
| Three-day Singapore plan with two adults, vegetarian food and public transport | `assistant_20260914T175904_893547Z.json` | Answered. Dates 21–23 September match weather; each day has sourced indoor main activities; 18 evidence references matched source passages. | 2 Gemini requests; 1 successful weather tool call; no correction request |
| Same-chat extension to four days without repeating preferences | `assistant_20260914T180330_534620Z.json` | Answered. Earlier preferences and dates 21–24 September retained; 30 evidence references matched source passages. | 2 Gemini requests; 1 successful weather tool call; no correction request |
| INR 10000 to SGD with date and source | `assistant_20260914T180712_035232Z.json` | Partial. The correct tool and arguments were selected, but Frankfurter returned HTTP 522. No rate, amount or date was fabricated. | 2 Gemini requests; 1 failed currency tool call; no correction request |

The currency report retains two earlier turns. It was not a New-chat reset test. Its failure is explicitly recorded as a provider HTTP error. The app answered the currency question without inventing a conversion or displaying prior weather as a new result.

## Local presentation corrections

Version 12 uses those exact saved reports to verify overlapping-quote removal, distinct alternatives and transport grouping. Replays are clearly labelled as offline, retain original provider timestamps, and use zero Gemini or external provider requests. They do not establish that the currency provider has recovered or that a new version 12 live model response has succeeded.

## Confirmed environment checks

The user confirmed all 26 earlier offline tests and real retrieval passed on Windows/Python 3.13: four resources regenerated 454 chunks, the manifest matched, 454 vectors were checked, and 14 passages were retrieved. Locally, version 12 passed all 33 tests, including seven new presentation cases. The user subsequently confirmed all 33 version 12 tests passed on Windows; the user has now also confirmed all 41 version 13 tests passed on Windows/Python 3.13.

## Remaining submission work

- Final sample answers have been selected in `examples/README.md`; retain their original JSON evidence.
- Finish Git/repository delivery, record the demonstration video, and package the final submission.
- Review every assignment checklist heading against the final submission. Offline tests and a successful itinerary do not by themselves complete these deliverables.

## Additional live results

- `assistant_20260914T182703_198768Z.json`: version 12 currency UI succeeded. One `convert_currency` MCP call used INR 10000 to SGD; the returned daily rate was 0.0133, producing SGD 133.00, published 2026-09-14. The answer includes ECB/Frankfurter sources and the fee exclusion. Two Gemini requests, no correction, and zero previous turns were recorded. This is a historical result, not a guarantee of the current rate.
- `assistant_20260914T182920_354958Z.json`: version 12 food question had relevant citations but failed the intended source-only scope: one unnecessary weather call, two Gemini requests and an unsolicited one-day itinerary. Its earlier context contained only the currency request, not a request to schedule a food tour. Do not present it as a passed RAG-only example.

Version 13 addresses the scope failure with an informational-request guard, direct final-answer binding, response-scope validation and focused correction instructions. All 41 local offline tests passed, using mocked SDK responses and no new Gemini/provider requests. The fresh version 13 food answer has now passed the review below. Final demo/Git delivery and packaging remain pending.

## Successful version 13 food rerun

Original report: `assistant_20260914T185019_941207Z.json`, captured at `2026-09-14T18:50:19.936529+00:00` on the user’s Windows installation.

- The question was “Which local dishes should I try at Singapore hawker centres?”
- One Gemini request, zero MCP calls, no blocked calls and no correction.
- `itinerary` and `assumptions` are empty; no weather or currency figures are displayed.
- Nine selected evidence references match six retrieved passages; all six passages and their metadata match the saved source chunks. Four passages are cited in the rendered answer. The original Markdown matches the report.
- The dish guidance and practical ordering tips answer the question. The answer is verbose because factual descriptions preserve selected source excerpts. It does not claim current prices or verified stall availability.
- This chat had no earlier turns. Earlier currency/trip context was covered in offline scope regressions, not in this particular live run.

The old generic suggestions heading still mentioned a plan and weather even though the report contained neither. The final display-only correction replaces it with “Practical tips from the sources”. Offline rendering verified that only this heading/introduction changed; content, citations and the original recorded report remain unchanged. See `examples/food_ui_recorded.md` and `examples/food_display_review.md`. No additional Gemini/provider call was used for this correction.
