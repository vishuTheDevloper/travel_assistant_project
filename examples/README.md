# Sample questions and application responses

These examples are drawn from the user's real saved runs. No new Gemini response was generated during this repair review. Original JSON files remain under `data/processed/`.

| Example | Answer | Original evidence | Status of evidence |
| --- | --- | --- | --- |
| Hawker-centre dishes | [RAG food response](rag_food_recorded.md) | `data/processed/rag_result_food.json` | Recorded RAG answer with citations. |
| INR to SGD | [Currency response](currency_recorded.md) | `data/processed/tool_agent_currency.json` | Recorded model-selected MCP currency call; original rate date applies. |
| Singapore forecast | [Weather response](weather_recorded.md) | `data/processed/tool_agent_weather.json` | Recorded MCP forecast; not a forecast fetched during this review. |
| Three-day combined itinerary | [Offline repaired answer](combined_three_day_offline_replay.md) | [Replay evidence](combined_three_day_offline_replay.json); original `assistant_20260914T162228_167494Z.json` | Saved model draft rechecked by version 11, with local balancing of an already selected activity. Zero new model/provider calls. |
| Four-day combined itinerary | [Offline rechecked answer](combined_four_day_offline_replay.md) | [Replay evidence](combined_four_day_offline_replay.json); original `assistant_20260914T155317_681432Z.json` | Recorded four-day plan, rendered with current source/duplicate checks. Zero new model/provider calls. |

Fresh version 11 combined and same-chat follow-up reports have now been reviewed; see the additional evidence below. Follow `docs/DEMO_GUIDE.md` to prepare the final video. Historical follow-up `assistant_20260914T153753_755960Z.json` demonstrates retained preferences but has a main-day gap; it should not be presented as a fully successful final plan.

The `tests/` examples are synthetic test fixtures, not sample generated travel answers. Keep that distinction when explaining the project.

## Fresh Windows evidence and version 12 display review

The original version 11 three-day and four-day follow-up runs succeeded on the user's laptop. The subsequent currency attempt returned provider HTTP 522 and a truthful partial answer. Their original JSON reports are preserved under `data/processed/assistant_runs/`.

- [Three-day display review](three_day_display_review.md): offline rendering of `assistant_20260914T175904_893547Z.json`.
- [Four-day display review](four_day_display_review.md): offline rendering of `assistant_20260914T180330_534620Z.json`.
- [Currency error display review](currency_error_display_review.md): offline rendering of `assistant_20260914T180712_035232Z.json`.

Each display review has a matching JSON and is labelled as a replay. No fresh weather/rates or Gemini response was obtained during these replays. A subsequent currency UI response succeeded; see the recorded version 12 currency example below.

## Subsequent currency success and food-scope failure

[Recorded currency UI answer](currency_ui_recorded.md) corresponds to `assistant_20260914T182703_198768Z.json`. This actual version 12 UI run succeeded after the provider recovered; its original rate date applies.

The food report `assistant_20260914T182920_354958Z.json` is preserved as a failing scope example because it called weather and invented a day plan. Version 13 prevents this for clear information questions. The actual version 13 food rerun subsequently passed; see the final evidence selection below. The derived source-only draft in `tests/test_request_scope.py` is explicitly a test fixture, not a generated demo answer.

## Selected successful examples for submission

These four scenarios have been reviewed. The combined/context display reviews retain the dates and tool data from their successful original live runs; they are not new forecasts.

| Capability | Readable answer | Original live JSON in `data/processed/assistant_runs/` | Recorded calls |
| --- | --- | --- | --- |
| Food question using RAG | [Original UI answer](food_ui_recorded.md); [heading-only offline display update](food_display_review.md) | `assistant_20260914T185019_941207Z.json` | 1 Gemini; 0 MCP |
| Currency using the candidate’s MCP server/client | [Currency UI answer](currency_ui_recorded.md) | `assistant_20260914T182703_198768Z.json` | 2 Gemini; 1 currency |
| Required three-day combined plan | [Three-day display review](three_day_display_review.md) | `assistant_20260914T175904_893547Z.json` | 2 Gemini; 1 weather |
| Four-day update retaining earlier preferences | [Four-day display review](four_day_display_review.md) | `assistant_20260914T180330_534620Z.json` | 2 Gemini; 1 weather |

Keep the failed currency-provider report as optional error-handling evidence. The version 12 food-scope failure and synthetic test drafts are not successful demo answers. The final submission also needs the candidate’s Git delivery and video.
