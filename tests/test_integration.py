"""Real LangChain + MCP transport; SDK generation and weather are fixtures.

No external Gemini, weather, or currency request is made. These tests verify
protocol/control flow, not the future quality of generated Gemini answers.
"""
import copy
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from src import assistant as a, mcp_client
from src.conversation_memory import ConversationMemory
from src.tool_agent import execute_tool

ROOT = Path(__file__).resolve().parents[1]
# Saved reports are UTF-8, regardless of the Windows console/default encoding.
SAVED = json.loads((ROOT / 'data/processed/assistant_runs/assistant_20260914T162228_167494Z.json').read_text(encoding='utf-8-sig'))


def sdk_response(name, args):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role='model', parts=[types.Part(
            function_call=types.FunctionCall(name=name, args=args), thought_signature=b'offline-test-fixture')]),
        finish_reason='STOP')])


class Integration(unittest.IsolatedAsyncioTestCase):
    async def run_model_sequence(self, responses):
        model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', api_key='offline-test-placeholder', vertexai=False, max_tokens=6000, max_retries=0)
        sdk = AsyncMock(side_effect=responses)
        memory = ConversationMemory()
        parameters = mcp_client.server_parameters().model_copy(update={'args': [str(ROOT / 'tests/fixture_mcp_server.py')]})
        with tempfile.TemporaryDirectory() as diagnostic_dir, \
             patch.object(a, 'PROJECT_ROOT', Path(diagnostic_dir)), \
             patch.object(mcp_client, 'server_parameters', return_value=parameters), \
             patch.object(a, 'create_gemini_model', return_value=model), \
             patch.object(a, 'retrieve_context', return_value={'queries': ['recorded test'], 'passages': copy.deepcopy(SAVED['retrieved_passages']), 'planning_search': True}), \
             patch.object(a, 'calendar_context', return_value=copy.deepcopy(SAVED['calendar'])), \
             patch.object(model.async_client.models, 'generate_content', sdk):
            report = await a.answer_turn(SAVED['question'], memory)
        return report, memory, sdk

    async def test_combined_saved_draft_needs_two_requests_not_three(self):
        report, memory, sdk = await self.run_model_sequence([
            sdk_response('get_weather', SAVED['tool_calls'][0]['arguments']),
            sdk_response(a.FINAL_FUNCTION, SAVED['submitted_answer']),
        ])
        self.assertEqual(sdk.await_count, 2)
        self.assertEqual(report['model_calls'], 2)
        self.assertEqual(len(report['tool_calls']), 1)
        self.assertEqual(report['tool_calls'][0]['result'], SAVED['tool_calls'][0]['result'])
        self.assertEqual(report['status'], 'answered')
        self.assertFalse(report['validation_correction']['attempted'])
        self.assertEqual(memory.turn_count, 1)
        self.assertTrue(all(not d['gap_reason'] for d in report['answer']['itinerary']))

    async def test_invalid_citation_gets_one_correction_tools_not_repeated(self):
        bad = copy.deepcopy(SAVED['submitted_answer'])
        bad['itinerary'][0]['activities'][0]['evidence'] = [{'excerpt_id': 'P999-E1'}]
        report, memory, sdk = await self.run_model_sequence([
            sdk_response('get_weather', SAVED['tool_calls'][0]['arguments']),
            sdk_response(a.FINAL_FUNCTION, bad),
            sdk_response(a.FINAL_FUNCTION, SAVED['submitted_answer']),
        ])
        self.assertEqual(sdk.await_count, 3)
        self.assertEqual(len(report['tool_calls']), 1)
        self.assertTrue(report['validation_correction']['corrected_draft_used'])
        self.assertEqual(report['status'], 'answered')
        self.assertEqual(memory.turn_count, 1)

    async def test_both_drafts_invalid_returns_honest_tool_data(self):
        bad = copy.deepcopy(SAVED['submitted_answer'])
        bad['itinerary'][0]['activities'][0]['evidence'] = [{'excerpt_id': 'P999-E1'}]
        report, memory, sdk = await self.run_model_sequence([
            sdk_response('get_weather', SAVED['tool_calls'][0]['arguments']),
            sdk_response(a.FINAL_FUNCTION, bad), sdk_response(a.FINAL_FUNCTION, bad),
        ])
        self.assertEqual(sdk.await_count, 3)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['answer']['itinerary'], [])
        self.assertIn('no itinerary has been confirmed', report['answer_markdown'])
        self.assertNotIn('P999', report['answer_markdown'])
        self.assertEqual(memory.turn_count, 1)  # The truthful partial response is retained.

    async def test_real_mcp_currency_identity_and_invalid_weather(self):
        # Real production server, no provider network calls for these inputs.
        async with mcp_client.open_travel_tools() as tools:
            by_name = {t.name: t for t in tools}
            self.assertEqual(set(by_name), {'get_weather', 'convert_currency'})
            _, record = await execute_tool({'name': 'convert_currency', 'args': {'amount': 200, 'source_currency': 'SGD', 'target_currency': 'SGD'}, 'id': 'identity', 'type': 'tool_call'}, by_name)
            self.assertEqual(record['status'], 'success')
            self.assertEqual(record['result']['converted_amount'], '200.00')
            self.assertFalse(record['result']['external_rate_fetched'])
            _, record = await execute_tool({'name': 'get_weather', 'args': {'city': 'Singapore', 'start_date': '2099-01-01', 'end_date': '2099-01-03'}, 'id': 'far-future', 'type': 'tool_call'}, by_name)
            self.assertEqual(record['status'], 'error')
            self.assertIsNone(record['result'])

if __name__ == '__main__': unittest.main()
