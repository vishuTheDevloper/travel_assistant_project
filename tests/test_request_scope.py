"""Offline scope regression tests; SDK answers below are explicitly mocked.

The source-only draft is a test fixture derived from the saved food response,
not a new Gemini answer. Production code never uses this fixture to answer.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_google_genai import ChatGoogleGenerativeAI
from src import assistant as a, retriever
from src.conversation_memory import ConversationMemory
from src.request_scope import request_scope
from tests.test_integration import sdk_response

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'data/processed/assistant_runs/assistant_20260914T182920_354958Z.json'
FOOD = json.loads(REPORT.read_text(encoding='utf-8-sig'))


def source_only_fixture():
    draft = copy.deepcopy(FOOD['submitted_answer'])
    draft['itinerary'] = []
    draft['assumptions'] = []
    return draft


class ScopeRules(unittest.TestCase):
    def test_clear_information_requests_need_no_live_tools(self):
        for question in [FOOD['question'], 'Where can I try vegetarian food on my trip?',
                         'Explain public transport in Singapore.', 'List local customs.',
                         'Suggest some museums to visit.',
                         'Which ski resorts should I visit in Switzerland?',
                         'What indoor attractions are good on a rainy day?',
                         'Which attractions can I visit in bad weather?',
                         'Can you recommend some local dishes?']:
            with self.subTest(question=question):
                self.assertFalse(request_scope(question)['allow_live_tools'])
                self.assertFalse(request_scope(question)['allow_itinerary'])

    def test_live_planning_and_ambiguous_requests_are_left_to_the_model(self):
        for question in [a.DEFAULT_QUESTION,
                         'Extend this trip to four days and keep my earlier preferences.',
                         'What is the weather in Singapore tomorrow?',
                         'Should I carry an umbrella tomorrow?',
                         'How much is 10000 INR in SGD?',
                         'Convert 10000 INR to SGD.',
                         'Suggest a three-day itinerary.',
                         'What should we do on the second day instead?',
                         'What about next Monday?', 'Plan something for us.']:
            with self.subTest(question=question):
                self.assertTrue(request_scope(question)['allow_live_tools'])

    def test_saved_unrequested_food_itinerary_is_not_relabelled_as_a_passing_replay(self):
        before = REPORT.read_bytes()
        with self.assertRaisesRegex(ValueError, 'information question'):
            a.replay_failure(REPORT)
        self.assertEqual(REPORT.read_bytes(), before)

    def test_lookup_retrieval_does_not_expand_an_earlier_trip(self):
        memory = ConversationMemory()
        memory.record_turn('Plan a four-day trip; I am vegetarian.', 'Earlier plan fixture.')
        tokenizer = SimpleNamespace(encode=lambda q, **kwargs: q.split(), decode=lambda words, **kwargs: ' '.join(words))
        with patch.object(retriever, 'get_reranker', return_value=SimpleNamespace(tokenizer=tokenizer)), \
             patch.object(retriever, 'retrieve_passages', return_value=FOOD['retrieved_passages']):
            result = a.retrieve_context(FOOD['question'], memory)
        self.assertFalse(result['planning_search'])
        self.assertNotIn(a.INDOOR_QUERY, result['queries'])
        self.assertFalse(any('four-day' in q for q in result['queries']))
        self.assertTrue(any('vegetarian' in q for q in result['queries']))
        self.assertEqual(memory.turn_count, 1)


class ScopedModelFlow(unittest.IsolatedAsyncioTestCase):
    async def run_sequence(self, responses, *, history=(), question=None, passages=None):
        model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', api_key='offline-test-placeholder',
                                      vertexai=False, max_tokens=6000, max_retries=0)
        sdk = AsyncMock(side_effect=responses)
        memory = ConversationMemory()
        for request in history:
            memory.record_turn(request, 'Earlier assistant response fixture.')
        no_mcp = Mock(side_effect=AssertionError('MCP must not open for this lookup'))
        no_tool = AsyncMock(side_effect=AssertionError('No live tool should execute'))
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(a, 'PROJECT_ROOT', Path(tmp)), \
             patch.object(a, 'create_gemini_model', return_value=model), \
             patch.object(a, 'retrieve_context', return_value={
                 'queries': ['test lookup'], 'passages': copy.deepcopy(FOOD['retrieved_passages'] if passages is None else passages),
                 'planning_search': False}), \
             patch.object(a, 'calendar_context', return_value=copy.deepcopy(FOOD['calendar'])), \
             patch.object(a, 'open_travel_tools', no_mcp), \
             patch.object(a, 'execute_tool', no_tool), \
             patch.object(model.async_client.models, 'generate_content', sdk):
            report = await a.answer_turn(question or FOOD['question'], memory)
        no_mcp.assert_not_called()
        no_tool.assert_not_awaited()
        for invocation in sdk.call_args_list:
            config = invocation.kwargs['config']
            tools = config.tools if hasattr(config, 'tools') else config['tools']
            names = [f.name for tool in tools for f in tool.function_declarations]
            self.assertEqual(names, [a.FINAL_FUNCTION])
        self.assertEqual(report['tool_calls'], [])
        self.assertEqual(report['answer']['itinerary'], [])
        self.assertEqual(report['answer']['assumptions'], [])
        self.assertNotIn('Temperature range', report['answer_markdown'])
        self.assertNotIn('**Day 1', report['answer_markdown'])
        return report, memory, sdk

    async def test_food_uses_one_model_request_with_or_without_previous_context(self):
        for history in [[], ['Convert 10000 INR to SGD.'], ['Create a four-day trip next week.']]:
            with self.subTest(history=history):
                report, memory, sdk = await self.run_sequence(
                    [sdk_response(a.FINAL_FUNCTION, source_only_fixture())], history=history)
                self.assertEqual(sdk.await_count, 1)
                self.assertEqual(report['model_calls'], 1)
                self.assertEqual(report['status'], 'answered')
                self.assertFalse(report['validation_correction']['attempted'])
                self.assertEqual(report['previous_turn_count'], len(history))
                self.assertEqual(memory.turn_count, len(history) + 1)
                self.assertIn('Hainanese chicken rice', report['answer_markdown'])

    async def test_malformed_weather_call_is_blocked_before_any_mcp_execution(self):
        report, _, sdk = await self.run_sequence([
            sdk_response('get_weather', FOOD['tool_calls'][0]['arguments']),
            sdk_response(a.FINAL_FUNCTION, source_only_fixture())])
        self.assertEqual(sdk.await_count, 2)
        self.assertEqual(report['status'], 'answered')
        self.assertEqual(len(report['blocked_tool_calls']), 1)
        self.assertTrue(report['validation_correction']['attempted'])

    async def test_unrequested_itinerary_gets_one_source_only_correction(self):
        report, _, sdk = await self.run_sequence([
            sdk_response(a.FINAL_FUNCTION, FOOD['submitted_answer']),
            sdk_response(a.FINAL_FUNCTION, source_only_fixture())])
        self.assertEqual(sdk.await_count, 2)
        self.assertEqual(report['status'], 'answered')
        self.assertTrue(report['validation_correction']['attempted'])
        self.assertTrue(any('information question' in issue for issue in report['validation_correction']['initial_issues']))

    async def test_missing_destination_information_stays_an_honest_gap(self):
        missing = {'status': 'insufficient_information', 'knowledge_facts': [], 'suggestions': [],
                   'itinerary': [], 'assumptions': [], 'missing_information': [
                       'The saved sources cover Singapore, not Swiss ski resorts.']}
        report, _, sdk = await self.run_sequence([sdk_response(a.FINAL_FUNCTION, missing)],
                    question='Which ski resorts should I visit in Switzerland?', passages=[])
        self.assertEqual(sdk.await_count, 1)
        self.assertEqual(report['status'], 'insufficient_information')


if __name__ == '__main__':
    unittest.main()
