"""Offline regressions over real saved drafts and explicitly synthetic fixtures."""
import copy
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from langchain_core.messages import AIMessage
from src import assistant as a
from src import assistant_response as ar
from src.planning import effective_trip_days, may_balance_new_plan, confirmed_child_limit
from src.conversation_memory import ConversationMemory
from mcp_server.weather import checked_number, WeatherToolError
from mcp_server.currency import money_amount, CurrencyToolError

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'data/processed/assistant_runs'
LATEST = RUNS / 'assistant_20260914T162228_167494Z.json'
FOUR = RUNS / 'assistant_20260914T155317_681432Z.json'


def saved():
    return json.loads(LATEST.read_text(encoding='utf-8'))


def message(raw):
    return AIMessage(content='', tool_calls=[{'name': a.FINAL_FUNCTION, 'args': raw, 'id': 'offline'}])


def assess(raw, history=None, question=None):
    report = saved()
    question = question or report['question']
    return a.assess_final(message(raw), report['retrieved_passages'], report['tool_calls'],
                          [*(history or []), question], question, report['calendar'])


class RecordedFailures(unittest.TestCase):
    def test_latest_three_day_gap_fixed_without_new_evidence(self):
        before = LATEST.read_bytes()
        report = a.replay_failure(LATEST)
        self.assertEqual(report['status'], 'answered')
        self.assertEqual(len(report['answer']['itinerary']), 3)
        self.assertTrue(all(not day['gap_reason'] for day in report['answer']['itinerary']))
        self.assertTrue(all(any(p['role'] == 'main' for p in day['activities']) for day in report['answer']['itinerary']))
        self.assertEqual(report['model_calls'], 0)
        self.assertFalse(report['live_tools_called'])
        self.assertEqual(before, LATEST.read_bytes())
        original = saved()
        self.assertEqual(report['tool_calls'], original['tool_calls'])
        self.assertNotIn('some stalls may use dairy or eggs', report['answer_markdown'])
        self.assertTrue(any(c['action'] == 'balanced_existing_main_activity' for c in report['quality_review']['presentation_adjustments']))
        all_quotes = {v['quote'] for v in original['source_excerpt_catalog'].values()}
        for day in report['answer']['itinerary']:
            for activity in day['activities']:
                self.assertTrue(all(e['quote'] in all_quotes for e in activity['evidence']))

    def test_four_day_success_and_duplicate_backup(self):
        report = a.replay_failure(FOUR)
        self.assertEqual(report['status'], 'answered')
        self.assertEqual(len(report['answer']['itinerary']), 4)
        self.assertTrue(all(day['indoor_alternative'] is None for day in report['answer']['itinerary']))
        self.assertNotIn('skydiving', report['answer_markdown'].lower())

    def test_partial_without_explanation_keeps_supported_content(self):
        raw = saved()['validation_correction']['returned_draft'][0]
        check = assess(raw)
        self.assertIsNotNone(check['answer'], check['issues'])
        self.assertEqual(check['answer'].status, 'partial')
        self.assertTrue(check['answer'].missing_information)
        self.assertTrue(any('incomplete status' in x for x in check['issues']))

    def test_fixed_day_is_not_moved(self):
        raw = saved()['submitted_answer']
        question = saved()['question'] + ' Keep the activities on Day 1 unchanged.'
        check = assess(raw, question=question)
        self.assertEqual(check['answer'].status, 'partial')
        self.assertTrue(check['answer'].itinerary[2].gap_reason)
        self.assertFalse(any(c['action'] == 'balanced_existing_main_activity' for c in check['review']['presentation_adjustments']))

    def test_followup_is_not_rebalanced(self):
        check = assess(saved()['submitted_answer'], history=['Keep our earlier plan.'])
        self.assertEqual(check['answer'].status, 'partial')
        self.assertTrue(check['answer'].itinerary[2].gap_reason)

    def test_unknown_excerpt_rejected(self):
        raw = saved()['submitted_answer']
        raw['itinerary'][0]['activities'][0]['evidence'] = [{'excerpt_id': 'P999-E1'}]
        self.assertIsNone(assess(raw)['answer'])

    def test_uncited_main_rejected(self):
        raw = saved()['submitted_answer']
        raw['itinerary'][0]['activities'][0]['evidence'] = []
        self.assertIsNone(assess(raw)['answer'])

    def test_empty_optional_does_not_destroy_valid_answer(self):
        raw = saved()['submitted_answer']
        raw['itinerary'][0]['indoor_alternative'] = {'text': 'Not needed', 'evidence': []}
        self.assertIsNotNone(assess(raw)['answer'])

    def test_wrong_forecast_date_rejected(self):
        raw = saved()['submitted_answer']
        raw['itinerary'][0]['date'] = '2026-09-20'
        self.assertIsNone(assess(raw)['answer'])

    def test_tampered_record_provenance_rejected(self):
        report = saved()
        report['tool_calls'][0]['result']['provenance']['channel'] = 'invented'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tampered.json'; path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                a.replay_failure(path)

    def test_tampered_excerpt_catalog_rejected(self):
        report = saved()
        next(iter(report['source_excerpt_catalog'].values()))['quote'] = 'Invented source text.'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tampered.json'; path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'catalog'):
                a.replay_failure(path)


class GeneralPlanning(unittest.TestCase):
    def test_one_through_seven_days_are_dynamic(self):
        for count in range(1, 8):
            with self.subTest(count=count):
                passages = [{'passage_id': f'P{i}', 'page_content': f'Synthetic gallery {i} has indoor art displays.',
                             'metadata': {'title': f'Fixture {i}', 'url': 'https://example.com/fixture', 'section_path': f'Gallery {i}'}}
                            for i in range(1, count + 1)]
                raw = {'status': 'answered', 'knowledge_facts': [], 'suggestions': [], 'assumptions': [], 'missing_information': [], 'itinerary': []}
                for i in range(1, count + 1):
                    raw['itinerary'].append({'day': i, 'date': '', 'activities': [{'text': f'Visit gallery {i}.',
                        'setting': 'indoor_sheltered', 'role': 'main', 'evidence': [{'excerpt_id': f'P{i}-E1'}]}],
                        'indoor_alternative': None, 'weather_adjustment': ''})
                question = f'Create a {count}-day itinerary.'
                check = a.assess_final(message(raw), passages, [], [question], question, {})
                self.assertIsNotNone(check['answer'], check['issues'])
                self.assertEqual(len(check['answer'].itinerary), count)
                self.assertEqual(check['answer'].status, 'answered')

    def test_followup_count_from_history(self):
        self.assertEqual(effective_trip_days('Change the second day to indoors.', ['Plan a three-day trip.', 'Make it a four-day trip.']), 4)
        answer = a.replay_failure(FOUR)['answer']
        shorter = ar.TravelAnswer.model_validate(answer)
        shorter.itinerary.pop()
        with self.assertRaisesRegex(ValueError, '4-day'):
            a.validate_explicit_request(shorter, 'Change the second day to indoors.', saved()['calendar'], [], ['Plan a four-day trip.'])

    def test_explicit_length_change_wins(self):
        self.assertEqual(effective_trip_days('Make it five days.', ['Create a three-day trip.']), 5)
        self.assertIsNone(effective_trip_days('Three days or five days?', ['Plan a four-day trip.']))

    def test_balancing_conservative(self):
        self.assertTrue(may_balance_new_plan('Create a three-day trip next week.', []))
        for question in ['Plan a trip with a booking on Monday.', 'Plan a trip and keep Day 2 free.',
                         'Create a trip, visit a gallery after lunch.', 'Create a trip, repeat Snow City.']:
            self.assertFalse(may_balance_new_plan(question, []))
        self.assertFalse(may_balance_new_plan('Create a four-day trip.', ['Earlier request.']))

    def test_child_eligibility_and_adult_update(self):
        self.assertFalse(confirmed_child_limit(['We are a family.'], 12))
        self.assertTrue(confirmed_child_limit(['My child is 8.'], 12))
        self.assertFalse(confirmed_child_limit(['My child is 8.', 'We are travelling without children.'], 12))
        self.assertFalse(confirmed_child_limit(['Our child is 15.'], 12))

    def test_more_than_seven_requires_clarification(self):
        answer = ar.TravelAnswer(status='clarification_needed', knowledge_facts=[], suggestions=[], itinerary=[], assumptions=[], missing_information=['Choose a segment of up to seven days.'])
        a.validate_explicit_request(answer, 'Plan a ten-day trip.', {}, [])
        answer.status = 'answered'
        with self.assertRaises(ValueError): a.validate_explicit_request(answer, 'Plan a ten-day trip.', {}, [])

    def test_weather_not_invented_for_tool_failure(self):
        answer = ar.TravelAnswer(status='partial', knowledge_facts=[], suggestions=[], itinerary=[], assumptions=[], missing_information=['Forecast unavailable.'])
        text = ar.render_answer(answer, [], [{'tool': 'get_weather', 'status': 'error', 'result': None, 'error': 'Outside the forecast window.'}])
        self.assertIn('could not provide', text)
        self.assertNotIn('Temperature range', text)

    def test_missing_weather_remains_missing(self):
        record = {'tool':'get_weather','status':'success','result':{'forecast':[{'date':'2026-09-21','weather_code':None,'precipitation_probability_max':None,'precipitation_sum':None}]}}
        before = copy.deepcopy(record)
        ar.weather_planning_policy([record]); self.assertEqual(record, before)
        self.assertIsNone(checked_number(None, 'precipitation_probability_max'))
        with self.assertRaises(WeatherToolError): checked_number(110, 'precipitation_probability_max')

    def test_money_input_validation(self):
        self.assertEqual(str(money_amount(10000, 'INR')), '10000.00')
        for value in [-1, float('nan'), True, 1.234]:
            with self.assertRaises(CurrencyToolError): money_amount(value, 'INR')

    def test_memory_preserves_preferences_and_clears(self):
        memory = ConversationMemory()
        memory.record_turn('I am vegetarian. Plan a three-day trip.', 'Example answer.')
        memory.record_turn('Make it four days and use public transport.', 'Example update.')
        context = memory.context_for_prompt()
        self.assertEqual(len(context['user_requests_in_order']), 2)
        self.assertIn('vegetarian', context['user_requests_in_order'][0]['text'])
        memory.clear(); self.assertEqual(memory.turn_count, 0)


if __name__ == '__main__': unittest.main()
