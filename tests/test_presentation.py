"""Offline presentation regressions: exact source content and recorded tool results."""
import copy
import json
import unittest
from pathlib import Path

from src import assistant as a, assistant_response as ar

RUNS = Path(__file__).resolve().parents[1] / 'data/processed/assistant_runs'
THREE = RUNS / 'assistant_20260914T175904_893547Z.json'
FOUR = RUNS / 'assistant_20260914T180330_534620Z.json'
CURRENCY_ERROR = RUNS / 'assistant_20260914T180712_035232Z.json'


def evidence(pid, quote):
    return ar.Evidence(passage_id=pid, quote=quote)


def source(pid, text, section):
    return {'passage_id': pid, 'page_content': text,
            'metadata': {'title': 'Synthetic fixture', 'url': 'https://example.com/fixture',
                         'section_path': section}}


def synthetic_answer(activities, alternative=None):
    return ar.TravelAnswer.model_validate({
        'status': 'answered', 'knowledge_facts': [], 'suggestions': [],
        'assumptions': [], 'missing_information': [],
        'itinerary': [{'day': 1, 'date': '', 'activities': activities,
                       'indoor_alternative': alternative, 'weather_adjustment': ''}]})


class Presentation(unittest.TestCase):
    def test_overlap_preserves_all_qualifications_in_any_selection_order(self):
        first = 'The gallery is indoors. Children must be accompanied.'
        second = 'Children must be accompanied. Tickets are not included.'
        whole = 'The gallery is indoors. Children must be accompanied. Tickets are not included.'
        sources = {'P1': source('P1', whole, 'Gallery')}
        for quotes in [[first, second], [second, first], [second, first, first]]:
            selected = [evidence('P1', q) for q in quotes]
            before = copy.deepcopy(selected)
            self.assertEqual(ar.source_display_text(selected, sources), whole)
            self.assertEqual(selected, before)

    def test_disjoint_selections_mark_the_gap_and_keep_intentional_repetition(self):
        text = 'Guide says stop, stop. This unselected sentence stays out. Check access before visiting.'
        sources = {'P1': source('P1', text, 'Guide')}
        actual = ar.source_display_text([
            evidence('P1', 'Guide says stop, stop.'),
            evidence('P1', 'Check access before visiting.')], sources)
        self.assertEqual(actual, 'Guide says stop, stop. […] Check access before visiting.')
        self.assertNotIn('unselected', actual)

    def test_three_day_report_keeps_dates_tools_and_one_exploria_on_day_two(self):
        original = json.loads(THREE.read_text(encoding='utf-8-sig'))
        replay = a.replay_failure(THREE)
        text = replay['answer_markdown']
        day_two = text.split('**Day 2 —')[1].split('**Day 3 —')[0]
        self.assertEqual(day_two.count('Exploria —'), 1)
        self.assertIn('Optional indoor activity', day_two)
        self.assertNotIn('Optional, only if weather allows', day_two)
        self.assertEqual([d['date'] for d in replay['answer']['itinerary']],
                         [d['date'] for d in original['answer']['itinerary']])
        self.assertEqual(replay['tool_calls'], original['tool_calls'])
        self.assertEqual(replay['status'], 'answered')
        snow_day = text.split('**Day 1 —')[1].split('**Day 2 —')[0]
        self.assertEqual(snow_day.count('Snow City offers visitors a chance to experience winter.'), 1)

    def test_four_day_transport_is_guidance_and_all_main_activities_are_retained(self):
        original = json.loads(FOUR.read_text(encoding='utf-8-sig'))
        answer = ar.TravelAnswer.model_validate(original['answer'])
        before = answer.model_dump()
        text = ar.render_answer(answer, original['retrieved_passages'], original['tool_calls'])
        self.assertEqual(answer.model_dump(), before)
        self.assertEqual(text.count('**Main plan:**'), 4)
        self.assertNotIn('**Optional, only if weather allows:** Public Transport', text)
        self.assertEqual(text.count('### Getting around'), 1)
        self.assertEqual(text.count("Singapore's public transport system is fast and efficient."), 1)
        self.assertIn('Plan your journey with OneMap', text)
        replay = a.replay_failure(FOUR)
        self.assertEqual(replay['tool_calls'], original['tool_calls'])
        self.assertEqual(replay['conversation_context'], original['conversation_context'])
        self.assertEqual(replay['status'], 'answered')

    def test_failed_currency_keeps_partial_status_and_never_reuses_old_rates(self):
        original = json.loads(CURRENCY_ERROR.read_text(encoding='utf-8-sig'))
        replay = a.replay_failure(CURRENCY_ERROR)
        self.assertEqual(replay['status'], 'partial')
        self.assertEqual(replay['tool_calls'], original['tool_calls'])
        self.assertEqual(replay['answer']['itinerary'], [])
        self.assertIn('could not provide', replay['answer_markdown'])
        self.assertNotIn('132.70', replay['answer_markdown'])
        self.assertNotIn('Rate publication date:', replay['answer_markdown'])
        self.assertNotIn('Temperature range', replay['answer_markdown'])

    def test_alternative_with_an_additional_qualification_is_not_hidden(self):
        indoor = 'The gallery has indoor exhibitions.'
        caution = 'Check accessibility before visiting.'
        passages = [source('P1', indoor + ' ' + caution, 'Gallery')]
        main = {'text': 'Visit the gallery.', 'setting': 'indoor_sheltered', 'role': 'main',
                'evidence': [evidence('P1', indoor).model_dump()]}
        alternative = {'text': 'Gallery alternative.', 'evidence': [
            evidence('P1', indoor).model_dump(), evidence('P1', caution).model_dump()]}
        reviewed, _ = ar.review_recommendations(
            synthetic_answer([main], alternative), [], [], passages=passages)
        self.assertIsNotNone(reviewed.itinerary[0].indoor_alternative)
        text = ar.render_answer(reviewed, passages, [])
        self.assertIn('**Indoor alternative:**', text)
        self.assertIn(caution, text)

    def test_mixed_venue_and_transport_evidence_is_not_classified_as_general_guidance(self):
        indoor = 'The gallery has indoor exhibitions.'
        transport = 'The city has a public bus network.'
        passages = [source('P1', indoor, 'Gallery'), source('P2', transport, 'Public Transport')]
        main = {'text': 'Visit the gallery.', 'setting': 'indoor_sheltered', 'role': 'main',
                'evidence': [evidence('P1', indoor).model_dump()]}
        optional = {'text': 'Visit using the bus network.', 'setting': 'mixed',
                    'role': 'optional_if_weather_allows', 'evidence': [
                        evidence('P1', indoor).model_dump(), evidence('P2', transport).model_dump()]}
        text = ar.render_answer(synthetic_answer([main, optional]), passages, [])
        self.assertIn('**Optional, only if weather allows:**', text)
        self.assertNotIn('### Getting around', text)


if __name__ == '__main__':
    unittest.main()
