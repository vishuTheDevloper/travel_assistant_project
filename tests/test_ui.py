"""Streamlit interaction checks with a fake backend; no model or tool calls."""
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from src import assistant

ROOT = Path(__file__).resolve().parents[1]

class Interface(unittest.TestCase):
    def test_submit_rerun_followup_failure_and_reset(self):
        calls = []
        async def fake_answer(question, memory, progress=None):
            calls.append((question, memory.context_for_prompt()))
            if progress: progress('Gemini request 1 of at most 3...')
            if question == 'test failure': raise RuntimeError('Synthetic failure')
            text = 'Offline UI fixture answer; this is not a generated itinerary.'
            memory.record_turn(question, text)
            return {'question': question, 'status': 'answered', 'answer_markdown': text,
                    'retrieved_passages': [], 'tool_calls': [], 'model_calls': 1,
                    'previous_turn_count': memory.turn_count - 1, 'model': 'offline-fixture',
                    'checks': {}, 'created_at_utc': '2026-09-14T00:00:00+00:00'}
        with patch.object(assistant, 'answer_turn', side_effect=fake_answer), \
             patch.object(assistant, 'save_report', return_value=Path('offline_ui_fixture.json')), \
             patch('dotenv.dotenv_values', return_value={'GOOGLE_API_KEY':'offline-placeholder', 'GEMINI_MODEL':'offline-fixture'}):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertEqual(calls, [])
            app.chat_input[0].set_value('I am vegetarian.').run()
            self.assertFalse(app.exception)
            self.assertEqual(len(calls), 1)
            app.run(); self.assertEqual(len(calls), 1)
            self.assertEqual(len(app.get('download_button')), 2)
            app.chat_input[0].set_value('Make it four days.').run()
            self.assertIn('vegetarian', calls[-1][1]['user_requests_in_order'][0]['text'])
            app.chat_input[0].set_value('test failure').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['travel_memory'].turn_count, 2)
            self.assertFalse(app.session_state['travel_busy'])
            next(b for b in app.button if b.label == 'New chat').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['travel_memory'].turn_count, 0)
            self.assertEqual(len(calls), 3)

if __name__ == '__main__': unittest.main()
