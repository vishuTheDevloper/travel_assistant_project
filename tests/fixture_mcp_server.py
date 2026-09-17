"""TEST ONLY: real custom MCP handlers with a recorded weather provider.

This subprocess is selected only by test_integration.py. The production client
always starts mcp_server/server.py. No fixture switch exists in app.py or .env.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcp_server import server
from mcp_server.weather import WeatherToolError

SAVED = json.loads((ROOT / 'data/processed/assistant_runs/assistant_20260914T162228_167494Z.json').read_text(encoding='utf-8-sig'))

async def recorded_weather(city, start_date, end_date):
    result = copy.deepcopy(SAVED['tool_calls'][0]['result'])
    if city != 'Singapore' or start_date != result['requested_start_date'] or end_date != result['requested_end_date']:
        raise WeatherToolError('test_dates', 'This offline fixture only covers its recorded dates.')
    result.pop('provenance', None)
    return result

server.fetch_weather = recorded_weather
server.mcp.run(transport='stdio')
