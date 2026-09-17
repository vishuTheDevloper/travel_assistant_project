"""Conservative scope checks for clear English information requests.

These rules restrict an unambiguous lookup; they do not select a live tool or
generate an answer. The LLM still interprets planning, live and ambiguous queries.
Earlier trip details remain available as preferences, but cannot turn a new
standalone information question into a dated itinerary.
"""
import re

if __package__:
    from .planning import DAY_COUNT
else:
    from planning import DAY_COUNT


LOOKUP_START = re.compile(
    r"^\s*(?:please\s+)?(?:what|which|where|who|why|how|tell me|explain|list|"
    r"recommend|suggest|show me|give me|(?:can|could|would) you(?: please)? "
    r"(?:recommend|suggest|list|tell|explain|show|give))\b", re.I)
PLAN_OR_UPDATE = re.compile(
    r"\b(?:itinerary|itineraries|schedule|day[- ]by[- ]day|plan|extend|shorten|"
    r"reschedule|replace|change|adjust|make (?:it|this))\b", re.I)
DATED_OR_DAY_REFERENCE = re.compile(
    r"\b(?:today|tomorrow|tonight|next week|this week|this weekend|next weekend|"
    r"(?:next|on) (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"day\s*\d+|(?:first|second|third|fourth|fifth|sixth|seventh|last) day)\b"
    r"|\b\d{4}-\d{2}-\d{2}\b", re.I)
WEATHER = re.compile(
    r"\b(?:weather|forecast|temperature|rain|raining|wind|storm|umbrella|hot|cold)\b", re.I)
HYPOTHETICAL = re.compile(
    r"\b(?:hypothetical|climate|typically|usually|rainy day|bad weather|"
    r"(?:when|if) it rains)\b", re.I)
CURRENCY = re.compile(
    r"\b(?:convert|conversion|currency|currencies|exchange rate|INR|SGD|USD|EUR|"
    r"GBP|AUD|CAD|CHF|JPY|rupees?|dollars?|euros?|pounds?)\b", re.I)


def request_scope(question: str) -> dict:
    """Constrain clear lookups; leave uncertain or live requests to the model.

    This is deliberately not a comprehensive multilingual intent classifier.
    The system prompt and response validation remain necessary.
    """
    information_only = bool(
        LOOKUP_START.search(question)
        and not PLAN_OR_UPDATE.search(question)
        and not DATED_OR_DAY_REFERENCE.search(question)
        and not DAY_COUNT.search(question)
        and not CURRENCY.search(question)
        and not (WEATHER.search(question) and not HYPOTHETICAL.search(question)))
    return {
        "mode": "destination_information" if information_only else "travel_request",
        "allow_itinerary": not information_only,
        "allow_live_tools": not information_only,
        "reason": ("Clear information question without a requested itinerary or live measurement/conversion."
                   if information_only else
                   "The model must interpret this planning, live or ambiguous request and select only necessary tools."),
    }
