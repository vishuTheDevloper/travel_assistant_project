"""Small, conservative itinerary checks; no model, network, or venue list.

The LLM interprets open-ended requests. These helpers enforce clear day counts
and permit balancing only a new, unconstrained itinerary, never an existing
conversation or a request that fixes an activity to a particular day.
"""

import re


NUMBER_WORDS = dict(enumerate((
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
)))
DAY_COUNT = re.compile(
    r"\b(\d+|" + "|".join(NUMBER_WORDS.values()) + r")(?:\s+|\s*[-\u2010-\u2014]\s*)days?\b", re.I
)


def requested_trip_days(question: str) -> int | None:
    words = {word: number for number, word in NUMBER_WORDS.items()}
    values = {int(value) if value.isdigit() else words[value.lower()]
              for value in DAY_COUNT.findall(question)}
    return next(iter(values)) if len(values) == 1 else None


def effective_trip_days(question: str, earlier_requests: list[str]) -> int | None:
    current = requested_trip_days(question)
    if current is not None:
        return current
    if DAY_COUNT.search(question):
        return None  # Conflicting counts in this question need interpretation.
    for previous in reversed(earlier_requests):
        if re.search(r"\b(?:itinerary|trip|plan|extend|shorten|make it)\b", previous, re.I):
            count = requested_trip_days(previous)
            if count is not None:
                return count
    return None


FIXED_SCHEDULE = re.compile(
    r"\b(?:day\s*\d+|(?:first|second|third|fourth|fifth|sixth|seventh|last)\s+day|"
    r"on\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"booked|booking|tickets?|reservation|appointment|must|keep|leave unchanged|"
    r"before|after|instead|replace|only|repeat|again)\b|\b\d{1,2}:\d{2}\b", re.I
)


def may_balance_new_plan(question: str, earlier_requests: list[str]) -> bool:
    return (not any(text.strip() for text in earlier_requests)
            and bool(re.search(r"\b(?:create|plan|suggest)\b", question, re.I))
            and bool(re.search(r"\b(?:itinerary|trip)\b", question, re.I))
            and not FIXED_SCHEDULE.search(question))


def confirmed_child_limit(user_requests: list[str], limit: int) -> bool:
    """Require an explicit eligible child age, not just the word 'family'.

    This supports common English phrasings. Ambiguous ages remain unconfirmed;
    the model can ask a clarification. A later adult-only update wins.
    """
    eligible = False
    for text in user_requests:
        if re.search(r"\b(?:without|no)\s+(?:children|kids)|\badults?[- ]only\b", text, re.I):
            eligible = False
        for pattern in (
            r"\b(?:child|children|kid|kids|son|daughter)(?:\s+(?:is|are|aged|age))?\s+(\d{1,2})\b",
            r"\b(\d{1,2})[- ]year[- ]old\s+(?:child|kid|son|daughter)\b",
        ):
            if any(0 <= int(age) <= limit for age in re.findall(pattern, text, re.I)):
                eligible = True
    return eligible
