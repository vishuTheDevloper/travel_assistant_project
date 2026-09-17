"""Structured answers, evidence checks, and rendering for the combined assistant.

The model selects source excerpt IDs; Python inserts their exact text. Live numbers
are rendered directly from checked MCP results. These are structural checks;
a reviewer must still assess whether each quote supports the associated claim.
"""

import re
from copy import deepcopy
from datetime import date, timedelta
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

if __package__:
    from .planning import confirmed_child_limit, FIXED_SCHEDULE
else:
    from planning import confirmed_child_limit, FIXED_SCHEDULE


MAX_EXCERPT_CHARACTERS = 480


class Evidence(BaseModel):
    passage_id: str = Field(description="A current retrieved identifier, e.g. P1.")
    quote: str = Field(min_length=8, max_length=MAX_EXCERPT_CHARACTERS, description="Source text inserted by Python, never written by the model.")


class SupportedPoint(BaseModel):
    # Resolved internal text can contain several complete source excerpts.
    # The model-facing ReferencedPoint still limits generated text to 700 chars.
    text: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class PlannedActivity(SupportedPoint):
    setting: Literal["indoor_sheltered", "outdoor", "mixed", "unknown"] = Field(
        description="Use indoor_sheltered ONLY if a quoted source explicitly describes this activity as indoors, sheltered or air-conditioned."
    )
    role: Literal["main", "optional_if_weather_allows"] = Field(
        description="Main activity, or an optional activity conditional on suitable weather. Wet-day main activities must be indoor_sheltered."
    )


class ItineraryDay(BaseModel):
    day: int = Field(ge=1, le=7)
    date: str = Field(description="YYYY-MM-DD if dates are known; otherwise an empty string.")
    activities: list[PlannedActivity] = Field(max_length=3)
    gap_reason: str = Field(default="", max_length=500, description="Local review explanation when no supported main activity remains.")
    indoor_alternative: SupportedPoint | None = Field(description="A sourced indoor alternative, or null if no supporting passage.")
    weather_adjustment: str = Field(max_length=500, description="Explain the choice using this day's returned forecast, or empty if no forecast was obtained. Do not repeat numeric weather figures.")


class TravelAnswer(BaseModel):
    """Internal answer with source quotes resolved from selected excerpt IDs."""

    status: Literal["answered", "partial", "insufficient_information", "clarification_needed"]
    knowledge_facts: list[SupportedPoint] = Field(description="Concise source facts; every fact must pass evidence validation.")
    suggestions: list[SupportedPoint] = Field(max_length=4)
    itinerary: list[ItineraryDay] = Field(max_length=7)
    assumptions: list[str] = Field(max_length=4, description="Planning assumptions only, such as Monday-Wednesday dates; no uncited destination or live facts.")
    missing_information: list[str] = Field(description="Original limitations plus bounded itinerary review gaps; no unsupported travel recommendations.")


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    excerpt_id: str = Field(description="Select an exact excerpt_id from the current sources, for example P2-E1. Do not write or copy a quote.")


class ReferencedPoint(BaseModel):
    text: str = Field(min_length=1, max_length=700)
    evidence: list[EvidenceReference] = Field(min_length=1)


class ReferencedActivity(ReferencedPoint):
    setting: Literal["indoor_sheltered", "outdoor", "mixed", "unknown"] = Field(
        description="Use indoor_sheltered ONLY when the selected excerpt explicitly describes THIS activity as indoors, sheltered or air-conditioned."
    )
    role: Literal["main", "optional_if_weather_allows"] = Field(
        description="Wet-day main activities must have supported indoor_sheltered settings; other activities can only be conditional options."
    )


class ReferencedDay(BaseModel):
    day: int = Field(ge=1, le=7)
    date: str = Field(description="YYYY-MM-DD if dates are known; otherwise an empty string.")
    activities: list[ReferencedActivity] = Field(max_length=3, description="Sourced activities for this day. If none is supported, leave empty and explain the gap; never invent an activity to fill a required slot.")
    indoor_alternative: ReferencedPoint | None = Field(default=None, description="A different sourced indoor alternative, or null when the main plan is already sheltered or no alternative is supported.")
    weather_adjustment: str = Field(max_length=500, description="Explain the activity choice using this day's returned forecast; empty when there is no forecast. Do not repeat numeric weather figures.")


class SubmitTravelAnswer(BaseModel):
    """Submit the final answer using excerpt IDs as evidence. Python supplies
    exact quotations from those IDs. This is a LOCAL formatting function, not
    an external/MCP tool. Wait for live tool results before submitting. Keep
    unused lists empty and never invent excerpt IDs.
    """
    status: Literal["answered", "partial", "insufficient_information", "clarification_needed"]
    knowledge_facts: list[ReferencedPoint] = Field(description="Prefer at most six useful facts. This is a presentation preference; every supplied fact requires valid evidence.")
    suggestions: list[ReferencedPoint] = Field(max_length=4)
    itinerary: list[ReferencedDay] = Field(max_length=7)
    assumptions: list[str] = Field(max_length=4, description="Planning assumptions only; no uncited destination or live facts.")
    missing_information: list[str] = Field(max_length=4, description="Gaps or clarification questions only; no unsupported recommendations.")


def build_excerpt_catalog(passages: list[dict]) -> dict[str, dict]:
    """Deterministic source excerpts, with no model calls or source rewriting.

    Adjacent sentences are kept together when possible so a venue name and its
    indoor description remain connected. Only whitespace is normalized. Long
    sentences are split at word boundaries, never shortened with ellipses.
    These are citation excerpts from retrieved chunks, not a new vector index.
    """
    catalog = {}
    for passage in passages:
        raw = passage["page_content"]
        if raw.startswith("Destination:") and "\n\n" in raw:
            raw = raw.split("\n\n", 1)[1]
        text = " ".join(raw.split())
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\u201c])', text)
        pieces = []
        for sentence in sentences:
            remaining = sentence.strip()
            while len(remaining) > MAX_EXCERPT_CHARACTERS:
                end = remaining.rfind(" ", 0, MAX_EXCERPT_CHARACTERS + 1)
                if end <= 0:
                    end = MAX_EXCERPT_CHARACTERS
                pieces.append(remaining[:end])
                remaining = remaining[end:].lstrip()
            if remaining:
                pieces.append(remaining)
        seen = set()
        for i, piece in enumerate(pieces):
            quote = piece
            if i + 1 < len(pieces) and len(piece) + len(pieces[i + 1]) + 1 <= MAX_EXCERPT_CHARACTERS:
                candidate = piece + " " + pieces[i + 1]
                if candidate in text:
                    quote = candidate
            if quote in seen or len(quote) < 8 or len(quote.split()) < 3:
                continue
            seen.add(quote)
            key = f"{passage['passage_id']}-E{len(seen)}"
            catalog[key] = {"passage_id": passage["passage_id"], "quote": quote}
    return catalog


def normalize_submission(raw: dict) -> tuple[dict, list[str]]:
    """Withhold uncited optional alternatives by structure, never by wording.

    No uncited alternative is displayed or treated as supported. Main activities
    still require citations, and weather validation still checks their settings.
    Nonempty but invalid evidence still fails. Original input remains in the audit.
    """
    data = deepcopy(raw)
    notes = []
    if not isinstance(data, dict):
        return data, notes
    # Preserve the model's declared gaps and label the answer consistently.
    # This never makes an incomplete answer complete or changes its evidence.
    missing = data.get("missing_information")
    if data.get("status") == "answered" and isinstance(missing, list) and missing:
        data["status"] = "partial"
        notes.append("Status changed to partial because the submitted answer declares missing information.")
    if data.get("status") in {"partial", "insufficient_information", "clarification_needed"} and missing == []:
        data["missing_information"] = [
            "The model marked this answer incomplete without identifying the missing detail. "
            "Please clarify what you would like expanded; completeness has not been confirmed."
        ]
        notes.append("An unexplained incomplete status was retained with an explicit limitation, rather than discarding supported content.")
    days = data.get("itinerary")
    for day in days if isinstance(days, list) else []:
        if not isinstance(day, dict):
            continue
        alternative = day.get("indoor_alternative")
        if not isinstance(alternative, dict) or alternative.get("evidence") not in (None, []):
            continue
        day["indoor_alternative"] = None
        notes.append(f"Day {day.get('day')}: uncited optional alternative withheld; original text retained in submitted_answer.")
    return data, notes


def resolve_submission(submission: SubmitTravelAnswer, passages: list[dict]) -> TravelAnswer:
    """Resolve only exact IDs. Unknown IDs and unsupported claims still fail."""
    catalog = build_excerpt_catalog(passages)
    payload = submission.model_dump()
    points = [*payload["knowledge_facts"], *payload["suggestions"]]
    for day in payload["itinerary"]:
        points.extend(day["activities"])
        if day["indoor_alternative"]:
            points.append(day["indoor_alternative"])
    for point in points:
        evidence = []
        for reference in point["evidence"]:
            key = reference["excerpt_id"]
            if key not in catalog:
                raise ValueError(f"Unknown source excerpt ID: {key}. Select an ID from the current retrieved sources.")
            evidence.append(dict(catalog[key]))
        point["evidence"] = evidence
    return TravelAnswer.model_validate(payload)


WET_WEATHER_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99}
INDOOR_WORDS = re.compile(r"\b(?:indoors?|sheltered|air[\s\-\u2010-\u2014]+conditioned)\b", re.I)


def weather_planning_policy(records: list[dict]) -> dict:
    """Application preference, NOT a provider warning or a meteorological rule.

    Prefer sheltered main activities when rain probability is at least 60%,
    precipitation is at least 5 mm, or the returned condition code indicates
    precipitation/a thunderstorm. Otherwise allow a flexible plan. Missing
    values never become zero, and no daily value predicts exact rain hours.
    """
    days = {}
    for record in records:
        if record["status"] != "success" or record["tool"] != "get_weather":
            continue
        for row in record["result"]["forecast"]:
            probability = row.get("precipitation_probability_max")
            precipitation = row.get("precipitation_sum")
            wet = (
                (probability is not None and probability >= 60)
                or (precipitation is not None and precipitation >= 5)
                or row.get("weather_code") in WET_WEATHER_CODES
            )
            days[row["date"]] = "sheltered_main_plan" if wet else "flexible_plan"
    return {
        "origin": "Application planning preference; not an Open-Meteo rule or a guarantee of rain.",
        "rule": "For sheltered_main_plan dates, use sourced indoor/sheltered MAIN activities. Outdoor, mixed or unknown settings may only be optional_if_weather_allows. If sheltered options lack evidence, return a partial answer and explain the gap.",
        "days": days,
    }


def has_indoor_evidence(point: SupportedPoint) -> bool:
    # An explicit descriptor is required, but its presence alone cannot prove
    # that it describes the named activity. Semantic review is still necessary.
    return any(INDOOR_WORDS.search(e.quote) for e in point.evidence)


def all_points(answer: TravelAnswer):
    yield from answer.knowledge_facts
    yield from answer.suggestions
    for day in answer.itinerary:
        yield from day.activities
        if day.indoor_alternative:
            yield day.indoor_alternative


def explicit_diet(user_requests: list[str]) -> str | None:
    """Read simple explicit diet statements in order, with later updates winning.

    This local check supplements the full conversation supplied to the LLM; it
    is not a general natural-language preference parser or allergy checker.
    """
    diet = None
    pattern = re.compile(
        r"\b(?:i am|i'm|we are|we're|my diet is|our diet is)\s+"
        r"(?:(not|no longer)\s+)?(?:a\s+)?(vegetarian|vegan)\b", re.I
    )
    for request in user_requests:
        for match in pattern.finditer(request.replace("\u2019", "'")):
            diet = None if match.group(1) else match.group(2).lower()
    return diet


def diet_conflicts(text: str, diet: str | None) -> list[str]:
    """Conservative checks for explicit animal-food recommendations.

    Negative ingredient warnings, named meat substitutes and animal sightseeing
    are not treated as recommendations to eat animal products. This finite list
    cannot establish that every dish is suitable; semantic review is still needed.
    """
    if diet not in {"vegetarian", "vegan"}:
        return []
    if not re.search(r"\b(?:eat|eating|food|meals?|dining|dishes|dish|curry|rice|tuck|taste|order|lunch|dinner|breakfast|seafood)\b", text, re.I):
        return []
    terms = r"fish|chicken|beef|pork|mutton|lamb|duck|bacon|ham|seafood|prawns?|shrimp|crabs?|cockles?|oysters?|lard|meat"
    if diet == "vegan":
        terms += r"|eggs?|milk|cheese|paneer|butter|ghee|yogurt|yoghurt|honey"
    conflicts = []
    for clause in re.split(r"[.!?;]|\bbut\b", text, flags=re.I):
        for match in re.finditer(r"\b(?:" + terms + r")\b", clause, re.I):
            prefix = clause[:match.start()]
            if re.search(r"\b(?:avoid|without|exclude|do not|don't|not suitable|not vegetarian|not vegan)\b", prefix, re.I):
                continue
            if re.search(r"\b(?:is|are)\s+not\s+(?:vegetarian|vegan)\b", clause[match.end():], re.I):
                continue
            local_prefix = re.split(r"[,;]|\b(?:and|or)\b", prefix, flags=re.I)[-1]
            if re.search(r"\b(?:mock|imitation|plant-based|meat-free|meatless|vegetarian|vegan)\s+(?:\w+\s+){0,2}$", local_prefix, re.I):
                continue
            if re.match(r"\s+(?:imitations?|substitutes?)\b", clause[match.end():], re.I):
                continue
            conflicts.append(match.group().lower())
    return sorted(set(conflicts))


def review_recommendations(answer: TravelAnswer, records: list[dict],
                           user_requests: list[str], *, passages: list[dict] | None = None,
                           allow_balancing: bool = False) -> tuple[TravelAnswer, dict]:
    """Withhold unsuitable recommendations; never invent substitute facts.

    Use only an already supplied, supported alternative when replacing a main
    activity. Keep a visible gap when none is available. The final strict source,
    date and weather validation still runs after this review.
    """
    result = answer.model_copy(deep=True)
    diet = explicit_diet(user_requests)
    policy = weather_planning_policy(records)["days"]
    changes = []
    presentation = []
    source_map = {p["passage_id"]: p["metadata"] for p in (passages or [])}

    def evidence_key(point):
        # Different excerpts from the same named section are not different venues.
        keys = []
        for evidence in point.evidence:
            metadata = source_map.get(evidence.passage_id, {})
            section = metadata.get("section_path")
            keys.append((metadata.get("source_id", evidence.passage_id), section)
                        if section else (evidence.passage_id, evidence.quote))
        return tuple(sorted(set(keys)))

    def use_source_text(point, section, day=None):
        text = " ".join(dict.fromkeys(e.quote for e in point.evidence))
        if point.text != text:
            presentation.append({"action": "used_verbatim_selected_excerpts", "section": section,
                                 "day": day, "original_point": point.model_dump(), "displayed_text": text})
            point.text = text

    # Facts are extractive: display the first exact, model-selected source
    # excerpt. Keep the original paraphrase in submitted_answer for inspection.
    # Generated activity choices are checked before source descriptions replace
    # their paraphrases. The original submission remains available in the audit.
    for point in result.knowledge_facts:
        use_source_text(point, "knowledge_facts")
    for point in result.suggestions:
        use_source_text(point, "suggestions")

    def usable(point, section, day=None, require_indoor=False):
        reasons = []
        if (require_indoor or INDOOR_WORDS.search(point.text)) and not has_indoor_evidence(point):
            reasons.append("The cited excerpt does not explicitly establish the claimed indoor/sheltered setting.")
        foods = diet_conflicts(point.text, diet)
        if foods:
            reasons.append(f"The recommendation contains animal foods that conflict with the explicit {diet} preference: {', '.join(foods)}.")
        # Preserve explicit child-age limits in recommendations. This limited
        # pattern is a guard for source wording, not an age eligibility engine.
        for evidence in point.evidence:
            restriction = re.search(r"\b(?:kids|children)(?: aged)?\s+(\d{1,2})\s+and\s+under\b", evidence.quote, re.I)
            if restriction and not confirmed_child_limit(user_requests, int(restriction.group(1))):
                reasons.append("This children's activity requires an eligible child whose age has not been confirmed: " + restriction.group(0) + ".")
        if reasons:
            changes.append({"action": "withheld", "section": section, "day": day,
                            "original_point": point.model_dump(), "reasons": reasons})
            return False
        return True

    result.suggestions = [p for p in result.suggestions if usable(p, "suggestions")]
    for day in result.itinerary:
        before = len(changes)
        kept = []
        for activity in day.activities:
            use_source_text(activity, "activities", day.day)
            if not usable(activity, "activities", day.day, activity.setting == "indoor_sheltered"):
                continue
            if policy.get(day.date) == "sheltered_main_plan" and activity.role == "main" and activity.setting != "indoor_sheltered":
                changes.append({"action": "withheld", "section": "activities", "day": day.day,
                                "original_point": activity.model_dump(),
                                "reasons": ["This wet-day main activity does not have a supported sheltered setting."]})
                continue
            kept.append(activity)
        day.activities = kept
        if day.indoor_alternative:
            use_source_text(day.indoor_alternative, "indoor_alternative", day.day)
        if day.indoor_alternative and not usable(day.indoor_alternative, "indoor_alternative", day.day, True):
            day.indoor_alternative = None
        if not any(p.role == "main" for p in kept):
            candidate = next((p for p in kept if p.setting == "indoor_sheltered" and has_indoor_evidence(p)), None)
            if candidate is not None:
                candidate.role = "main"
                changes.append({"action": "promoted_existing_option", "day": day.day, "text": candidate.text})
            elif day.indoor_alternative:
                alternative = day.indoor_alternative
                for extra in day.activities[2:]:
                    changes.append({"action": "withheld", "section": "activities", "day": day.day,
                                    "original_point": extra.model_dump(),
                                    "reasons": ["Keep at most three activities when adding the supported main alternative."]})
                day.activities = day.activities[:2]
                day.activities.insert(0, PlannedActivity(**alternative.model_dump(), setting="indoor_sheltered", role="main"))
                day.indoor_alternative = None
                changes.append({"action": "used_existing_indoor_alternative", "day": day.day, "text": alternative.text})
            else:
                day.gap_reason = "No supported main activity remains for this date. A suitable venue still needs to be confirmed."
                changes.append({"action": "main_activity_gap", "day": day.day, "reason": day.gap_reason})
        if len(changes) > before and day.date in policy:
            day.weather_adjustment = (
                "Given the wet forecast, the remaining main plan uses sourced sheltered options; any outdoor choices are conditional."
                if policy[day.date] == "sheltered_main_plan" and not day.gap_reason else
                "The forecast is shown above. A supported main activity still needs to be confirmed for this date."
                if day.gap_reason else
                "Use the supported activities shown here and check the forecast again before travelling."
            )
    # Cover an empty day using a spare activity the model already selected.
    # This is a schedule allocation rule, not retrieval or a canned itinerary.
    # It is disabled for follow-ups and day-specific requests. Time-qualified
    # source excerpts are never moved. The donor keeps at least one main plan.
    if allow_balancing:
        earlier_main_keys = set()
        for target in result.itinerary:
            target_mains = [p for p in target.activities if p.role == "main"]
            only_repeated = bool(target_mains) and all(evidence_key(p) in earlier_main_keys for p in target_mains)
            earlier_main_keys.update(evidence_key(p) for p in target_mains)
            if target_mains and not only_repeated:
                continue
            candidate_pair = None
            for donor in result.itinerary:
                mains = [p for p in donor.activities if p.role == "main"]
                if donor is target or len(mains) < 2:
                    continue
                for candidate in reversed(mains):
                    if FIXED_SCHEDULE.search(candidate.text) or re.search(
                        r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
                        r"January|February|March|April|May|June|July|August|September|October|November|December|"
                        r"open|closed|weekends?|season|daily at|\d{4}-\d{2}-\d{2})\b", candidate.text, re.I
                    ):
                        continue
                    if policy.get(target.date) == "sheltered_main_plan" and candidate.setting != "indoor_sheltered":
                        continue
                    if any(evidence_key(p) == evidence_key(candidate)
                           for day in result.itinerary if day is not donor
                           for p in day.activities if p.role == "main"):
                        continue
                    candidate_pair = donor, candidate
                    break
                if candidate_pair:
                    break
            if candidate_pair:
                donor, candidate = candidate_pair
                donor.activities.remove(candidate)
                if only_repeated:
                    target.activities = [p for p in target.activities if p.role != "main"]
                target.activities.insert(0, candidate)
                earlier_main_keys.add(evidence_key(candidate))
                target.activities = target.activities[:3]
                target.gap_reason = ""
                presentation.append({"action": "balanced_existing_main_activity", "from_day": donor.day,
                                     "to_day": target.day, "evidence": [e.model_dump() for e in candidate.evidence]})
                changes = [c for c in changes if not (c["action"] == "main_activity_gap" and c.get("day") == target.day)]

    main_keys = [(day.day, evidence_key(p)) for day in result.itinerary for p in day.activities if p.role == "main"]
    for day in result.itinerary:
        retained = []
        for point in day.activities:
            if point.role != "main" and any(other_day != day.day and key == evidence_key(point) for other_day, key in main_keys):
                presentation.append({"action": "omitted_repeated_optional_evidence", "day": day.day,
                                     "original_point": point.model_dump()})
            else:
                retained.append(point)
        day.activities = retained
        alternative = day.indoor_alternative
        # A shared venue heading is not enough: the alternative may contain an
        # extra qualification. Omit only evidence already present this day.
        if alternative and all(any(
                e.passage_id == other.passage_id
                and " ".join(e.quote.split()) in " ".join(other.quote.split())
                for p in day.activities for other in p.evidence)
                for e in alternative.evidence):
            presentation.append({"action": "omitted_repeated_indoor_alternative", "day": day.day,
                                 "original_point": alternative.model_dump()})
            day.indoor_alternative = None

        # The model chooses which cited activity goes on which day. Its chosen
        # source excerpts supply the factual description, so generated details
        # cannot be added to that description. Preserve all distinct excerpts
        # (including qualifications), rather than keeping only the first one.
        points = [*day.activities]
        if day.indoor_alternative:
            points.append(day.indoor_alternative)
        for point in points:
            source_text = " ".join(dict.fromkeys(e.quote for e in point.evidence))
            if point.text != source_text:
                presentation.append({"action": "used_verbatim_activity_excerpts", "day": day.day,
                                     "original_point": point.model_dump(), "displayed_text": source_text})
                point.text = source_text
        # Explain the actual retained plan using the current day's policy.
        # This avoids keeping an explanation for an activity that was removed,
        # and does not paraphrase temperatures or invent rain hours.
        if day.date in policy:
            day.weather_adjustment = (
                "A supported main activity still needs to be confirmed for this date."
                if day.gap_reason else
                "The forecast favours a sheltered main plan. Keep any outdoor options conditional on the weather."
                if policy[day.date] == "sheltered_main_plan" else
                "The returned forecast allows a flexible plan; recheck conditions before outdoor activities."
            )
    if changes:
        # Removing an unsupported draft option need not make a new, broad plan
        # incomplete when all its days now have supported main activities.
        # For constrained requests/follow-ups we retain the explicit limitation.
        resolved_broad_plan = (allow_balancing and bool(result.itinerary)
                               and all(any(p.role == "main" for p in d.activities) for d in result.itinerary))
        gaps = ([] if resolved_broad_plan else
                ["Some proposed activities were removed because their cited evidence did not support the claimed setting, eligible children were not confirmed, or they conflicted with your stated dietary preference."]
                if any(c["action"] == "withheld" for c in changes) else
                ["Some proposed day plans required adjustments before they could be used; the itinerary remains partial."])
        if not resolved_broad_plan and any(c["action"] == "used_existing_indoor_alternative" for c in changes):
            gaps.append("Some days use the already suggested general indoor alternatives. Specific replacement venues, opening hours and transfers still need to be confirmed.")
        if any(d.gap_reason for d in result.itinerary):
            gaps.append("A supported main activity is still missing for one or more itinerary days.")
        result.missing_information.extend(g for g in gaps if g not in result.missing_information)
        has_supported_information = bool(list(all_points(result))) or any(r["status"] == "success" for r in records)
        if gaps:
            result.status = "partial" if has_supported_information else "insufficient_information"
    if not any(re.search(r"\b(?:repeat|again|revisit|same venue)\b", q, re.I) for q in user_requests):
        seen = set()
        for day in result.itinerary:
            keys = {evidence_key(p) for p in day.activities if p.role == "main"}
            if keys and keys.issubset(seen):
                note = f"Day {day.day} repeats an earlier main activity; a different suitable activity has not been confirmed."
                result.missing_information.append(note)
                result.status = "partial"
                changes.append({"action": "repeated_main_activity", "day": day.day, "reason": note})
            seen.update(keys)
    result = TravelAnswer.model_validate(result.model_dump())
    return result, {"explicit_diet": diet, "adjustments": changes,
                    "presentation_adjustments": presentation,
                    "semantic_review_still_required": True,
                    "method": "Exact selected excerpts supply factual descriptions. Unsuitable points are withheld. Existing flexible main activities may be balanced only in a new unconstrained plan; no new venue is invented. No extra model request."}


# Answer validation
# Check citation quotes against retrieved passages and align itinerary dates with forecasts.
# Require evidence for indoor claims and label missing information explicitly.
# These checks verify structure and references; they do not prove every interpretation.
def validate_answer(answer: TravelAnswer, passages: list[dict], records: list[dict]) -> None:
    sources = {p["passage_id"]: p for p in passages}
    successes = [r for r in records if r["status"] == "success"]
    if answer.status == "answered" and (answer.missing_information or any(r["status"] != "success" for r in records)):
        raise ValueError("A response with missing information or a failed tool must not claim to be fully answered.")
    if answer.status != "answered" and not any(s.strip() for s in answer.missing_information):
        raise ValueError("Explain the missing information or ask a clarification question.")
    points = list(all_points(answer))
    if answer.status == "answered" and not (points or successes):
        raise ValueError("An answered response needs sourced information or successful tool data.")
    if answer.status == "insufficient_information" and (points or successes):
        raise ValueError("Use partial when some supported information is available.")
    for point in points:
        if not point.text.strip():
            raise ValueError("An answer point cannot be blank.")
        for evidence in point.evidence:
            passage = sources.get(evidence.passage_id)
            quote = " ".join(evidence.quote.split())
            if passage is None or len(quote.split()) < 3 or quote not in " ".join(passage["page_content"].split()):
                raise ValueError("A citation or evidence quote does not match a retrieved passage.")
    # URLs are inserted only by the renderer from actual source metadata.
    # An exact source excerpt may itself contain a URL. Reject model-written
    # URLs in free text, not genuine source text inserted by the application.
    texts = [p.text for p in points if p.text != " ".join(dict.fromkeys(e.quote for e in p.evidence))]
    texts += answer.assumptions + answer.missing_information
    texts += [d.weather_adjustment for d in answer.itinerary]
    if any(re.search(r"https?://", text, re.I) for text in texts):
        raise ValueError("Answer text must not supply its own source URLs.")
    if any(not text.strip() or len(text) > 700 for text in answer.assumptions + answer.missing_information):
        raise ValueError("Assumptions and limitations must be short, nonempty statements.")
    forecast_dates = {
        row["date"]
        for r in successes if r["tool"] == "get_weather"
        for row in r["result"]["forecast"]
    }
    planning_days = weather_planning_policy(records)["days"]
    dates = []
    for number, day in enumerate(answer.itinerary, start=1):
        if day.day != number:
            raise ValueError("Itinerary days must be sequential, starting at day 1.")
        if day.date:
            parsed = date.fromisoformat(day.date)
            if parsed.isoformat() != day.date:
                raise ValueError("Use YYYY-MM-DD itinerary dates.")
            dates.append(parsed)
        if day.weather_adjustment and day.date not in forecast_dates:
            raise ValueError("A weather adjustment needs an MCP forecast for the same day.")
        if forecast_dates and day.date not in forecast_dates:
            raise ValueError("The itinerary dates do not match the returned forecast dates.")
        main = [activity for activity in day.activities if activity.role == "main"]
        if not main and day.gap_reason and answer.status != "answered":
            pass  # Explicit partial-plan gap, not an invented activity.
        elif not 1 <= len(main) <= 2:
            raise ValueError("Use one or two main activities per day; keep other choices optional.")
        for activity in day.activities:
            if activity.setting == "indoor_sheltered" and not has_indoor_evidence(activity):
                raise ValueError(f"Day {day.day}: indoor/sheltered activity '{activity.text[:100]}' needs an excerpt explicitly describing that activity as indoor, sheltered or air-conditioned.")
        if day.indoor_alternative and not has_indoor_evidence(day.indoor_alternative):
            raise ValueError("An indoor alternative needs explicit indoor/sheltered evidence.")
        if planning_days.get(day.date) == "sheltered_main_plan":
            if any(activity.setting != "indoor_sheltered" for activity in main):
                raise ValueError("The wet-day main plan must use supported sheltered activities; outdoor options must be conditional alternatives.")
    if dates and (len(dates) != len(answer.itinerary) or any(b != a + timedelta(days=1) for a, b in zip(dates, dates[1:]))):
        raise ValueError("Dated itineraries must use consecutive dates for every day.")


def source_link(title: str, url: str | None) -> str:
    title = str(title).replace("[", "(").replace("]", ")")
    if url and urlsplit(url).scheme in {"https", "http"} and not any(c in url for c in "<>\n\r"):
        return f"[{title}](<{url}>)"
    return title


def render_live_data(records: list[dict]) -> list[str]:
    """Copy checked tool values into the answer, without model arithmetic."""
    if not records:
        return []
    lines = ["### Current information from MCP tools", ""]
    value = lambda x: "Unavailable" if x is None else str(x)
    for record in records:
        if record["status"] != "success":
            lines.extend([f"- `{record['tool']}` could not provide the requested information. No substitute data was invented.", ""])
            continue
        data = record["result"]
        if record["tool"] == "get_weather":
            lines.extend([
                "**Weather — " + source_link(data["provider"], data.get("source_url")) + "**",
                f"Forecast location: {data['city']} ({data['timezone']}).",
                "", "| Date | Conditions | Temperature range | Rain probability |",
                "| --- | --- | --- | --- |",
            ])
            for row in data["forecast"]:
                lines.append(f"| {row['date']} | {value(row.get('condition'))} | {value(row.get('temperature_2m_min'))} to {value(row.get('temperature_2m_max'))} °C | {value(row.get('precipitation_probability_max'))}{'%' if row.get('precipitation_probability_max') is not None else ''} |")
            current = data.get("current")
            if current:
                lines.extend(["", f"Current model estimate at {current['time']}: {value(current.get('condition'))}, {value(current.get('temperature_2m'))} °C. This snapshot is separate from future forecasts."])
            lines.extend(["", f"Retrieved: {data['fetched_at_utc']}", "Forecasts can change. Daily rain probabilities do not specify rain hours or mean rain all day. Unavailable values are not zero.", ""])
        elif record["tool"] == "convert_currency":
            lines.extend([
                f"**Currency: {data['amount']} {data['source_currency']} = {data['converted_amount']} {data['target_currency']}**",
                f"- Rate: 1 {data['source_currency']} = {data['exchange_rate']} {data['target_currency']}",
            ])
            if data.get("external_rate_fetched"):
                lines.extend([
                    f"- Rate publication date: {data['rate_date']}",
                    "- Source: " + source_link(data["rate_source"], data.get("source_url")) + " via " + source_link(data["provider"], data.get("documentation_url")),
                    "- Daily reference-rate estimate; bank fees and provider margins are excluded.",
                ])
            else:
                lines.append("- Same currency: identity conversion; no external rate was fetched.")
            lines.append("")
    return lines


def selected_source_spans(evidence: list[Evidence], sources: dict) -> list[tuple[str, int, int]]:
    """Merge only overlapping selections in the same source; retain every selected word.

    Source positions, rather than sentence splitting, preserve qualifications,
    abbreviations and repeated phrases that genuinely occur within one excerpt.
    """
    grouped = {}
    for item in evidence:
        source = " ".join(sources[item.passage_id]["page_content"].split())
        quote = " ".join(item.quote.split())
        start = source.find(quote)
        if start < 0:
            raise ValueError("A selected excerpt does not match its source.")
        grouped.setdefault(item.passage_id, []).append((start, start + len(quote)))
    result = []
    for pid, spans in grouped.items():
        merged = []
        for start, end in sorted(set(spans)):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        result.extend((pid, start, end) for start, end in merged)
    return result


def source_display_text(evidence: list[Evidence], sources: dict) -> str:
    spans = selected_source_spans(evidence, sources)
    # A marked gap keeps separate quotations from looking like continuous text.
    return " […] ".join(" ".join(sources[pid]["page_content"].split())[start:end]
                        for pid, start, end in spans)


def spans_covered(spans: list[tuple], shown: list[tuple]) -> bool:
    return bool(spans) and all(any(pid == old_pid and old_start <= start and end <= old_end
                                  for old_pid, old_start, old_end in shown)
                              for pid, start, end in spans)


def render_answer(answer: TravelAnswer, passages: list[dict], records: list[dict]) -> str:
    validate_answer(answer, passages, records)
    lines = (["**Partial answer — some requested details remain unverified.** See 'Information still needed' below.", ""]
             if answer.status == "partial" else [])
    used = set()
    sources = {p["passage_id"]: p for p in passages}
    shown_guidance = []
    transport = []

    def point_text(point):
        ids = list(dict.fromkeys(e.passage_id for e in point.evidence))
        used.update(ids)
        return source_display_text(point.evidence, sources) + " " + " ".join(f"[{pid}]" for pid in ids)

    def activity_text(point):
        sections = list(dict.fromkeys(
            sources[e.passage_id]["metadata"].get("section_path", "").split(" > ")[-1]
            for e in point.evidence))
        label = "; ".join(s for s in sections if s)
        return (label + " — " if label else "") + point_text(point)

    def is_transport_guidance(point):
        # General guide headings are presentation categories, not venue names.
        topics = {"public transport", "getting around", "accessibility options",
                  "taxi or private hire car"}
        return bool(answer.itinerary) and all(
            sources[e.passage_id]["metadata"].get("section_path", "").split(" > ")[-1].strip().casefold()
            in topics for e in point.evidence)

    def add_guidance(point, prefix):
        spans = selected_source_spans(point.evidence, sources)
        if not spans_covered(spans, shown_guidance):
            lines.extend([prefix + point_text(point), ""])
            shown_guidance.extend(spans)

    if answer.assumptions:
        lines.extend(["### Planning assumptions", "", *[f"- {s}" for s in answer.assumptions], ""])
    facts = []
    for point in answer.knowledge_facts:
        if is_transport_guidance(point):
            transport.append(point)
        else:
            facts.append(point)
    if facts:
        lines.extend(["### Facts from the knowledge base", "",
                      "Selected excerpts from the saved travel sources; […] marks a gap between selections:", ""])
        for point in facts:
            add_guidance(point, "> ")
    lines.extend(render_live_data(records))
    if answer.itinerary or answer.suggestions:
        if answer.itinerary:
            introduction = (
                "Day allocation and activity choices are AI suggestions. Descriptions use selected source excerpts; "
                "[…] marks a gap between selections."
            )
            if any(record.get("tool") == "get_weather" and record.get("status") == "success"
                   for record in records):
                introduction += " Weather adjustments use the tool data shown above."
            lines.extend(["### Suggested plan — AI recommendations", "", introduction, ""])
        else:
            lines.extend(["### Practical tips from the sources", ""])
        for day in answer.itinerary:
            lines.extend([f"**Day {day.day}" + (f" — {day.date}" if day.date else "") + "**", ""])
            if day.gap_reason:
                lines.extend(["**Main activity still needed:** " + day.gap_reason, ""])
            shown_activities = []
            for point in sorted(day.activities, key=lambda p: p.role != "main"):
                if point.role != "main" and is_transport_guidance(point):
                    transport.append(point)
                    continue
                label = ("Main plan" if point.role == "main" else
                         "Optional indoor activity" if point.setting == "indoor_sheltered" and has_indoor_evidence(point) else
                         "Optional, only if weather allows")
                lines.append(f"- **{label}:** {activity_text(point)}")
                shown_activities.extend(selected_source_spans(point.evidence, sources))
            if day.indoor_alternative and not spans_covered(
                    selected_source_spans(day.indoor_alternative.evidence, sources), shown_activities):
                lines.append("- **Indoor alternative:** " + activity_text(day.indoor_alternative))
            if day.weather_adjustment:
                lines.append("- **Weather adjustment (suggestion):** " + day.weather_adjustment)
            lines.append("")
        for point in answer.suggestions:
            if is_transport_guidance(point):
                transport.append(point)
            else:
                add_guidance(point, "- ")
    if transport:
        # Combine existing references, without generating a new route or claim.
        evidence = list({(e.passage_id, e.quote): e for p in transport for e in p.evidence}.values())
        point = SupportedPoint(text="Transport guidance", evidence=evidence)
        lines.extend(["### Getting around", "", point_text(point), ""])
    if answer.missing_information:
        lines.extend(["### Information still needed", "", *[f"- {s}" for s in answer.missing_information], ""])
    if used:
        lines.extend(["### Knowledge-base sources", ""])
        for passage in passages:
            pid, metadata = passage["passage_id"], passage["metadata"]
            if pid in used:
                lines.append(f"- [{pid}] " + source_link(metadata["title"], metadata["url"]) + " — " + metadata.get("section_path", ""))
        lines.extend(["", "Destination sources are saved reference documents. Current opening hours, ticket availability and prices have not been verified."])
    return "\n".join(lines).strip()


def run_offline_checks() -> None:
    """Synthetic fixtures only: no Gemini request, MCP connection or web call."""
    from copy import deepcopy

    passage = {"passage_id": "P1", "page_content": "Synthetic venue has indoor galleries. Synthetic park has outdoor walking paths.", "metadata": {"title": "Test fixture", "url": "https://example.com/fixture"}}
    indoor = {"text": "Visit the synthetic venue galleries.", "evidence": [{"passage_id": "P1", "quote": "Synthetic venue has indoor galleries."}], "setting": "indoor_sheltered", "role": "main"}
    outdoor = {"text": "Consider the synthetic park if weather allows.", "evidence": [{"passage_id": "P1", "quote": "Synthetic park has outdoor walking paths."}], "setting": "outdoor", "role": "optional_if_weather_allows"}
    record = {"tool": "get_weather", "status": "success", "result": {"forecast": [{"date": "2026-09-23", "weather_code": 55, "precipitation_probability_max": 75, "precipitation_sum": 7.8}]}}
    payload = {"status": "answered", "knowledge_facts": [], "suggestions": [], "assumptions": [], "missing_information": [], "itinerary": [{"day": 1, "date": "2026-09-23", "activities": [indoor, outdoor], "indoor_alternative": None, "weather_adjustment": "Choose sheltered activities; leave the park conditional on suitable weather."}]}

    def check(data, weather, expect_error=False):
        try:
            validate_answer(TravelAnswer.model_validate(data), [passage], [weather])
        except ValueError:
            if not expect_error:
                raise
        else:
            if expect_error:
                raise AssertionError("An unsuitable plan was accepted.")

    check(payload, record)
    print("Sheltered main plan with optional outdoor activity: passed")
    bad = deepcopy(payload)
    bad["itinerary"][0]["activities"] = [{**outdoor, "role": "main"}]
    check(bad, record, expect_error=True)
    print("Outdoor main plan on a wet day rejected: passed")
    bad["itinerary"][0]["activities"][0]["setting"] = "indoor_sheltered"
    check(bad, record, expect_error=True)
    print("Unsupported indoor label rejected: passed")
    dry = deepcopy(record)
    dry["result"]["forecast"][0].update(weather_code=1, precipitation_probability_max=10, precipitation_sum=0)
    bad["itinerary"][0]["activities"][0]["setting"] = "outdoor"
    check(bad, dry)
    print("Outdoor main plan with a dry forecast allowed: passed")
    unknown = deepcopy(dry)
    unknown["result"]["forecast"][0].update(weather_code=None, precipitation_probability_max=None, precipitation_sum=None)
    snapshot = deepcopy(unknown)
    weather_planning_policy([unknown])
    if unknown != snapshot:
        raise AssertionError("Missing weather values were modified.")
    print("Missing weather values preserved: passed")
    catalog = build_excerpt_catalog([passage])
    selected_id = next(iter(catalog))
    referenced = deepcopy(payload)
    for activity in referenced["itinerary"][0]["activities"]:
        activity["evidence"] = [{"excerpt_id": selected_id}]
    resolved = resolve_submission(SubmitTravelAnswer.model_validate(referenced), [passage])
    validate_answer(resolved, [passage], [record])
    assert resolved.itinerary[0].activities[0].evidence[0].quote == catalog[selected_id]["quote"]
    print("Excerpt IDs resolved to exact source text: passed")
    referenced["itinerary"][0]["activities"][0]["evidence"] = [{"excerpt_id": "P999-E1"}]
    try:
        resolve_submission(SubmitTravelAnswer.model_validate(referenced), [passage])
    except ValueError:
        print("Invented excerpt ID rejected: passed")
    else:
        raise AssertionError("An invented source reference was accepted.")
    unsuitable = deepcopy(payload)
    unsuitable["itinerary"][0]["activities"] = [{**outdoor, "role": "main", "setting": "indoor_sheltered"}]
    unsuitable["itinerary"][0]["indoor_alternative"] = {k: v for k, v in indoor.items() if k in {"text", "evidence"}}
    partial, review = review_recommendations(TravelAnswer.model_validate(unsuitable), [record], ["I am vegetarian."])
    validate_answer(partial, [passage], [record])
    assert partial.status == "partial" and partial.itinerary[0].activities[0].text == indoor["evidence"][0]["quote"]
    assert any(c["action"] == "withheld" for c in review["adjustments"])
    print("Unsupported activity withheld; existing indoor alternative retained: passed")
    assert diet_conflicts("Try fish head curry for dinner.", "vegetarian") == ["fish"]
    assert not diet_conflicts("Visit an aquarium to see fish.", "vegetarian")
    assert not diet_conflicts("Avoid fish curry; order a vegetarian meal.", "vegetarian")
    assert not diet_conflicts("Order mock chicken made from tofu.", "vegetarian")
    assert explicit_diet(["I am vegetarian.", "I am no longer vegetarian."]) is None
    print("Explicit dietary conflicts and later preference changes checked: passed")
    print("Gemini API calls: 0. These were synthetic checks, not a new itinerary.")


if __name__ == "__main__":
    run_offline_checks()
