"""Retention probes (AGENTKV_SPEC.md §3.2).

"Inject synthetic facts at controlled depths in the trajectory ("the account
ID is X", "the user prefers Y"). At step 100, query for each. Produces a
retention-vs-depth curve per policy. This is how you show precisely what each
policy forgets."

Unlike `bench/agreement.py`, this never needs the raw uncompacted context as a
baseline: a policy's compacted context stays bounded by construction (the
threshold in `configs/policies/*.yaml`), so it can always be replayed all the
way to the query step without hitting `max_model_len` the way Phase 3.1's
"full" condition did.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from agentkv.bench.replay import Turn
from agentkv.context.layout import Tokenizer, render_turns_to_token_ids
from agentkv.metrics.collector import RecordCollector


@dataclass(frozen=True)
class _FactTemplate:
    kind: str
    statement_template: str
    question: str
    generate_value: Callable[[random.Random], str]


_FACT_TEMPLATES: list[_FactTemplate] = [
    _FactTemplate(
        kind="account_id",
        statement_template="For reference, the account ID for this incident is {value}.",
        question="What is the account ID for this incident?",
        generate_value=lambda rng: f"ACC-{rng.randint(10000, 99999)}",
    ),
    _FactTemplate(
        kind="user_preference",
        statement_template="Note: the user prefers to be contacted via {value}.",
        question="How does the user prefer to be contacted?",
        generate_value=lambda rng: rng.choice(["email", "Slack", "phone", "SMS"]),
    ),
    _FactTemplate(
        kind="ticket_number",
        statement_template="The support ticket number for this incident is {value}.",
        question="What is the support ticket number for this incident?",
        generate_value=lambda rng: f"TCK-{rng.randint(1000, 9999)}",
    ),
    _FactTemplate(
        kind="priority_level",
        statement_template="This incident has been marked priority {value}.",
        question="What priority level has this incident been marked?",
        generate_value=lambda rng: rng.choice(["P0", "P1", "P2", "P3"]),
    ),
    _FactTemplate(
        kind="escalation_contact",
        statement_template="If this cannot be resolved, escalate to {value}.",
        question="Who should this incident be escalated to if it cannot be resolved?",
        generate_value=lambda rng: rng.choice(
            ["Priya Patel", "Marcus Chen", "Sofia Rossi", "Daniel Okafor"]
        ),
    ),
    _FactTemplate(
        kind="deadline",
        statement_template="This incident must be resolved by {value}.",
        question="By when must this incident be resolved?",
        generate_value=lambda rng: f"{rng.choice(['14:00', '18:00', '20:00', '23:00'])} UTC",
    ),
]


@dataclass(frozen=True)
class SyntheticFact:
    fact_id: str
    kind: str
    statement: str
    question: str
    expected_answer: str


def generate_facts(seed: int, n: int) -> list[SyntheticFact]:
    """`n` facts of distinct kinds, deterministic given `seed` — one call per
    trajectory/seed combination, same convention as `synth_env.py`."""
    if n > len(_FACT_TEMPLATES):
        raise ValueError(f"only {len(_FACT_TEMPLATES)} fact templates available, requested {n}")
    rng = random.Random(seed)
    templates = list(_FACT_TEMPLATES)
    rng.shuffle(templates)
    facts: list[SyntheticFact] = []
    for i, template in enumerate(templates[:n]):
        value = template.generate_value(rng)
        facts.append(
            SyntheticFact(
                fact_id=f"{template.kind}-{i}",
                kind=template.kind,
                statement=template.statement_template.format(value=value),
                question=template.question,
                expected_answer=value,
            )
        )
    return facts


def inject_facts(
    turns: list[Turn], facts: list[SyntheticFact], insertion_steps: list[int]
) -> list[Turn]:
    """Returns `turns` with each fact's statement spliced in as a `user` turn
    immediately before `turns[insertion_steps[i]]` (in `turns`' own, pre-
    injection indexing — earlier insertions do not shift where later ones
    land). An `insertion_steps[i] >= len(turns)` places that fact at the end.
    """
    if len(facts) != len(insertion_steps):
        raise ValueError("facts and insertion_steps must be the same length")
    pending = sorted(zip(insertion_steps, facts, strict=True), key=lambda pair: pair[0])
    result: list[Turn] = []
    cursor = 0
    for i, turn in enumerate(turns):
        while cursor < len(pending) and pending[cursor][0] == i:
            result.append(Turn(role="user", content=pending[cursor][1].statement))
            cursor += 1
        result.append(turn)
    while cursor < len(pending):
        result.append(Turn(role="user", content=pending[cursor][1].statement))
        cursor += 1
    return result


# A single fixed (user, assistant) QA exemplar, spliced in right after the
# anchor in every query prompt — same rationale as `bench/agreement.py`'s
# `_EXEMPLAR_TURNS`: the base trajectory is entirely tool-call turns, so a
# direct natural-language answer has no in-context precedent otherwise.
# Deliberately arithmetic, not incident-domain: an earlier version used a
# ticket-number example, and the model started literally copying that
# exemplar's answer whenever queried about the real `ticket_number` fact
# (verified: answered "TCK-1029" — the exemplar's fixed value — regardless
# of what the actual injected fact was) instead of retrieving the real
# value, silently invalidating that fact kind's retention score. A neutral
# example that shares no vocabulary with any `_FACT_TEMPLATES` kind avoids
# that failure mode structurally rather than requiring per-kind exemplars.
_QA_EXEMPLAR_TURNS = [
    Turn(role="user", content="(example) What is 2 + 2?"),
    Turn(role="assistant", content="The answer is 4."),
]


# Forwarded as `VLLMEngine.complete_text`'s `guided_json` for every query.
# Required, not optional hardening: verified empirically (same failure mode
# `bench/agreement.py` hit) that free-text elicitation here — even primed
# with `_QA_EXEMPLAR_TURNS` and `min_tokens` — mostly re-enters `<think>`
# mode instead of answering directly, because the base trajectory context is
# entirely `<think>...</think>`-wrapped tool-call turns; the reasoning
# preamble alone then exhausts a reasonably-sized token budget before ever
# reaching an answer. Grammar-constrained decoding rules that out from the
# first token on. `check_retention` matches the expected value as a
# substring, so the JSON wrapper around the answer doesn't need stripping.
ANSWER_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def build_query_prompt_ids(
    context_turns: list[Turn], question: str, tokenizer: Tokenizer
) -> list[int]:
    """Renders `context_turns` — with `_QA_EXEMPLAR_TURNS` spliced in right
    after the anchor — plus a trailing `(question, bare <assistant> cue)`
    pair, so the engine's continuation naturally answers `question`."""
    if context_turns:
        anchor, rest = context_turns[0], context_turns[1:]
        primed = [anchor, *_QA_EXEMPLAR_TURNS, *rest]
    else:
        primed = list(_QA_EXEMPLAR_TURNS)
    query_turn = Turn(role="user", content=question)
    return render_turns_to_token_ids([*primed, query_turn, Turn(role="assistant")], tokenizer)


def check_retention(fact: SyntheticFact, answer_text: str) -> bool:
    """A fact counts as retained if its exact expected value appears
    verbatim (case-insensitive) in the model's answer — deliberately strict:
    this measures whether the *specific injected value* survived compaction,
    not whether the model produced a plausible-sounding answer."""
    return fact.expected_answer.lower() in answer_text.lower()


@dataclass(frozen=True)
class RetentionRecord:
    """One row per (fact, policy) query (spec §3.2's retention-vs-depth curve)."""

    fact_id: str
    fact_kind: str
    policy: str
    seed: int
    depth: int
    retained: bool
    answer_text: str


class RetentionEventCollector(RecordCollector[RetentionRecord]):
    """Buffers RetentionRecords and flushes them to an append-only parquet
    file, same contract as `bench.agreement.AgreementEventCollector`."""
