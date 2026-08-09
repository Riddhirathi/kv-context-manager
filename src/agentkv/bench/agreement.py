"""Next-action agreement (AGENTKV_SPEC.md §3.1).

"For each step in a replayed trajectory, compute the model's next action
given (a) full uncompacted context and (b) compacted context. Score exact
tool-name match, argument match, and a semantic similarity fallback. Report
agreement rate per policy."

The recorded trajectories (bench/replay.py) hold a strong recorder model's
actions — irrelevant here, since Gate 2/3 care about the *measurement* model's
own behavior, not whether it matches the recorder. So this module elicits a
fresh counterfactual action from the local model twice per decision point,
once against context that was never compacted and once against a policy's
compacted context, and compares the two outputs to each other.

Elicitation went through two rounds of empirical correction before landing
here:

1. Plain free-text continuation on the raw `/v1/completions` engine, betting
   that every prior assistant turn's `[{"name": ..., "args": {...}}]`
   rendering (`context/layout.py`) would be enough in-context precedent for
   the model to imitate. It mostly wasn't: at a trajectory's *first* decision
   point (nothing precedes it but the anchor and maybe one user turn) the
   model had zero examples to pattern-match and degenerated into a
   300+-word non-JSON ramble that never closed a valid `[...]` (traj-000
   step 2).
2. Fixed the no-example case with a single fixed one-shot exemplar
   (`_EXEMPLAR_TURNS`, spliced in right after the anchor by
   `build_decision_prompt_ids`) — but that surfaced a second, worse failure:
   the model then predicted an immediate end-of-sequence token (finish_reason
   "stop", 1 completion token, empty text) right after the primed `<assistant>`
   cue, apparently reading the exemplar's closed tool-call turn as a signal
   that the exchange was already over. Forcing generation past that point
   (`min_tokens`) didn't help either — it produced plausible prose reasoning
   but still never actually emitted a JSON call before stopping again.

Both failures are the same underlying problem: nothing here needs the model
to freely decide *when* and *whether* to emit valid JSON. `VLLMEngine.complete_text`'s
`guided_json` parameter (forwarded to vLLM's per-request grammar-constrained
decoding — no server restart, no tool-call parser plugin, still the plain
`/v1/completions` endpoint) forces every generated token to match
`ACTION_JSON_SCHEMA` from the first token on, which structurally rules out
both failure modes: an immediate stop is impossible (grammar excludes EOS
until the object is well-formed) and prose-without-a-call is impossible (the
grammar admits nothing else). The one-shot exemplar from round 2 is still
needed, though, for a problem grammar constraints can't fix on their own:
*semantic* grounding. Without it, the same example-free first decision point
reliably still emits well-formed JSON, but hallucinates a tool that doesn't
exist and repeats a garbage key in a loop until truncated (confirmed:
`{"name": "oncall_engineer", "args": {"tool_call": "tool_call_1", ...}}`,
repeating to the token cap). The exemplar gives the model at least one real
tool name to anchor on, applied identically to both the full and compacted
side so it cannot bias the (a) vs (b) comparison.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any

from agentkv.bench.replay import Turn
from agentkv.context.layout import Tokenizer, render_turns_to_token_ids
from agentkv.metrics.collector import RecordCollector

# A single fixed (user, assistant-with-tool-call) exemplar pair, inserted
# right after the anchor in every decision prompt (see module docstring) —
# not part of any real trajectory, purely a semantic-grounding primer.
_EXEMPLAR_TURNS = [
    Turn(role="user", content="(example) Begin."),
    Turn(
        role="assistant",
        content="<think>\nI will check the available files first.\n</think>\n",
        tool_calls=[{"name": "list_files", "args": {}}],
    ),
]

# Forwarded as `VLLMEngine.complete_text`'s `guided_json` for every decision
# elicitation (see module docstring for why grammar-constrained decoding is
# required here, not optional hardening).
ACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "args": {"type": "object"}},
    "required": ["name", "args"],
}


def is_decision_point(turn: Turn) -> bool:
    """True for a recorded turn where the (recorder) model chose a tool call —
    the only turns where "the model's next action" is a meaningful question."""
    return turn.role == "assistant" and bool(turn.tool_calls)


def build_decision_prompt_ids(context_turns: list[Turn], tokenizer: Tokenizer) -> list[int]:
    """Renders `context_turns` — with `_EXEMPLAR_TURNS` spliced in right after
    the anchor (`context_turns[0]`, by this project's universal convention) —
    plus a bare trailing `<assistant>` cue, so the engine's continuation
    naturally lands where a real next action would go."""
    if context_turns:
        anchor, rest = context_turns[0], context_turns[1:]
        primed = [anchor, *_EXEMPLAR_TURNS, *rest]
    else:
        primed = list(_EXEMPLAR_TURNS)
    return render_turns_to_token_ids([*primed, Turn(role="assistant")], tokenizer)


@dataclass(frozen=True)
class ParsedAction:
    tool_name: str | None
    args: dict[str, Any] | None
    raw_text: str
    parse_ok: bool


def _json_candidates(text: str) -> list[str]:
    """Every top-level `{...}` or `[...]` substring of `text`, by bracket-depth
    scan (not regex — args can themselves contain nested brackets), most
    recent first. `guided_json`-elicited text is expected to be nothing but
    a bare `{...}` object, but this also tolerates the (list-wrapped,
    possibly prose-surrounded) shape older/non-guided callers and this
    module's own unit tests use, by treating `[` and `{` as interchangeable
    "candidate open" markers and letting `json.loads` reject anything that
    doesn't actually parse."""
    candidates: list[str] = []
    depth = 0
    start: int | None = None
    for i, ch in enumerate(text):
        if ch in "[{":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "]}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : i + 1])
                    start = None
    return list(reversed(candidates))


def _as_call_dict(parsed: object) -> dict[str, Any] | None:
    """`parsed` is a valid call if it's a `{"name": ..., "args": {...}}` dict
    directly, or a non-empty list whose first element is such a dict."""
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict) or "name" not in parsed:
        return None
    name = parsed["name"]
    args = parsed.get("args")
    if not isinstance(name, str) or not isinstance(args, (dict, type(None))):
        return None
    return {"name": name, "args": args if args is not None else {}}


def parse_action(raw_text: str) -> ParsedAction:
    """Extracts the first well-formed `{"name": ..., "args": {...}}` call
    found in `raw_text` — either the whole string (the `guided_json` case) or
    embedded in it — tolerating everything else (reasoning text, truncated
    output, no tool call at all) as a valid, non-exceptional outcome: a parse
    failure is real data (the policy under test produced unparseable output),
    not a bug in this function."""
    for candidate in _json_candidates(raw_text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        call = _as_call_dict(parsed)
        if call is not None:
            return ParsedAction(
                tool_name=call["name"], args=call["args"], raw_text=raw_text, parse_ok=True
            )
    return ParsedAction(tool_name=None, args=None, raw_text=raw_text, parse_ok=False)


@dataclass(frozen=True)
class AgreementRecord:
    """One row per decision point per policy (spec §3.1's three scores)."""

    step_idx: int
    policy: str
    seed: int
    full_parse_ok: bool
    compacted_parse_ok: bool
    full_tool_name: str | None
    compacted_tool_name: str | None
    tool_name_match: bool
    args_match: bool
    semantic_similarity: float


def score_agreement(
    *,
    step_idx: int,
    policy: str,
    seed: int,
    full_action: ParsedAction,
    compacted_action: ParsedAction,
) -> AgreementRecord:
    """Exact tool-name match, exact argument match (only meaningful given a
    name match), and a `difflib`-based semantic similarity over the raw
    generated text as the fallback signal spec §3.1 asks for — always
    computed, not just when the exact checks fail, so it doubles as a
    continuous near-miss signal (e.g. right tool, near-identical args string)
    rather than only a binary yes/no.
    """
    tool_name_match = (
        full_action.parse_ok
        and compacted_action.parse_ok
        and full_action.tool_name == compacted_action.tool_name
    )
    args_match = tool_name_match and full_action.args == compacted_action.args
    semantic_similarity = difflib.SequenceMatcher(
        None, full_action.raw_text, compacted_action.raw_text
    ).ratio()
    return AgreementRecord(
        step_idx=step_idx,
        policy=policy,
        seed=seed,
        full_parse_ok=full_action.parse_ok,
        compacted_parse_ok=compacted_action.parse_ok,
        full_tool_name=full_action.tool_name,
        compacted_tool_name=compacted_action.tool_name,
        tool_name_match=tool_name_match,
        args_match=args_match,
        semantic_similarity=semantic_similarity,
    )


class AgreementEventCollector(RecordCollector[AgreementRecord]):
    """Buffers AgreementRecords and flushes them to an append-only parquet
    file, same contract as `bench.divergence.CompactionEventCollector`."""
