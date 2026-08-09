from __future__ import annotations

from agentkv.bench.agreement import (
    AgreementRecord,
    build_decision_prompt_ids,
    is_decision_point,
    parse_action,
    score_agreement,
)
from agentkv.bench.replay import Turn


class WordCountTokenizer:
    """Deterministic fake: one token id per word, no vLLM/GPU dependency."""

    def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]


def test_is_decision_point_true_for_assistant_tool_call():
    turn = Turn(role="assistant", content=None, tool_calls=[{"name": "list_files", "args": {}}])
    assert is_decision_point(turn) is True


def test_is_decision_point_false_for_plain_assistant_content():
    turn = Turn(role="assistant", content="just some text, no tool call")
    assert is_decision_point(turn) is False


def test_is_decision_point_false_for_non_assistant_roles():
    assert is_decision_point(Turn(role="user", content="hi")) is False
    tool_turn = Turn(role="tool", tool_results=[{"name": "x", "output": "y"}])
    assert is_decision_point(tool_turn) is False


def test_build_decision_prompt_ids_appends_bare_assistant_cue():
    tokenizer = WordCountTokenizer()
    turns = [Turn(role="system", content="you are an agent"), Turn(role="user", content="go")]
    ids = build_decision_prompt_ids(turns, tokenizer)
    # Same as rendering [anchor, *exemplar, rest...] + a trailing empty <assistant> turn.
    expected = tokenizer.encode(
        "<system>\nyou are an agent\n"
        "<user>\n(example) Begin.\n"
        "<assistant>\n<think>\nI will check the available files first.\n</think>\n"
        '[{"name": "list_files", "args": {}}]\n'
        "<user>\ngo\n<assistant>"
    )
    assert ids == expected


def test_build_decision_prompt_ids_primes_even_the_first_decision_point():
    """The one gap the exemplar exists to close: with only an anchor in
    context (no real assistant tool-call turn recorded yet), the model would
    otherwise have zero in-context examples of the JSON tool-call format."""
    tokenizer = WordCountTokenizer()
    turns = [Turn(role="system", content="you are an agent")]
    ids = build_decision_prompt_ids(turns, tokenizer)
    rendered = tokenizer.encode(
        "<system>\nyou are an agent\n"
        "<user>\n(example) Begin.\n"
        "<assistant>\n<think>\nI will check the available files first.\n</think>\n"
        '[{"name": "list_files", "args": {}}]\n'
        "<assistant>"
    )
    assert ids == rendered


def test_parse_action_extracts_trailing_tool_call():
    text = (
        "<think>\nI should list the files first.\n</think>\n\n"
        '[{"name": "list_files", "args": {}}]'
    )
    action = parse_action(text)
    assert action.parse_ok is True
    assert action.tool_name == "list_files"
    assert action.args == {}
    assert action.raw_text == text


def test_parse_action_extracts_bare_json_object():
    """The real `guided_json` output shape: a single JSON object, no list
    wrapper, no surrounding prose — vLLM's grammar constraint guarantees
    exactly this shape (see agreement.py's ACTION_JSON_SCHEMA)."""
    text = '{"name": "read_file", "args": {"path": "a.py"}}'
    action = parse_action(text)
    assert action.parse_ok is True
    assert action.tool_name == "read_file"
    assert action.args == {"path": "a.py"}


def test_parse_action_handles_nested_brackets_in_args():
    text = '[{"name": "search_logs", "args": {"queries": ["a", "b", "c"]}}]'
    action = parse_action(text)
    assert action.parse_ok is True
    assert action.tool_name == "search_logs"
    assert action.args == {"queries": ["a", "b", "c"]}


def test_parse_action_no_tool_call_is_a_valid_non_exceptional_outcome():
    text = "<think>\nI am not sure what to do yet.\n</think>\nLet me think more."
    action = parse_action(text)
    assert action.parse_ok is False
    assert action.tool_name is None
    assert action.args is None
    assert action.raw_text == text


def test_parse_action_ignores_malformed_json_and_falls_back_to_earlier_candidate():
    text = 'reasoning [1, 2, 3] then [{"name": "read_file", "args": {"path": "a.py"}}'  # truncated
    action = parse_action(text)
    # The truncated trailing candidate never parses; [1, 2, 3] parses but has
    # no dict with a "name" key, so this is correctly "no tool call found".
    assert action.parse_ok is False


def test_score_agreement_exact_match():
    full = parse_action('[{"name": "read_file", "args": {"path": "a.py"}}]')
    compacted = parse_action('[{"name": "read_file", "args": {"path": "a.py"}}]')
    record = score_agreement(
        step_idx=5, policy="naive", seed=0, full_action=full, compacted_action=compacted
    )
    assert isinstance(record, AgreementRecord)
    assert record.tool_name_match is True
    assert record.args_match is True
    assert record.semantic_similarity == 1.0


def test_score_agreement_same_tool_different_args():
    full = parse_action('[{"name": "read_file", "args": {"path": "a.py"}}]')
    compacted = parse_action('[{"name": "read_file", "args": {"path": "b.py"}}]')
    record = score_agreement(
        step_idx=5, policy="naive", seed=0, full_action=full, compacted_action=compacted
    )
    assert record.tool_name_match is True
    assert record.args_match is False
    assert 0.0 < record.semantic_similarity < 1.0


def test_score_agreement_different_tools():
    full = parse_action('[{"name": "read_file", "args": {"path": "a.py"}}]')
    compacted = parse_action('[{"name": "list_files", "args": {}}]')
    record = score_agreement(
        step_idx=5, policy="append_only", seed=0, full_action=full, compacted_action=compacted
    )
    assert record.tool_name_match is False
    assert record.args_match is False


def test_score_agreement_unparseable_action_never_matches():
    full = parse_action('[{"name": "read_file", "args": {"path": "a.py"}}]')
    compacted = parse_action("no tool call here at all")
    record = score_agreement(
        step_idx=5, policy="naive", seed=0, full_action=full, compacted_action=compacted
    )
    assert record.compacted_parse_ok is False
    assert record.tool_name_match is False
    assert record.args_match is False
