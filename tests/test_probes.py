from __future__ import annotations

from agentkv.bench.replay import Turn
from agentkv.bench.tasks.probes import (
    RetentionRecord,
    build_query_prompt_ids,
    check_retention,
    generate_facts,
    inject_facts,
)


class WordCountTokenizer:
    """Deterministic fake: one token id per word, no vLLM/GPU dependency."""

    def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]


def test_generate_facts_deterministic_given_seed():
    facts_a = generate_facts(seed=7, n=5)
    facts_b = generate_facts(seed=7, n=5)
    assert [f.statement for f in facts_a] == [f.statement for f in facts_b]
    assert [f.expected_answer for f in facts_a] == [f.expected_answer for f in facts_b]


def test_generate_facts_different_seeds_diverge():
    facts_a = generate_facts(seed=1, n=5)
    facts_b = generate_facts(seed=2, n=5)
    assert [f.statement for f in facts_a] != [f.statement for f in facts_b]


def test_generate_facts_distinct_kinds():
    facts = generate_facts(seed=3, n=5)
    assert len(facts) == 5
    assert len({f.kind for f in facts}) == 5


def test_generate_facts_rejects_too_many():
    import pytest

    with pytest.raises(ValueError):
        generate_facts(seed=0, n=999)


def test_inject_facts_splices_before_target_index():
    turns = [Turn(role="user", content=str(i)) for i in range(5)]
    facts = generate_facts(seed=1, n=2)
    combined = inject_facts(turns, facts, insertion_steps=[1, 3])

    assert combined[0] == turns[0]
    assert combined[1].content == facts[0].statement
    assert combined[2] == turns[1]
    assert combined[3] == turns[2]
    assert combined[4].content == facts[1].statement
    assert combined[5] == turns[3]
    assert combined[6] == turns[4]


def test_inject_facts_multiple_at_same_step_preserve_order():
    turns = [Turn(role="user", content=str(i)) for i in range(3)]
    facts = generate_facts(seed=2, n=2)
    combined = inject_facts(turns, facts, insertion_steps=[1, 1])
    assert combined[1].content == facts[0].statement
    assert combined[2].content == facts[1].statement
    assert combined[3] == turns[1]


def test_inject_facts_step_past_end_appends():
    turns = [Turn(role="user", content="only")]
    facts = generate_facts(seed=4, n=1)
    combined = inject_facts(turns, facts, insertion_steps=[10])
    assert combined == [turns[0], Turn(role="user", content=facts[0].statement)]


def test_inject_facts_length_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        inject_facts([Turn(role="user", content="x")], generate_facts(seed=0, n=2), [0])


def test_build_query_prompt_ids_includes_exemplar_and_question():
    tokenizer = WordCountTokenizer()
    turns = [Turn(role="system", content="you are an agent")]
    ids = build_query_prompt_ids(turns, "What is the account ID?", tokenizer)
    expected = tokenizer.encode(
        "<system>\nyou are an agent\n"
        "<user>\n(example) What is 2 + 2?\n"
        "<assistant>\nThe answer is 4.\n"
        "<user>\nWhat is the account ID?\n"
        "<assistant>"
    )
    assert ids == expected


def test_check_retention_true_when_value_present():
    fact = generate_facts(seed=5, n=1)[0]
    answer = f"Based on the context, {fact.expected_answer} is the value."
    assert check_retention(fact, answer) is True


def test_check_retention_false_when_value_absent():
    fact = generate_facts(seed=5, n=1)[0]
    assert check_retention(fact, "I don't have that information.") is False


def test_check_retention_case_insensitive():
    fact = generate_facts(seed=6, n=1)[0]
    answer = f"the answer is {fact.expected_answer.upper()}"
    assert check_retention(fact, answer.lower()) is True


def test_retention_record_is_a_plain_dataclass():
    record = RetentionRecord(
        fact_id="account_id-0",
        fact_kind="account_id",
        policy="naive",
        seed=0,
        depth=50,
        retained=True,
        answer_text="ACC-12345 is the account ID.",
    )
    assert record.depth == 50
    assert record.retained is True
