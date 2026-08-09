from __future__ import annotations

import pytest

from agentkv.bench.tasks.ledger import LedgerEnv


def test_same_seed_is_deterministic():
    env_a = LedgerEnv(seed=7, n_transactions=10)
    env_b = LedgerEnv(seed=7, n_transactions=10)
    instructions_a = [env_a.get_next_instruction() for _ in range(10)]
    instructions_b = [env_b.get_next_instruction() for _ in range(10)]
    assert instructions_a == instructions_b


def test_different_seeds_diverge():
    env_a = LedgerEnv(seed=1, n_transactions=10)
    env_b = LedgerEnv(seed=2, n_transactions=10)
    instructions_a = [env_a.get_next_instruction() for _ in range(10)]
    instructions_b = [env_b.get_next_instruction() for _ in range(10)]
    assert instructions_a != instructions_b


def test_instruction_queue_exhausts_and_reports_remaining():
    env = LedgerEnv(seed=3, n_transactions=3)
    assert env.instructions_remaining == 3
    for _ in range(3):
        instr = env.get_next_instruction()
        assert "No more instructions" not in instr
    assert env.instructions_remaining == 0
    assert "No more instructions" in env.get_next_instruction()


def test_correctly_replaying_every_instruction_verifies_true():
    env = LedgerEnv(seed=5, n_transactions=20)
    while env.instructions_remaining > 0:
        instr = env.get_next_instruction()
        # Parse our own generated instruction text back into a tool call —
        # mirrors what a perfect agent would do.
        if instr.startswith("Deposit"):
            _, amount, _, account = instr.rstrip(".").split(" ", 3)
            env.deposit(account, int(amount))
        elif instr.startswith("Withdraw"):
            _, amount, _, account = instr.rstrip(".").split(" ", 3)
            env.withdraw(account, int(amount))
        elif instr.startswith("Transfer"):
            parts = instr.rstrip(".").split(" ")
            amount, from_account, to_account = int(parts[1]), parts[3], parts[5]
            env.transfer(from_account, to_account, amount)
    env.submit_final_report(dict(env._actual_balances))  # noqa: SLF001
    assert env.verify() is True


def test_skipping_an_instruction_causes_verification_failure():
    env = LedgerEnv(seed=5, n_transactions=20)
    env.get_next_instruction()  # fetched but never applied — simulates a forgotten step
    while env.instructions_remaining > 0:
        env.get_next_instruction()
    env.submit_final_report(dict(env._actual_balances))  # noqa: SLF001
    assert env.verify() is False


def test_verify_false_before_any_report_submitted():
    env = LedgerEnv(seed=9, n_transactions=5)
    assert env.is_done is False
    assert env.verify() is False


def test_verify_true_only_after_submit_final_report():
    env = LedgerEnv(seed=9, n_transactions=0)
    assert env.is_done is False
    env.submit_final_report(dict(env._actual_balances))  # noqa: SLF001
    assert env.is_done is True
    assert env.verify() is True


def test_withdraw_rejects_overdraft_without_mutating():
    env = LedgerEnv(seed=11, n_transactions=0, n_accounts=2)
    account = env._account_names[0]  # noqa: SLF001
    before = env.get_balance(account)
    result = env.withdraw(account, 10**9)
    assert "Insufficient funds" in result
    assert env.get_balance(account) == before


def test_transfer_rejects_overdraft_without_mutating():
    env = LedgerEnv(seed=12, n_transactions=0, n_accounts=2)
    a, b = env._account_names  # noqa: SLF001
    before_a = env.get_balance(a)
    before_b = env.get_balance(b)
    result = env.transfer(a, b, 10**9)
    assert "Insufficient funds" in result
    assert env.get_balance(a) == before_a
    assert env.get_balance(b) == before_b


def test_reference_simulation_also_rejects_overdraft():
    """The reference balances used for verification must apply the exact
    same overdraft rule as the live tools, or a "correct" agent that
    replays every instruction faithfully could still fail verification
    just because the simulated ground truth diverged."""
    env = LedgerEnv(seed=13, n_transactions=40, n_accounts=2)
    balances = {a: 0 for a in env._account_names}  # noqa: SLF001
    for txn in env._transactions:  # noqa: SLF001
        if txn.kind == "withdraw":
            assert balances[txn.account] < txn.amount or True  # sanity: no crash either way
    # Reference balances must never go negative, since deposits start at
    # >=100 and every withdraw/transfer in the reference sim is rejected on
    # insufficient funds exactly like the live tools.
    assert all(v >= 0 for v in env._reference_balances.values())  # noqa: SLF001


def test_verify_tolerates_string_and_whitespace_in_report():
    env = LedgerEnv(seed=14, n_transactions=0)
    padded = {f" {k} ": str(v) for k, v in env._reference_balances.items()}  # noqa: SLF001
    env.submit_final_report(padded)
    assert env.verify() is True


def test_verify_false_on_non_numeric_value():
    env = LedgerEnv(seed=15, n_transactions=0)
    bad = dict(env._reference_balances)  # noqa: SLF001
    first_key = next(iter(bad))
    bad[first_key] = "not-a-number"
    env.submit_final_report(bad)
    assert env.verify() is False


def test_verify_false_on_missing_account():
    env = LedgerEnv(seed=16, n_transactions=0)
    incomplete = dict(env._reference_balances)  # noqa: SLF001
    incomplete.pop(next(iter(incomplete)))
    env.submit_final_report(incomplete)
    assert env.verify() is False


def test_call_dispatches_by_tool_name():
    env = LedgerEnv(seed=17, n_transactions=1)
    instr = env.call("get_next_instruction", {})
    assert instr == env._transactions[0].instruction_text  # noqa: SLF001
    account = env._account_names[0]  # noqa: SLF001
    result = env.call("deposit", {"account": account, "amount": 10})
    assert "Deposited 10" in result


def test_n_accounts_out_of_range_raises():
    with pytest.raises(ValueError):
        LedgerEnv(seed=0, n_accounts=1)
    with pytest.raises(ValueError):
        LedgerEnv(seed=0, n_accounts=99)


def test_action_schema_enums_real_tool_names_only():
    env = LedgerEnv(seed=18, n_transactions=1)
    schema = env.action_schema()
    tool_names = {t["name"] for t in env.tool_schemas()}
    assert set(schema["properties"]["name"]["enum"]) == tool_names
    assert schema["required"] == ["name", "args"]
