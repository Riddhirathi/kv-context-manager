"""Synthetic verifiable ledger task (AGENTKV_SPEC.md §3.3).

"Build one long-horizon task a 1.7B model can actually complete, with
programmatic verification: e.g. an inventory/ledger environment where the
agent makes ~100 tool calls that mutate state, and the final reported balance
is checked against ground truth. Binary reward. This is your real 'task
success rate' number."

Unlike `bench/tasks/synth_env.py` (a read-only investigation environment
replayed from pre-recorded trajectories), this environment is *live*: the
agent fetches one queued transaction instruction at a time via
`get_next_instruction`, must call the matching mutating tool
(`deposit`/`withdraw`/`transfer`) to actually apply it, and eventually calls
`submit_final_report` with every account's final balance. There is no
recorded trajectory to replay here — `experiments/phase3_ledger.py` drives
the loop live against the measurement model.

Two balance dicts are tracked, deliberately:

- `_actual_balances` — mutated only by the agent's own real tool calls, i.e.
  "what actually happened."
- `_reference_balances` — computed once at construction by simulating the
  full transaction queue correctly, i.e. "what should have happened."

`verify()` compares the agent's *reported* final balances against
`_reference_balances`, not `_actual_balances`: success means the agent
correctly executed and remembered the entire queue despite however much
context got compacted along the way — not merely that it accurately reported
whatever mistakes it made.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from agentkv.metrics.collector import RecordCollector

_ACCOUNTS = ["ACC-A", "ACC-B", "ACC-C", "ACC-D", "ACC-E", "ACC-F", "ACC-G", "ACC-H"]


@dataclass(frozen=True)
class Transaction:
    kind: str  # "deposit" | "withdraw" | "transfer"
    account: str  # source account for all three kinds
    to_account: str | None  # only set for "transfer"
    amount: int
    instruction_text: str


def _apply(balances: dict[str, int], txn: Transaction) -> bool:
    """Mutates `balances` in place per `txn`, matching the real tools'
    overdraft rule below (no mutation, returns False, if funds are
    insufficient) — used both by the live tools and by the reference
    simulation, so the two stay comparable under the exact same rule."""
    if txn.kind == "deposit":
        balances[txn.account] += txn.amount
        return True
    if txn.kind == "withdraw":
        if balances[txn.account] < txn.amount:
            return False
        balances[txn.account] -= txn.amount
        return True
    if txn.kind == "transfer":
        assert txn.to_account is not None
        if balances[txn.account] < txn.amount:
            return False
        balances[txn.account] -= txn.amount
        balances[txn.to_account] += txn.amount
        return True
    raise ValueError(f"unknown transaction kind: {txn.kind}")


class LedgerEnv:
    def __init__(self, seed: int, n_transactions: int = 50, n_accounts: int = 5) -> None:
        if not 2 <= n_accounts <= len(_ACCOUNTS):
            raise ValueError(f"n_accounts must be in [2, {len(_ACCOUNTS)}]")
        rng = random.Random(seed)
        self._account_names = list(_ACCOUNTS[:n_accounts])
        initial_balances = {a: rng.randint(100, 1000) for a in self._account_names}
        self._transactions = self._generate_transactions(rng, n_transactions)
        self._actual_balances = dict(initial_balances)
        self._reference_balances = dict(initial_balances)
        for txn in self._transactions:
            _apply(self._reference_balances, txn)
        self._instruction_cursor = 0
        self._final_report: dict[str, int] | None = None

    def _generate_transactions(self, rng: random.Random, n: int) -> list[Transaction]:
        txns: list[Transaction] = []
        for _ in range(n):
            kind = rng.choice(["deposit", "withdraw", "transfer"])
            account = rng.choice(self._account_names)
            amount = rng.randint(5, 200)
            if kind == "transfer":
                to_account = rng.choice([a for a in self._account_names if a != account])
                text = f"Transfer {amount} from {account} to {to_account}."
                txns.append(Transaction(kind, account, to_account, amount, text))
            elif kind == "deposit":
                text = f"Deposit {amount} into {account}."
                txns.append(Transaction(kind, account, None, amount, text))
            else:
                text = f"Withdraw {amount} from {account}."
                txns.append(Transaction(kind, account, None, amount, text))
        return txns

    def system_prompt(self) -> str:
        accounts = ", ".join(self._account_names)
        return (
            "You are an autonomous ledger clerk. There is a queue of pending transactions. "
            "Call get_next_instruction to fetch one instruction at a time, then call the "
            "matching tool (deposit, withdraw, or transfer) to apply it before fetching the "
            "next one. For example: call get_next_instruction, receive an instruction like "
            "'Deposit 45 into ACC-B.', then call deposit with account='ACC-B' and amount=45, "
            "then call get_next_instruction again for the next instruction. Do not skip or "
            "reorder instructions. "
            f"The accounts are: {accounts}. "
            "Start now by calling get_next_instruction. Repeat the fetch-then-apply cycle for "
            "every single instruction in the queue — there may be many. Do NOT call "
            "submit_final_report early: only call it after get_next_instruction has explicitly "
            "told you the queue is empty. Calling submit_final_report while instructions remain "
            "is an immediate failure."
        )

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "get_next_instruction",
                "description": "Fetch the next pending transaction instruction.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "deposit",
                "description": "Deposit an amount into an account.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "account": {"type": "string"},
                        "amount": {"type": "integer"},
                    },
                    "required": ["account", "amount"],
                },
            },
            {
                "name": "withdraw",
                "description": "Withdraw an amount from an account.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "account": {"type": "string"},
                        "amount": {"type": "integer"},
                    },
                    "required": ["account", "amount"],
                },
            },
            {
                "name": "transfer",
                "description": "Transfer an amount from one account to another.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_account": {"type": "string"},
                        "to_account": {"type": "string"},
                        "amount": {"type": "integer"},
                    },
                    "required": ["from_account", "to_account", "amount"],
                },
            },
            {
                "name": "get_balance",
                "description": "Look up an account's current balance.",
                "parameters": {
                    "type": "object",
                    "properties": {"account": {"type": "string"}},
                    "required": ["account"],
                },
            },
            {
                "name": "submit_final_report",
                "description": (
                    "Report the final balance of every account once all instructions "
                    "have been processed. Ends the task."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"balances": {"type": "object"}},
                    "required": ["balances"],
                },
            },
        ]

    def action_schema(self) -> dict[str, Any]:
        """`guided_json` schema for eliciting the next tool call — same
        `{"name": ..., "args": {...}}` shape `bench/agreement.py` uses, but
        with `name` enum-restricted to this environment's real tools, so
        grammar-constrained decoding also rules out a hallucinated tool
        name (a real, observed failure mode without an enum — see
        `bench/agreement.py`'s module docstring for the equivalent case)."""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [schema["name"] for schema in self.tool_schemas()],
                },
                "args": {"type": "object"},
            },
            "required": ["name", "args"],
        }

    def get_next_instruction(self) -> str:
        if self._instruction_cursor >= len(self._transactions):
            return (
                "No more instructions. Call submit_final_report with the "
                "final balance of every account."
            )
        text = self._transactions[self._instruction_cursor].instruction_text
        self._instruction_cursor += 1
        return text

    def deposit(self, account: str, amount: int) -> str:
        if account not in self._actual_balances:
            return f"Unknown account: {account}."
        self._actual_balances[account] += amount
        return f"Deposited {amount} into {account}. New balance: {self._actual_balances[account]}."

    def withdraw(self, account: str, amount: int) -> str:
        if account not in self._actual_balances:
            return f"Unknown account: {account}."
        if self._actual_balances[account] < amount:
            return f"Insufficient funds in {account}: cannot withdraw {amount}."
        self._actual_balances[account] -= amount
        return f"Withdrew {amount} from {account}. New balance: {self._actual_balances[account]}."

    def transfer(self, from_account: str, to_account: str, amount: int) -> str:
        if from_account not in self._actual_balances or to_account not in self._actual_balances:
            return f"Unknown account: {from_account} or {to_account}."
        if self._actual_balances[from_account] < amount:
            return f"Insufficient funds in {from_account}: cannot transfer {amount}."
        self._actual_balances[from_account] -= amount
        self._actual_balances[to_account] += amount
        return f"Transferred {amount} from {from_account} to {to_account}."

    def get_balance(self, account: str) -> str:
        if account not in self._actual_balances:
            return f"Unknown account: {account}."
        return f"{account} balance: {self._actual_balances[account]}."

    def submit_final_report(self, balances: dict[str, Any]) -> str:
        self._final_report = balances
        return "Final report recorded."

    @property
    def is_done(self) -> bool:
        return self._final_report is not None

    @property
    def instructions_remaining(self) -> int:
        return len(self._transactions) - self._instruction_cursor

    def call(self, name: str, args: dict[str, Any]) -> str:
        if name == "get_next_instruction":
            return self.get_next_instruction()
        if name == "deposit":
            return self.deposit(str(args.get("account", "")), int(args.get("amount", 0)))
        if name == "withdraw":
            return self.withdraw(str(args.get("account", "")), int(args.get("amount", 0)))
        if name == "transfer":
            return self.transfer(
                str(args.get("from_account", "")),
                str(args.get("to_account", "")),
                int(args.get("amount", 0)),
            )
        if name == "get_balance":
            return self.get_balance(str(args.get("account", "")))
        if name == "submit_final_report":
            balances = args.get("balances", {})
            return self.submit_final_report(balances if isinstance(balances, dict) else {})
        return f"Unknown tool: {name}"

    def verify(self) -> bool:
        """Binary reward (spec §3.3): did the agent's final report exactly
        match every account's reference balance? Reported keys/values are
        normalized (stripped account names, values coerced to int) since the
        model reports through free-form JSON, not a typed API — a
        near-miss on formatting shouldn't count as a wrong answer, but a
        near-miss on the actual numbers should."""
        if self._final_report is None:
            return False
        normalized: dict[str, int] = {}
        for key, value in self._final_report.items():
            try:
                normalized[str(key).strip()] = int(value)
            except (TypeError, ValueError):
                return False
        return normalized == self._reference_balances


@dataclass(frozen=True)
class LedgerRunRecord:
    """One row per (seed, policy) episode — spec §3.4's "task success is the
    honest bottom line" number, plus enough context to see *why* a run
    failed (ran out of budget vs. reported wrong numbers)."""

    seed: int
    policy: str
    n_transactions: int
    n_tool_calls: int
    instructions_remaining: int
    stopped_reason: str
    success: bool


class LedgerRunCollector(RecordCollector[LedgerRunRecord]):
    """Buffers LedgerRunRecords and flushes them to an append-only parquet
    file, same contract as `bench.agreement.AgreementEventCollector`."""
