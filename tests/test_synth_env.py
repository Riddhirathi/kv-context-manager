from __future__ import annotations

from agentkv.bench.tasks.synth_env import SyntheticIncidentEnv


def test_same_seed_is_deterministic():
    env_a = SyntheticIncidentEnv(seed=7)
    env_b = SyntheticIncidentEnv(seed=7)
    assert env_a.call("list_files", {}).output == env_b.call("list_files", {}).output
    path = env_a.call("list_files", {}).output.splitlines()[0]
    assert env_a.call("read_file", {"path": path}).output == env_b.call(
        "read_file", {"path": path}
    ).output


def test_different_seeds_diverge():
    env_a = SyntheticIncidentEnv(seed=1)
    env_b = SyntheticIncidentEnv(seed=2)
    assert env_a.call("list_files", {}).output != env_b.call("list_files", {}).output


def test_read_file_repeated_calls_are_stable_within_one_env():
    env = SyntheticIncidentEnv(seed=3)
    path = [p for p in env.call("list_files", {}).output.splitlines() if p.startswith("logs/")][0]
    first = env.call("read_file", {"path": path}).output
    second = env.call("read_file", {"path": path}).output
    assert first == second


def test_read_unknown_file_is_an_error():
    env = SyntheticIncidentEnv(seed=4)
    result = env.call("read_file", {"path": "does/not/exist.py"})
    assert result.is_error


def test_log_files_contain_mixed_small_and_large_outputs():
    env = SyntheticIncidentEnv(seed=5)
    files = env.call("list_files", {}).output.splitlines()
    lens = {p: len(env.call("read_file", {"path": p}).output) for p in files}
    source_lens = [n for p, n in lens.items() if not p.startswith("logs/")]
    log_lens = [n for p, n in lens.items() if p.startswith("logs/")]
    assert max(log_lens) > max(source_lens)


def test_search_logs_finds_injected_error():
    env = SyntheticIncidentEnv(seed=6)
    # Force log generation.
    for p in env.call("list_files", {}).output.splitlines():
        if p.startswith("logs/"):
            env.call("read_file", {"path": p})
    result = env.call("search_logs", {"query": "ERROR"})
    assert "ERROR" in result.output


def test_unknown_tool_is_an_error():
    env = SyntheticIncidentEnv(seed=8)
    result = env.call("not_a_real_tool", {})
    assert result.is_error
