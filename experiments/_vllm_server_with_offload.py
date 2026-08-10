#!/usr/bin/env python3
"""Thin launcher for §4.3's `TrajectoryAwareOffloadConnector` (AGENTKV_SPEC.md
§4.3) — registers it with vLLM's `KVConnectorFactory` before delegating to
vLLM's own OpenAI-server entrypoint, then runs the identical startup sequence
vLLM's own `if __name__ == "__main__":` block runs (see
`vllm/entrypoints/openai/api_server.py`'s tail).

Necessary because vLLM 0.8.5's `KVConnectorFactory` registry is populated by a
hardcoded list at import time (see `kv/offload_connector.py`'s module
docstring) — there is no CLI flag to load an arbitrary external connector
module path in this version. This project's `VLLMEngine` spawns vLLM as a
*separate OS process* (`subprocess.Popen([sys.executable, "-m",
"vllm.entrypoints.openai.api_server", ...])`), so a plain in-process import in
our own script's interpreter would register the connector in the wrong
process entirely — this launcher runs *as* that subprocess instead of
`-m vllm.entrypoints.openai.api_server`, with everything else identical.

Usage: exactly the same CLI arguments `vllm serve` / `-m vllm.entrypoints.
openai.api_server` accepts — this script forwards to the same arg parser.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory  # noqa: E402

KVConnectorFactory.register_connector(
    "TrajectoryAwareOffloadConnector",
    "agentkv.kv.offload_connector",
    "TrajectoryAwareOffloadConnector",
)

import uvloop  # noqa: E402
from vllm.entrypoints.openai.api_server import cli_env_setup, run_server  # noqa: E402
from vllm.entrypoints.openai.cli_args import (  # noqa: E402
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.utils import FlexibleArgumentParser  # noqa: E402

if __name__ == "__main__":
    cli_env_setup()
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))
