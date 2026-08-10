"""`KVConnectorBase_V1` glue for `kv/offload.py`'s trajectory-aware policy
(AGENTKV_SPEC.md §4.3).

Deliberately split from `kv/offload.py` — see that module's docstring — this
file imports real vLLM classes at module level and is therefore only ever
imported from *inside* the vLLM server subprocess (via
`experiments/_vllm_server_with_offload.py`, the launcher script that
registers this connector with vLLM's `KVConnectorFactory` before vLLM parses
`--kv-transfer-config`), never from this project's own test suite or
Windows-side scripts.

Adapted from vLLM's reference `SharedStorageConnector`
(`vllm/distributed/kv_transfer/kv_connector/v1/shared_storage_connector.py`,
a disk-backed debug connector) — same request-level store/load protocol, but:

1. Storage is an in-memory CPU tensor dict (`self._host_ram`), not disk files
   — spec's "host RAM," not disk/NVMe.
2. `build_connector_meta` only marks a request for storage if
   `_worth_storing_mask` (backed by `kv.offload.classify_token_ids` and
   `blocks_worth_offloading`) says at least one block is — a plain
   LRU/always-offload tier would persist everything indiscriminately.
   Classification decodes `request.prompt_token_ids` with this connector's
   own tokenizer (loaded once in `__init__`) rather than being told the
   answer by an external driver — this project's `VLLMEngine` talks to vLLM
   only over HTTP, and this connector runs *inside* vLLM's own process, so
   there is no in-process call an outside driver could make to hand over a
   `ContextState` even if it wanted to. See `kv/offload.py`'s module
   docstring for why this replaced an earlier, broken same-process design.

Not independently verified end-to-end against every attention backend vLLM
supports — this project only ever runs Qwen3 (non-MLA, GQA attention), so the
tensor reshape logic below only implements that path, unlike the reference
connector's separate MLA branch.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Protocol

import torch
from transformers import AutoTokenizer
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.shared_storage_connector import (
    SharedStorageConnectorMetadata,
)
from vllm.logger import init_logger

from agentkv.kv.offload import (
    blocks_worth_offloading,
    classify_token_ids,
    token_worth_offloading_mask,
)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

# vLLM creates a SEPARATE connector instance per role — one for
# KVConnectorRole.SCHEDULER (co-located with the scheduler, does the
# store/load *decisions*: get_num_new_matched_tokens, build_connector_meta)
# and one for KVConnectorRole.WORKER (co-located with the worker, does the
# actual tensor save/load against the paged KV buffer: save_kv_layer,
# start_load_kv) — see KVConnectorBase_V1's own docstring. Confirmed
# empirically (not assumed): every boot logs "Creating v1 connector with
# name: TrajectoryAwareOffloadConnector" twice. An earlier version of this
# module stored offloaded blocks in `self._host_ram` as a plain *instance*
# dict — invisible across the two separate instances, so every lookup
# reported "0 keys in host_ram" even right after a store, and every
# supposedly-offloaded request silently fell back to full re-prefill instead
# of an external cache hit. Fixed by making storage a *module-level* dict:
# both role-instances live in the same Python process for this project's
# single-GPU, non-distributed setup (confirmed empirically: no second worker
# subprocess appears in `ps aux` during a boot with this connector active),
# so a module-level singleton is visible to both without needing real
# inter-process sharing (unlike the reference `SharedStorageConnector`,
# which uses disk files specifically because that mechanism *does* survive
# a real multi-process/multi-node split — this project never needs that).
_HOST_RAM: dict[str, torch.Tensor] = {}


class _HasPromptTokenIds(Protocol):
    """Structural type covering both vLLM's `Request` (used by
    `get_num_new_matched_tokens`) and `NewRequestData` (used by
    `build_connector_meta`'s `scheduler_output.scheduled_new_reqs`) — two
    distinct vLLM types that both expose `prompt_token_ids`, the only
    attribute `_found_match_for_request` actually needs from either."""

    prompt_token_ids: list[int]


class TrajectoryAwareOffloadConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Request] = {}
        self._host_ram = _HOST_RAM
        # Own tokenizer, loaded once here — see module docstring for why
        # this connector must self-classify rather than being told by an
        # external driver.
        self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            vllm_config.model_config.model
        )

    def _worth_storing_mask(self, token_ids: list[int], num_blocks: int) -> list[bool]:
        spans = classify_token_ids(token_ids, self._tokenizer.decode)
        total_tokens = spans[-1].end if spans else 0
        mask = token_worth_offloading_mask(spans, total_tokens)
        per_block = blocks_worth_offloading(mask, self._block_size)
        if len(per_block) < num_blocks:
            per_block = per_block + [True] * (num_blocks - len(per_block))
        return per_block[:num_blocks]

    def _content_hash(self, token_ids: torch.Tensor) -> str:
        return hashlib.md5(  # noqa: S324 - content addressing, not security
            token_ids.numpy().tobytes(), usedforsecurity=False
        ).hexdigest()

    def _key(self, layer_name: str, token_ids: torch.Tensor) -> str:
        return f"{layer_name}:{self._content_hash(token_ids)}"

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, SharedStorageConnectorMetadata)
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return
        for request in metadata.requests:
            if request.is_store:
                continue
            for layer_name in forward_context.no_compile_layers:
                attn_layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = attn_layer.kv_cache[forward_context.virtual_engine]
                key = self._key(layer_name, request.token_ids)
                if key not in self._host_ram:
                    continue  # not offloaded (skipped by policy, or never stored)
                self._inject_kv_into_layer(
                    kv_cache_layer,
                    self._host_ram[key].to(kv_cache_layer.device),
                    request.slot_mapping,
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, SharedStorageConnectorMetadata)
        for request in metadata.requests:
            if not request.is_store:
                continue
            key = self._key(layer_name, request.token_ids)
            kv_cache = self._extract_kv_from_layer(kv_layer, request.slot_mapping)
            self._host_ram[key] = kv_cache.detach().cpu()

    def wait_for_save(self) -> None:
        return

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> int:
        num_tokens_to_check = self._align_to_block_size(len(request.prompt_token_ids) - 1)
        if not self._found_match_for_request(request, num_tokens_to_check):
            logger.info(
                "TrajectoryAwareOffloadConnector: external cache MISS for request %s "
                "(%d tokens to check, %d keys in host_ram)",
                request.request_id,
                num_tokens_to_check,
                len(self._host_ram),
            )
            return 0
        logger.info(
            "TrajectoryAwareOffloadConnector: external cache HIT for request %s, %d tokens "
            "available beyond the %d already computed",
            request.request_id,
            num_tokens_to_check,
            num_computed_tokens,
        )
        return num_tokens_to_check - num_computed_tokens

    def update_state_after_alloc(self, request: Request, num_external_tokens: int) -> None:
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> SharedStorageConnectorMetadata:
        meta = SharedStorageConnectorMetadata()  # type: ignore[no-untyped-call]
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    token_ids=new_req.prompt_token_ids,
                    block_ids=new_req.block_ids,
                    block_size=self._block_size,
                    is_store=False,
                )
            else:
                num_tokens_to_check = self._align_to_block_size(len(new_req.prompt_token_ids) - 1)
                if not self._found_match_for_request(new_req, num_tokens_to_check):
                    num_blocks = len(new_req.block_ids)
                    worth_mask = self._worth_storing_mask(new_req.prompt_token_ids, num_blocks)
                    logger.info(
                        "TrajectoryAwareOffloadConnector: %d/%d blocks worth offloading for "
                        "request %s",
                        sum(worth_mask),
                        num_blocks,
                        new_req.req_id,
                    )
                    if any(worth_mask):
                        meta.add_request(
                            token_ids=new_req.prompt_token_ids,
                            block_ids=new_req.block_ids,
                            block_size=self._block_size,
                            is_store=True,
                        )
        self._requests_need_load.clear()
        return meta

    # ==============================
    # Helpers (mirroring SharedStorageConnector's, in-memory instead of disk)
    # ==============================

    def _align_to_block_size(self, num_tokens: int) -> int:
        return (num_tokens - 1) // self._block_size * self._block_size

    def _found_match_for_request(
        self, request: _HasPromptTokenIds, num_tokens_to_check: int
    ) -> bool:
        token_ids = torch.tensor(request.prompt_token_ids)[:num_tokens_to_check]
        digest = self._content_hash(token_ids)
        return any(k.endswith(f":{digest}") for k in self._host_ram)

    def _inject_kv_into_layer(
        self,
        dst_kv_cache_layer: torch.Tensor,
        src_kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        shape = dst_kv_cache_layer.shape
        num_pages, page_size = shape[1], shape[2]
        reshaped = dst_kv_cache_layer.reshape(2, num_pages * page_size, -1)
        reshaped[:, slot_mapping, ...] = src_kv_cache
        dst_kv_cache_layer.reshape(shape)

    def _extract_kv_from_layer(
        self, layer: torch.Tensor, slot_mapping: torch.Tensor
    ) -> torch.Tensor:
        num_pages, page_size = layer.shape[1], layer.shape[2]
        return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]
