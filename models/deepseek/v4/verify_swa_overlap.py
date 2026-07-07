# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch-golden check for SWA overlap equivalence: [t1, t2] -> [t2, t3]."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]


def _prepare_import_path() -> None:
    for path in (str(REPO_ROOT), str(THIS_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _patch_decode_shape(batch: int, seq: int):
    config = importlib.import_module("config")
    config.DECODE_BATCH = batch
    config.DECODE_SEQ = seq
    config.DECODE_TOKENS = batch * seq
    config.MOE_TOKENS = config.DECODE_TOKENS
    config.DECODE_ORI_BLOCK_NUM = batch * config.KV_ORI_MAX_BLOCKS
    config.DECODE_CMP_BLOCK_NUM = batch * config.KV_CMP_MAX_BLOCKS
    config.DECODE_IDX_BLOCK_NUM = batch * config.IDX_CACHE_MAX_BLOCKS
    return config


def _make_tensors(attn_swa) -> dict[str, torch.Tensor]:
    metadata = importlib.import_module("decode_metadata")
    tensors = {spec.name: spec.create_tensor().clone() for spec in attn_swa.build_tensor_specs()}
    tensors["block_table"] = metadata.block_table(
        batch=attn_swa.B,
        table_blocks=attn_swa.ORI_MAX_BLOCKS,
        physical_blocks=attn_swa.ORI_MAX_BLOCKS,
    )
    return tensors


def _clone_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in tensors.items()}


def _set_request_round(
    tensors: dict[str, torch.Tensor],
    *,
    row: int,
    start_pos: int,
    x_first: torch.Tensor,
    x_second: torch.Tensor,
    attn_swa,
    metadata,
    config,
) -> None:
    starts = tensors["position_ids"].reshape(attn_swa.B, attn_swa.S)[:, 0].clone().to(torch.int32)
    starts[row] = int(start_pos)
    positions = metadata.position_ids_from_starts(starts, seq=attn_swa.S)
    tensors["position_ids"] = positions.reshape(-1).contiguous()
    tensors["swa_slot_mapping"] = metadata.paged_slot_mapping(
        positions,
        tensors["block_table"],
        block_size=config.BLOCK_SIZE,
    ).reshape(-1).contiguous()
    swa_indices, swa_lens = metadata.swa_indices_and_lens(
        positions,
        tensors["block_table"],
        block_size=config.BLOCK_SIZE,
        window=attn_swa.WIN,
    )
    tensors["swa_indices"] = swa_indices.contiguous()
    tensors["swa_lens"] = swa_lens.contiguous()

    x_hc = tensors["x_hc"].clone()
    base = row * attn_swa.S
    x_hc[base + 0] = x_first
    x_hc[base + 1] = x_second
    tensors["x_hc"] = x_hc
    tensors["x_out"] = torch.zeros_like(tensors["x_out"])


def _print_top_diffs(abs_diff: torch.Tensor, a: torch.Tensor, b: torch.Tensor, limit: int) -> None:
    flat = abs_diff.flatten()
    count = min(limit, flat.numel())
    if count == 0:
        return
    vals, idxs = torch.topk(flat, k=count)
    hidden_dim = a.shape[-1]
    print("top diffs:")
    for rank, (val, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        hc = idx // hidden_dim
        col = idx % hidden_dim
        print(
            f"  #{rank}: hc={hc} dim={col} "
            f"abs={val:.8g} first={float(a.flatten()[idx]):.8g} second={float(b.flatten()[idx]):.8g}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify whether SWA hidden for t2 matches across [t1,t2] then [t2,t3]."
    )
    parser.add_argument("--batch", type=int, default=4, help="Decode batch for the in-process torch golden shape.")
    parser.add_argument("--seq", type=int, default=2, help="Decode sequence. This check requires seq=2.")
    parser.add_argument("--row", type=int, default=0, help="Batch row to inspect.")
    parser.add_argument("--t1-pos", type=int, default=None, help="Absolute position for t1; default is WIN - 1.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    if args.seq != 2:
        raise SystemExit("This overlap check is defined for --seq 2.")
    if args.batch * args.seq != 8:
        raise SystemExit("Use a decode shape with batch * seq == 8 to match the current SWA tile contract.")
    if not 0 <= args.row < args.batch:
        raise SystemExit(f"--row must be in [0, {args.batch}), got {args.row}.")

    _prepare_import_path()
    config = _patch_decode_shape(args.batch, args.seq)
    attn_swa = importlib.import_module("decode_attention_swa")
    metadata = importlib.import_module("decode_metadata")

    if attn_swa.S != 2:
        raise SystemExit(f"Imported SWA shape has S={attn_swa.S}, expected 2.")

    t1_pos = attn_swa.WIN - 1 if args.t1_pos is None else args.t1_pos
    if not 0 <= t1_pos <= attn_swa.MAX_SEQ_LEN - 3:
        raise SystemExit(f"--t1-pos must allow t1,t2,t3 inside RoPE table, got {t1_pos}.")

    torch.manual_seed(args.seed)
    base = _make_tensors(attn_swa)
    row_base = args.row * attn_swa.S
    x_t1 = base["x_hc"][row_base + 0].clone()
    x_t2 = base["x_hc"][row_base + 1].clone()
    x_t3 = torch.empty_like(x_t2).uniform_(-1.0, 1.0)

    first = _clone_tensors(base)
    _set_request_round(
        first,
        row=args.row,
        start_pos=t1_pos,
        x_first=x_t1,
        x_second=x_t2,
        attn_swa=attn_swa,
        metadata=metadata,
        config=config,
    )
    attn_swa.golden_attention_swa(first)
    t2_from_first = first["x_out"][row_base + 1].clone()

    second = _clone_tensors(base)
    second["kv_cache"] = first["kv_cache"].clone()
    _set_request_round(
        second,
        row=args.row,
        start_pos=t1_pos + 1,
        x_first=x_t2,
        x_second=x_t3,
        attn_swa=attn_swa,
        metadata=metadata,
        config=config,
    )
    attn_swa.golden_attention_swa(second)
    t2_from_second = second["x_out"][row_base + 0].clone()

    a = t2_from_first.float()
    b = t2_from_second.float()
    abs_diff = torch.abs(a - b)
    rel_den = torch.maximum(torch.maximum(torch.abs(a), torch.abs(b)), torch.full_like(a, 1e-6))
    rel_diff = abs_diff / rel_den
    allowed = args.atol + args.rtol * torch.abs(b)
    over = abs_diff > allowed

    first_pos = first["position_ids"].reshape(attn_swa.B, attn_swa.S)[args.row].tolist()
    second_pos = second["position_ids"].reshape(attn_swa.B, attn_swa.S)[args.row].tolist()
    first_slots = first["swa_slot_mapping"].reshape(attn_swa.B, attn_swa.S)[args.row].tolist()
    second_slots = second["swa_slot_mapping"].reshape(attn_swa.B, attn_swa.S)[args.row].tolist()
    block_id = int(first["block_table"][args.row, 0].item())

    print("SWA overlap check: [t1,t2] -> [t2,t3]")
    print(f"shape: B={attn_swa.B} S={attn_swa.S} T={attn_swa.T}")
    print(f"row={args.row} block_table[row,0]={block_id}")
    print(f"round1 positions={first_pos} slot_mapping={first_slots}")
    print(f"round2 positions={second_pos} slot_mapping={second_slots}")
    print(
        "movement: round1 writes [t1,t2] into SWA cache before attention; "
        "round2 reuses t2 through physical swa_indices."
    )
    print(f"threshold: atol={args.atol:g} rtol={args.rtol:g}")
    print(f"allclose={torch.allclose(a, b, atol=args.atol, rtol=args.rtol)}")
    print(f"max_abs={float(abs_diff.max()):.8g}")
    print(f"mean_abs={float(abs_diff.mean()):.8g}")
    print(f"p99_abs={float(torch.quantile(abs_diff.flatten(), 0.99)):.8g}")
    print(f"max_rel={float(rel_diff.max()):.8g}")
    print(f"over_threshold={int(over.sum().item())}/{over.numel()}")
    _print_top_diffs(abs_diff, a, b, args.topk)

    return 0 if torch.allclose(a, b, atol=args.atol, rtol=args.rtol) else 1


if __name__ == "__main__":
    raise SystemExit(main())
