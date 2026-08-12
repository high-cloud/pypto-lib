# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-side non-causal SWA metadata for the three DSpark draft layers."""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    DSPARK_NUM_LAYERS,
    DSPARK_QUERY_TOKENS,
    DSPARK_QUERY_TOKENS_PADDED,
    DSPARK_SWA_INDEX_WIDTH,
    FLASH as M,
    KV_ORI_MAX_BLOCKS,
)


B_DYN = pl.dynamic("DSPARK_METADATA_B_DYN")
MAX_BLOCKS = KV_ORI_MAX_BLOCKS
WIN = M.sliding_window


@pl.jit.inline
def dspark_metadata_core(
    context_lens: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_NUM_LAYERS, B_DYN, MAX_BLOCKS], pl.INT32],
    slot_mapping: pl.Out[
        pl.Tensor[[DSPARK_NUM_LAYERS, B_DYN, DSPARK_QUERY_TOKENS_PADDED], pl.INT64]
    ],
    swa_indices: pl.Out[
        pl.Tensor[
            [DSPARK_NUM_LAYERS, B_DYN, DSPARK_QUERY_TOKENS_PADDED, DSPARK_SWA_INDEX_WIDTH],
            pl.INT32,
        ]
    ],
    swa_lens: pl.Out[
        pl.Tensor[[DSPARK_NUM_LAYERS, B_DYN, DSPARK_QUERY_TOKENS_PADDED], pl.INT32]
    ],
):
    batch = pl.tensor.dim(context_lens, 0)
    work_items = DSPARK_NUM_LAYERS * batch * DSPARK_QUERY_TOKENS_PADDED
    slots_flat = pl.reshape(slot_mapping, [work_items])
    indices_flat = pl.reshape(swa_indices, [work_items, DSPARK_SWA_INDEX_WIDTH])
    lens_flat = pl.reshape(swa_lens, [work_items])
    for task in pl.spmd(work_items, name_hint="dspark_noncausal_swa_metadata"):
        layer_stride = batch * DSPARK_QUERY_TOKENS_PADDED
        layer = task // layer_stride
        request_query = task % layer_stride
        request = request_query // DSPARK_QUERY_TOKENS_PADDED
        query = request_query % DSPARK_QUERY_TOKENS_PADDED
        index_row = pl.full([1, DSPARK_SWA_INDEX_WIDTH], dtype=pl.INT32, value=-1)
        slot = pl.cast(-1, pl.INT64)
        visible_len = pl.cast(0, pl.INDEX)
        if query < DSPARK_QUERY_TOKENS:
            context_len = pl.cast(pl.read(context_lens, [request]), pl.INDEX)
            anchor_end = context_len + pl.cast(1, pl.INDEX)
            history_start = pl.max(anchor_end - WIN, 0)
            visible_len = anchor_end + DSPARK_QUERY_TOKENS - history_start
            for column in pl.range(DSPARK_SWA_INDEX_WIDTH):
                if column < visible_len:
                    logical_position = history_start + column
                    logical_block = logical_position // BLOCK_SIZE
                    physical_block = pl.read(
                        block_tables,
                        [layer, request, pl.cast(logical_block, pl.INDEX)],
                    )
                    physical_slot = physical_block * BLOCK_SIZE + logical_position % BLOCK_SIZE
                    pl.write(index_row, [0, column], pl.cast(physical_slot, pl.INT32))
            query_position = anchor_end + query
            query_block = query_position // BLOCK_SIZE
            query_physical_block = pl.read(
                block_tables, [layer, request, pl.cast(query_block, pl.INDEX)]
            )
            slot = pl.cast(
                query_physical_block * BLOCK_SIZE + query_position % BLOCK_SIZE, pl.INT64
            )
        indices_flat[task : task + 1, :] = index_row
        pl.write(slots_flat, [task], slot)
        pl.write(lens_flat, [task], pl.cast(visible_len, pl.INT32))
    return slot_mapping, swa_indices, swa_lens


@pl.jit
def dspark_metadata(
    context_lens: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_NUM_LAYERS, B_DYN, MAX_BLOCKS], pl.INT32],
    slot_mapping: pl.Out[
        pl.Tensor[[DSPARK_NUM_LAYERS, B_DYN, DSPARK_QUERY_TOKENS_PADDED], pl.INT64]
    ],
    swa_indices: pl.Out[
        pl.Tensor[
            [DSPARK_NUM_LAYERS, B_DYN, DSPARK_QUERY_TOKENS_PADDED, DSPARK_SWA_INDEX_WIDTH],
            pl.INT32,
        ]
    ],
    swa_lens: pl.Out[
        pl.Tensor[[DSPARK_NUM_LAYERS, B_DYN, DSPARK_QUERY_TOKENS_PADDED], pl.INT32]
    ],
):
    context_lens.bind_dynamic(0, B_DYN)
    block_tables.bind_dynamic(1, B_DYN)
    slot_mapping.bind_dynamic(1, B_DYN)
    swa_indices.bind_dynamic(1, B_DYN)
    swa_lens.bind_dynamic(1, B_DYN)
    return dspark_metadata_core(
        context_lens, block_tables, slot_mapping, swa_indices, swa_lens
    )


def golden_dspark_metadata(tensors):
    from dspark_contract import build_noncausal_swa_metadata

    batch = tensors["context_lens"].numel()
    for layer in range(DSPARK_NUM_LAYERS):
        slots, indices, lens = build_noncausal_swa_metadata(
            tensors["block_tables"][layer], tensors["context_lens"]
        )
        tensors["slot_mapping"][layer] = slots.reshape(batch, DSPARK_QUERY_TOKENS_PADDED)
        tensors["swa_indices"][layer] = indices.reshape(
            batch, DSPARK_QUERY_TOKENS_PADDED, DSPARK_SWA_INDEX_WIDTH
        )
        tensors["swa_lens"][layer] = lens.reshape(batch, DSPARK_QUERY_TOKENS_PADDED)


def build_tensor_specs(batch=4):
    import torch
    from golden import TensorSpec

    from dspark_contract import validate_dspark_batch

    validate_dspark_batch(batch)

    def init_block_tables():
        logical = torch.arange(MAX_BLOCKS, dtype=torch.int32)
        layers = [logical + layer * MAX_BLOCKS for layer in range(DSPARK_NUM_LAYERS)]
        return torch.stack(layers).unsqueeze(1).expand(-1, batch, -1).contiguous()

    return [
        TensorSpec(
            "context_lens",
            [batch],
            torch.int32,
            init_value=lambda: torch.tensor([1, 31, 32, 129], dtype=torch.int32).repeat(
                (batch + 3) // 4
            )[:batch],
        ),
        TensorSpec(
            "block_tables",
            [DSPARK_NUM_LAYERS, batch, MAX_BLOCKS],
            torch.int32,
            init_value=init_block_tables,
        ),
        TensorSpec(
            "slot_mapping",
            [DSPARK_NUM_LAYERS, batch, DSPARK_QUERY_TOKENS_PADDED],
            torch.int64,
            is_output=True,
        ),
        TensorSpec(
            "swa_indices",
            [DSPARK_NUM_LAYERS, batch, DSPARK_QUERY_TOKENS_PADDED, DSPARK_SWA_INDEX_WIDTH],
            torch.int32,
            is_output=True,
        ),
        TensorSpec(
            "swa_lens",
            [DSPARK_NUM_LAYERS, batch, DSPARK_QUERY_TOKENS_PADDED],
            torch.int32,
            is_output=True,
        ),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash DSpark SWA metadata validation.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4, choices=[4, 8, 12, 16])
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    result = run_jit(
        fn=dspark_metadata,
        specs=build_tensor_specs(args.batch),
        golden_fn=golden_dspark_metadata,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
