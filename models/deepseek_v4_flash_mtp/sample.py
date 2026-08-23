# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Greedy and counter-based Gumbel-max sampling for DeepSeek-V4 logits."""

import pypto.language as pl

from config import DECODE_TOKENS, FLASH as M, FP32_NEG_INF


# model config
VOCAB = M.vocab_size
SAMPLE_ROWS = DECODE_TOKENS
SAMPLED_IDS_PAD = 8

# tiling
SAMPLE_ROW_WIDTH_TILE = 256
SAMPLE_BLOCK_ROWS_TILE = 8
SAMPLE_GRID_ROWS = VOCAB // SAMPLE_ROW_WIDTH_TILE
SAMPLE_FULL_BLOCKS = SAMPLE_GRID_ROWS // SAMPLE_BLOCK_ROWS_TILE
SAMPLE_TAIL_ROWS = SAMPLE_GRID_ROWS % SAMPLE_BLOCK_ROWS_TILE
SAMPLE_CANDIDATES_TILE = 512

# sampling constants
SAMPLING_EPS = 1e-5
SAMPLING_EPS_INV = 100000.0
RANDOM_KEY_MODULUS = 1073741824
HASH_MULTIPLIER = 0x045D9F3B
POSITION_MULTIPLIER = 65537
UINT23_SCALE = 1.0 / 8388608.0


@pl.jit.inline
def _counter_gumbel(
    block: pl.Scalar[pl.INDEX],
    random_key: pl.Scalar[pl.INT32],
):
    """Generate one Gumbel-noise tile from a key and vocabulary counters."""
    counter_zeros = pl.full([SAMPLE_BLOCK_ROWS_TILE, SAMPLE_ROW_WIDTH_TILE], dtype=pl.INT32, value=0)
    column_ids = pl.arange(0, [1, SAMPLE_ROW_WIDTH_TILE], dtype=pl.INT32)
    column_counters = pl.col_expand(counter_zeros, column_ids)
    row_counters_pad = pl.full([SAMPLE_BLOCK_ROWS_TILE, SAMPLE_BLOCK_ROWS_TILE], dtype=pl.INT32, value=0)
    for row in pl.range(SAMPLE_BLOCK_ROWS_TILE):
        row_counter = pl.cast(row * SAMPLE_ROW_WIDTH_TILE, pl.INT32)
        pl.write(row_counters_pad, [row, 0], row_counter)
    row_counters = pl.row_max(row_counters_pad)
    counters = pl.row_expand_add(column_counters, row_counters)
    block_base = pl.cast(block * SAMPLE_BLOCK_ROWS_TILE * SAMPLE_ROW_WIDTH_TILE, pl.INT32)
    counters = pl.add(counters, block_base)

    key_zeros = pl.mul(column_ids, pl.cast(0, pl.INT32))
    key_row = pl.add(key_zeros, random_key)
    random_key_tile = pl.col_expand(counter_zeros, key_row)
    positive_mask = pl.full([SAMPLE_BLOCK_ROWS_TILE, SAMPLE_ROW_WIDTH_TILE], dtype=pl.INT32, value=0x7FFFFFFF)
    random_bits = pl.xor(counters, random_key_tile)
    shifted = pl.shrs(random_bits, 16)
    random_bits = pl.xor(random_bits, shifted)
    random_bits = pl.mul(random_bits, pl.cast(HASH_MULTIPLIER, pl.INT32))
    random_bits = pl.and_(random_bits, positive_mask)
    shifted = pl.shrs(random_bits, 16)
    random_bits = pl.xor(random_bits, shifted)
    random_bits = pl.mul(random_bits, pl.cast(HASH_MULTIPLIER, pl.INT32))
    random_bits = pl.and_(random_bits, positive_mask)
    shifted = pl.shrs(random_bits, 16)
    random_bits = pl.xor(random_bits, shifted)

    uniform_bits = pl.shrs(random_bits, 8)
    uniform_fp32 = pl.cast(uniform_bits, pl.FP32)
    uniform_centered = pl.add(uniform_fp32, 0.5)
    uniform = pl.mul(uniform_centered, UINT23_SCALE)
    log_uniform = pl.log(uniform)
    negative_log_uniform = pl.neg(log_uniform)
    log_negative_log_uniform = pl.log(negative_log_uniform)
    return pl.neg(log_negative_log_uniform)


@pl.jit.inline
def sample(
    logits: pl.Tensor[[SAMPLE_ROWS, VOCAB], pl.FP32],
    sampling_temperatures: pl.Tensor[[SAMPLE_ROWS], pl.FP32],
    sampling_seeds: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    sampling_positions: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    sampled_ids: pl.Tensor[[SAMPLE_ROWS, SAMPLED_IDS_PAD], pl.INT32],
):
    """Sample each logits row with greedy or counter-based Gumbel-max."""
    logits_grid = pl.reshape(logits, [SAMPLE_ROWS * SAMPLE_GRID_ROWS, SAMPLE_ROW_WIDTH_TILE])
    for row in pl.spmd(SAMPLE_ROWS, name_hint="sample_gumbel_argmax"):
        temperature = pl.read(sampling_temperatures, [row])
        sampling_enabled = pl.cast(pl.mul(temperature, SAMPLING_EPS_INV), pl.INT32)
        seed = pl.read(sampling_seeds, [row])
        position = pl.read(sampling_positions, [row])
        seed_index = pl.cast(seed, pl.INDEX)
        position_index = pl.cast(position, pl.INDEX)
        random_key_index = seed_index + position_index * POSITION_MULTIPLIER
        random_key = pl.cast(random_key_index % RANDOM_KEY_MODULUS, pl.INT32)
        row_base = row * SAMPLE_GRID_ROWS
        candidate_maxima = pl.full([SAMPLE_BLOCK_ROWS_TILE, SAMPLE_CANDIDATES_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
        candidate_token_ids = pl.full([1, SAMPLE_CANDIDATES_TILE], dtype=pl.INT32, value=0)
        for block in pl.range(SAMPLE_FULL_BLOCKS):
            block_row = row_base + block * SAMPLE_BLOCK_ROWS_TILE
            scores = logits_grid[
                block_row : block_row + SAMPLE_BLOCK_ROWS_TILE,
                0:SAMPLE_ROW_WIDTH_TILE,
            ]
            if sampling_enabled >= 1:
                scaled_scores = pl.div(scores, temperature)
                gumbel_noise = _counter_gumbel(block, random_key)
                scores = pl.add(scaled_scores, gumbel_noise)
            local_winners = pl.row_argmax(scores)
            for lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
                local_token = pl.read(local_winners, [lane, 0])
                local_score = pl.read(scores, [lane, pl.cast(local_token, pl.INDEX)])
                candidate = block * SAMPLE_BLOCK_ROWS_TILE + lane
                block_token_base = block * SAMPLE_BLOCK_ROWS_TILE * SAMPLE_ROW_WIDTH_TILE
                token_base = block_token_base + lane * SAMPLE_ROW_WIDTH_TILE
                token_id = pl.cast(token_base, pl.INT32) + local_token
                pl.write(candidate_maxima, [0, candidate], local_score)
                pl.write(candidate_token_ids, [0, candidate], token_id)

        tail_row = row_base + SAMPLE_FULL_BLOCKS * SAMPLE_BLOCK_ROWS_TILE
        tail_scores = logits_grid[tail_row : tail_row + SAMPLE_TAIL_ROWS, 0:SAMPLE_ROW_WIDTH_TILE]
        if sampling_enabled >= 1:
            tail_scaled_scores = pl.div(tail_scores, temperature)
            tail_gumbel_noise = _counter_gumbel(SAMPLE_FULL_BLOCKS, random_key)
            tail_noise = tail_gumbel_noise[0:SAMPLE_TAIL_ROWS, 0:SAMPLE_ROW_WIDTH_TILE]
            tail_scores = pl.add(tail_scaled_scores, tail_noise)
        scores = pl.full([SAMPLE_BLOCK_ROWS_TILE, SAMPLE_ROW_WIDTH_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
        scores[0:SAMPLE_TAIL_ROWS, 0:SAMPLE_ROW_WIDTH_TILE] = tail_scores
        for lane in pl.range(SAMPLE_TAIL_ROWS, SAMPLE_BLOCK_ROWS_TILE):
            scores[lane : lane + 1, 0:SAMPLE_ROW_WIDTH_TILE] = tail_scores[0:1, 0:SAMPLE_ROW_WIDTH_TILE]
        local_winners = pl.row_argmax(scores)
        for lane in pl.range(SAMPLE_TAIL_ROWS):
            local_token = pl.read(local_winners, [lane, 0])
            local_score = pl.read(scores, [lane, pl.cast(local_token, pl.INDEX)])
            candidate = SAMPLE_FULL_BLOCKS * SAMPLE_BLOCK_ROWS_TILE + lane
            token_base = SAMPLE_FULL_BLOCKS * SAMPLE_BLOCK_ROWS_TILE * SAMPLE_ROW_WIDTH_TILE
            token_id = pl.cast(token_base + lane * SAMPLE_ROW_WIDTH_TILE, pl.INT32) + local_token
            pl.write(candidate_maxima, [0, candidate], local_score)
            pl.write(candidate_token_ids, [0, candidate], token_id)

        winning_candidate = pl.read(pl.row_argmax(candidate_maxima), [0, 0])
        best_index = pl.read(candidate_token_ids, [0, pl.cast(winning_candidate, pl.INDEX)])

        sampled_row = pl.create_tensor([1, SAMPLED_IDS_PAD], dtype=pl.INT32)
        sampled_fill = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
        sampled_row[:, :] = sampled_fill
        pl.write(sampled_row, [0, 0], best_index)
        sampled_ids[row : row + 1, :] = sampled_row
    return sampled_ids


@pl.jit
def sample_test(
    logits: pl.Tensor[[SAMPLE_ROWS, VOCAB], pl.FP32],
    temperatures: pl.Tensor[[SAMPLE_ROWS], pl.FP32],
    seeds: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    positions: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[SAMPLE_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    return sample(logits, temperatures, seeds, positions, sampled_ids)


def _gumbel_noise(seed, position):
    import numpy as np

    counters = np.arange(VOCAB, dtype=np.uint32)
    random_key = np.uint32((seed + position * POSITION_MULTIPLIER) % RANDOM_KEY_MODULUS)
    random_bits = counters ^ random_key
    random_bits ^= random_bits >> np.uint32(16)
    random_bits *= np.uint32(HASH_MULTIPLIER)
    random_bits &= np.uint32(0x7FFFFFFF)
    random_bits ^= random_bits >> np.uint32(16)
    random_bits *= np.uint32(HASH_MULTIPLIER)
    random_bits &= np.uint32(0x7FFFFFFF)
    random_bits ^= random_bits >> np.uint32(16)
    uniform_bits = (random_bits >> np.uint32(8)).astype(np.float32)
    uniform_centered = uniform_bits + np.float32(0.5)
    uniform = uniform_centered * np.float32(UINT23_SCALE)
    return -np.log(-np.log(uniform))


def build_tensor_specs(sampling_temperature=None):
    import torch

    from golden import TensorSpec

    def init_logits():
        generator = torch.Generator().manual_seed(20260821)
        logits = torch.randn(SAMPLE_ROWS, VOCAB, generator=generator, dtype=torch.float32)
        logits[0, 7] = 20.0
        if SAMPLE_ROWS > 4:
            logits[4, 42] = 20.0
        return logits

    def repeat_rows(values, dtype):
        repeats = (SAMPLE_ROWS + len(values) - 1) // len(values)
        return torch.tensor(values, dtype=dtype).repeat(repeats)[:SAMPLE_ROWS]

    return [
        TensorSpec("logits", [SAMPLE_ROWS, VOCAB], torch.float32, init_value=init_logits),
        TensorSpec(
            "temperatures",
            [SAMPLE_ROWS],
            torch.float32,
            init_value=lambda: (
                repeat_rows([0.0, 0.3, 0.7, 1.0, 0.0, 0.5, 1.3, 2.0], torch.float32)
                if sampling_temperature is None
                else torch.full([SAMPLE_ROWS], sampling_temperature, dtype=torch.float32)
            ),
        ),
        TensorSpec(
            "seeds",
            [SAMPLE_ROWS],
            torch.int32,
            init_value=lambda: repeat_rows([1, 7, 19, 1234, 42, 99, 2026, 65537], torch.int32),
        ),
        TensorSpec(
            "positions",
            [SAMPLE_ROWS],
            torch.int32,
            init_value=lambda: repeat_rows([0, 1, 17, 1024, 4, 55, 4096, 32767], torch.int32),
        ),
        TensorSpec("sampled_ids", [SAMPLE_ROWS, SAMPLED_IDS_PAD], torch.int32, is_output=True),
    ]


def golden_sample(tensors):
    import torch

    tensors["sampled_ids"].zero_()
    for row in range(SAMPLE_ROWS):
        logits = tensors["logits"][row].float()
        temperature = float(tensors["temperatures"][row])
        if temperature < SAMPLING_EPS:
            selected = torch.argmax(logits)
        else:
            seed = int(tensors["seeds"][row])
            position = int(tensors["positions"][row])
            noise = torch.from_numpy(_gumbel_noise(seed, position))
            selected = torch.argmax(logits / temperature + noise)
        tensors["sampled_ids"][row, 0] = selected.to(torch.int32)


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()
    assert args.temperature is None or args.temperature >= 0.0, (
        f"--temperature must be non-negative, got {args.temperature}"
    )

    result = run_jit(
        fn=sample_test,
        specs=build_tensor_specs(args.temperature),
        golden_fn=golden_sample,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
