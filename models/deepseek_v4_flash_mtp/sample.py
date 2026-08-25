# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Greedy and temperature/top-k/Gumbel sampling for DeepSeek-V4 logits."""

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
TOPK_MAX = 64
TOPK_GROUP_WIDTH = 2048
TOPK_NUM_FULL_GROUPS = VOCAB // TOPK_GROUP_WIDTH
TOPK_GROUP_TAIL = VOCAB % TOPK_GROUP_WIDTH
TOPK_CANDIDATE_PAD = 4096
TOPK_FINAL_HALF = TOPK_CANDIDATE_PAD // 2

# sampling constants
SAMPLING_EPS = 1e-5
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
def apply_temperature(
    logits: pl.Tensor[[SAMPLE_BLOCK_ROWS_TILE, SAMPLE_ROW_WIDTH_TILE], pl.FP32],
    temperature: pl.Scalar[pl.FP32],
):
    """Scale logits by temperature."""
    return pl.div(logits, temperature)


@pl.jit.inline
def _topk_group_pairs(
    logits: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    group: pl.Scalar[pl.INDEX],
    temperature: pl.Scalar[pl.FP32],
):
    """Return the largest TOPK_MAX scaled logits from one full group."""
    group_start = group * TOPK_GROUP_WIDTH
    scores = pl.slice(logits, [1, TOPK_GROUP_WIDTH], [row, group_start])
    scores_tile = pl.reshape(scores, [SAMPLE_BLOCK_ROWS_TILE, SAMPLE_ROW_WIDTH_TILE])
    scaled_scores_tile = apply_temperature(scores_tile, temperature)
    scaled_scores = pl.reshape(scaled_scores_tile, [1, TOPK_GROUP_WIDTH])
    indices = pl.arange(pl.cast(group_start, pl.UINT32), [1, TOPK_GROUP_WIDTH], dtype=pl.UINT32)
    pairs = pl.sort32(scaled_scores, indices)
    pairs = pl.mrgsort(pairs, block_len=64)
    pairs = pl.mrgsort(pairs, block_len=256)
    pairs = pl.mrgsort(pairs, block_len=1024)
    return pairs[:, 0 : 2 * TOPK_MAX]


@pl.jit.inline
def gumbel_sample(
    candidate_values: pl.Tensor[[1, TOPK_MAX], pl.FP32],
    candidate_ids: pl.Tensor[[1, TOPK_MAX], pl.INT32],
    random_key: pl.Scalar[pl.INT32],
):
    """Sample one token from filtered candidates with Gumbel-max."""
    key_zeros = pl.mul(candidate_ids, pl.cast(0, pl.INT32))
    random_key_tile = pl.add(key_zeros, random_key)
    positive_mask = pl.full([1, TOPK_MAX], dtype=pl.INT32, value=0x7FFFFFFF)
    random_bits = pl.xor(candidate_ids, random_key_tile)
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
    gumbel_noise = pl.neg(log_negative_log_uniform)
    sampled_scores = pl.add(candidate_values, gumbel_noise)
    sampled_scores_pad = pl.create_tensor([SAMPLE_BLOCK_ROWS_TILE, TOPK_MAX], dtype=pl.FP32)
    for lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
        sampled_scores_pad[lane : lane + 1, :] = sampled_scores
    winner = pl.read(pl.row_argmax(sampled_scores_pad), [0, 0])
    return pl.read(candidate_ids, [0, pl.cast(winner, pl.INDEX)])


@pl.jit.inline
def apply_top_k(
    logits: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    top_k: pl.Scalar[pl.INT32],
    temperature: pl.Scalar[pl.FP32],
    filtered_values: pl.Tensor[[1, TOPK_MAX], pl.FP32],
    topk_ids: pl.Tensor[[1, TOPK_MAX], pl.INT32],
):
    """Write the exact top-k candidates from one scaled logits row."""
    candidate_values = pl.create_tensor([1, TOPK_CANDIDATE_PAD], dtype=pl.FP32)
    candidate_values_fill = pl.full([1, TOPK_CANDIDATE_PAD], dtype=pl.FP32, value=FP32_NEG_INF)
    candidate_values[:, :] = candidate_values_fill
    candidate_ids = pl.create_tensor([1, TOPK_CANDIDATE_PAD], dtype=pl.INT32)
    candidate_ids[:, :] = pl.full([1, TOPK_CANDIDATE_PAD], dtype=pl.INT32, value=0)

    for group in pl.range(TOPK_NUM_FULL_GROUPS):
        group_pairs = _topk_group_pairs(logits, row, group, temperature)
        group_values = pl.gather(group_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
        group_ids = pl.gather(group_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        candidate_offset = group * TOPK_MAX
        for k in pl.range(TOPK_MAX):
            pl.write(candidate_values, [0, candidate_offset + k], pl.read(group_values, [0, k]))
            pl.write(candidate_ids, [0, candidate_offset + k], pl.read(group_ids, [0, k]))

    tail_start = TOPK_NUM_FULL_GROUPS * TOPK_GROUP_WIDTH
    tail_scores_raw = pl.slice(logits, [1, TOPK_GROUP_TAIL], [row, tail_start])
    tail_scores_pad = pl.full([SAMPLE_BLOCK_ROWS_TILE, SAMPLE_ROW_WIDTH_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
    tail_scores_pad[0:1, 0:TOPK_GROUP_TAIL] = tail_scores_raw
    tail_scaled_scores_pad = apply_temperature(tail_scores_pad, temperature)
    tail_scores = tail_scaled_scores_pad[0:1, 0:TOPK_GROUP_TAIL]
    tail_indices = pl.arange(pl.cast(tail_start, pl.UINT32), [1, TOPK_GROUP_TAIL], dtype=pl.UINT32)
    tail_pairs = pl.sort32(tail_scores, tail_indices)
    tail_pairs = pl.mrgsort(tail_pairs, block_len=64)
    tail_pairs = pl.mrgsort(tail_pairs, block_len=256)
    tail_pairs = tail_pairs[:, 0 : 2 * TOPK_MAX]
    tail_values = pl.gather(tail_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
    tail_ids = pl.gather(tail_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
    tail_offset = TOPK_NUM_FULL_GROUPS * TOPK_MAX
    for k in pl.range(TOPK_MAX):
        pl.write(candidate_values, [0, tail_offset + k], pl.read(tail_values, [0, k]))
        pl.write(candidate_ids, [0, tail_offset + k], pl.read(tail_ids, [0, k]))

    candidate_positions = pl.arange(0, [1, TOPK_CANDIDATE_PAD], dtype=pl.UINT32)
    candidate_pairs = pl.sort32(candidate_values, candidate_positions)
    candidate_pairs = pl.mrgsort(candidate_pairs, block_len=64)
    candidate_pairs = pl.mrgsort(candidate_pairs, block_len=256)
    candidate_pairs = pl.mrgsort(candidate_pairs, block_len=1024)
    half0_pairs = candidate_pairs[:, 0 : 2 * TOPK_MAX]
    half1_start = 2 * TOPK_FINAL_HALF
    half1_pairs = candidate_pairs[:, half1_start : half1_start + 2 * TOPK_MAX]
    topk_pairs = pl.mrgsort(half0_pairs, half1_pairs)[:, 0 : 2 * TOPK_MAX]
    topk_values = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
    selected_positions = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
    for k in pl.range(TOPK_MAX):
        candidate_position = pl.read(selected_positions, [0, k])
        token_id = pl.read(candidate_ids, [0, pl.cast(candidate_position, pl.INDEX)])
        pl.write(topk_ids, [0, k], token_id)

    for k in pl.range(TOPK_MAX):
        if k < top_k:
            pl.write(filtered_values, [0, k], pl.read(topk_values, [0, k]))


@pl.jit.inline
def _sample_unrestricted(
    logits: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    temperature: pl.Scalar[pl.FP32],
    random_flag: pl.Scalar[pl.INT32],
    random_key: pl.Scalar[pl.INT32],
):
    """Sample one row with greedy decoding or unrestricted Gumbel-max."""
    candidate_maxima = pl.full([SAMPLE_BLOCK_ROWS_TILE, SAMPLE_CANDIDATES_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
    candidate_token_ids = pl.full([1, SAMPLE_CANDIDATES_TILE], dtype=pl.INT32, value=0)
    for block in pl.range(SAMPLE_FULL_BLOCKS):
        token_start = block * SAMPLE_BLOCK_ROWS_TILE * SAMPLE_ROW_WIDTH_TILE
        scores_flat = pl.slice(logits, [1, SAMPLE_BLOCK_ROWS_TILE * SAMPLE_ROW_WIDTH_TILE], [row, token_start])
        scores = pl.reshape(scores_flat, [SAMPLE_BLOCK_ROWS_TILE, SAMPLE_ROW_WIDTH_TILE])
        if random_flag > 0:
            scaled_scores = apply_temperature(scores, temperature)
            gumbel_noise = _counter_gumbel(block, random_key)
            scores = pl.add(scaled_scores, gumbel_noise)
        local_winners = pl.row_argmax(scores)
        for lane in pl.range(SAMPLE_BLOCK_ROWS_TILE):
            local_token = pl.read(local_winners, [lane, 0])
            local_score = pl.read(scores, [lane, pl.cast(local_token, pl.INDEX)])
            candidate = block * SAMPLE_BLOCK_ROWS_TILE + lane
            token_base = token_start + lane * SAMPLE_ROW_WIDTH_TILE
            token_id = pl.cast(token_base, pl.INT32) + local_token
            pl.write(candidate_maxima, [0, candidate], local_score)
            pl.write(candidate_token_ids, [0, candidate], token_id)

    tail_start = SAMPLE_FULL_BLOCKS * SAMPLE_BLOCK_ROWS_TILE * SAMPLE_ROW_WIDTH_TILE
    tail_flat = pl.slice(logits, [1, SAMPLE_TAIL_ROWS * SAMPLE_ROW_WIDTH_TILE], [row, tail_start])
    tail_scores = pl.reshape(tail_flat, [SAMPLE_TAIL_ROWS, SAMPLE_ROW_WIDTH_TILE])
    tail_scores_pad = pl.full([SAMPLE_BLOCK_ROWS_TILE, SAMPLE_ROW_WIDTH_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
    tail_scores_pad[0:SAMPLE_TAIL_ROWS, 0:SAMPLE_ROW_WIDTH_TILE] = tail_scores
    tail_block_scores = pl.mul(tail_scores_pad, 1.0)
    if random_flag > 0:
        tail_scaled_scores_pad = apply_temperature(tail_scores_pad, temperature)
        tail_gumbel_noise = _counter_gumbel(SAMPLE_FULL_BLOCKS, random_key)
        tail_block_scores = pl.add(tail_scaled_scores_pad, tail_gumbel_noise)
    for lane in pl.range(SAMPLE_TAIL_ROWS, SAMPLE_BLOCK_ROWS_TILE):
        tail_block_scores[lane : lane + 1, 0:SAMPLE_ROW_WIDTH_TILE] = tail_scores[0:1, 0:SAMPLE_ROW_WIDTH_TILE]
    local_winners = pl.row_argmax(tail_block_scores)
    for lane in pl.range(SAMPLE_TAIL_ROWS):
        local_token = pl.read(local_winners, [lane, 0])
        local_score = pl.read(tail_block_scores, [lane, pl.cast(local_token, pl.INDEX)])
        candidate = SAMPLE_FULL_BLOCKS * SAMPLE_BLOCK_ROWS_TILE + lane
        token_base = tail_start + lane * SAMPLE_ROW_WIDTH_TILE
        token_id = pl.cast(token_base, pl.INT32) + local_token
        pl.write(candidate_maxima, [0, candidate], local_score)
        pl.write(candidate_token_ids, [0, candidate], token_id)

    winning_candidate = pl.read(pl.row_argmax(candidate_maxima), [0, 0])
    return pl.read(candidate_token_ids, [0, pl.cast(winning_candidate, pl.INDEX)])


@pl.jit.inline
def sample(
    logits: pl.Tensor,
    sampling_temperatures: pl.Tensor,
    sampling_top_ks: pl.Tensor,
    sampling_seeds: pl.Tensor,
    sampling_positions: pl.Tensor,
    sampled_ids: pl.Tensor,
):
    """Sample each logits row with greedy or temperature/top-k/Gumbel-max."""
    sample_rows = pl.tensor.dim(logits, 0)
    for row in pl.spmd(sample_rows, name_hint="sample_gumbel_argmax"):
        temperature_base = row // SAMPLE_BLOCK_ROWS_TILE * SAMPLE_BLOCK_ROWS_TILE
        temperature_lane = row % SAMPLE_BLOCK_ROWS_TILE
        temperature_tile = pl.slice(sampling_temperatures, [SAMPLE_BLOCK_ROWS_TILE], [temperature_base])
        safe_temperature_tile = pl.maximum(temperature_tile, SAMPLING_EPS)
        random_ratios = pl.div(temperature_tile, safe_temperature_tile)
        random_flags = pl.cast(random_ratios, pl.INT32, mode="floor")
        temperature = pl.read(safe_temperature_tile, [temperature_lane])
        random_flag = pl.read(random_flags, [temperature_lane])
        top_k = pl.read(sampling_top_ks, [row])
        seed = pl.read(sampling_seeds, [row])
        position = pl.read(sampling_positions, [row])
        seed_index = pl.cast(seed, pl.INDEX)
        position_index = pl.cast(position, pl.INDEX)
        random_key_index = seed_index + position_index * POSITION_MULTIPLIER
        random_key = pl.cast(random_key_index % RANDOM_KEY_MODULUS, pl.INT32)
        if random_flag > 0:
            if top_k > 0 and top_k <= TOPK_MAX:
                candidate_values = pl.create_tensor([1, TOPK_MAX], dtype=pl.FP32)
                candidate_values[:, :] = pl.full([1, TOPK_MAX], dtype=pl.FP32, value=FP32_NEG_INF)
                candidate_ids = pl.create_tensor([1, TOPK_MAX], dtype=pl.INT32)
                candidate_ids[:, :] = pl.full([1, TOPK_MAX], dtype=pl.INT32, value=0)
                apply_top_k(logits, row, top_k, temperature, candidate_values, candidate_ids)
                best_index = gumbel_sample(candidate_values, candidate_ids, random_key)
            else:
                best_index = _sample_unrestricted(logits, row, temperature, random_flag, random_key)
        else:
            best_index = _sample_unrestricted(logits, row, temperature, random_flag, random_key)

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
    top_ks: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    seeds: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    positions: pl.Tensor[[SAMPLE_ROWS], pl.INT32],
    sampled_ids: pl.Out[pl.Tensor[[SAMPLE_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    return sample(logits, temperatures, top_ks, seeds, positions, sampled_ids)


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


def build_tensor_specs(temperature=None, top_k=None):
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
            init_value=(
                (lambda: torch.full([SAMPLE_ROWS], temperature, dtype=torch.float32))
                if temperature is not None
                else (lambda: repeat_rows([0.0, 0.3, 0.7, 1.0, 0.0, 0.5, 1.3, 2.0], torch.float32))
            ),
        ),
        TensorSpec(
            "top_ks",
            [SAMPLE_ROWS],
            torch.int32,
            init_value=(
                (lambda: torch.full([SAMPLE_ROWS], top_k, dtype=torch.int32))
                if top_k is not None
                else (lambda: repeat_rows([VOCAB, 1, 8, 32, VOCAB, 50, 64, VOCAB], torch.int32))
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
        top_k = int(tensors["top_ks"][row])
        if temperature < SAMPLING_EPS:
            selected = torch.argmax(logits)
        else:
            seed = int(tensors["seeds"][row])
            position = int(tensors["positions"][row])
            noise = torch.from_numpy(_gumbel_noise(seed, position))
            scaled_logits = logits / temperature
            if 0 < top_k <= TOPK_MAX:
                topk_indices = torch.topk(scaled_logits, top_k).indices
                filtered_logits = torch.full_like(scaled_logits, -torch.inf)
                filtered_logits[topk_indices] = scaled_logits[topk_indices]
                scaled_logits = filtered_logits
            selected = torch.argmax(scaled_logits + noise)
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
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()

    if args.temperature is not None and args.temperature < 0.0:
        parser.error(f"--temperature must be non-negative, got {args.temperature}")
    if args.top_k is not None:
        valid_top_k = args.top_k <= 0 or args.top_k <= TOPK_MAX or args.top_k >= VOCAB
        if not valid_top_k:
            parser.error(f"--top-k must be <= {TOPK_MAX} or unrestricted, got {args.top_k}")

    result = run_jit(
        fn=sample_test,
        specs=build_tensor_specs(args.temperature, args.top_k),
        golden_fn=golden_sample,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=int(args.enable_l2_swimlane),
        ),
        rtol=0,
        atol=0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
