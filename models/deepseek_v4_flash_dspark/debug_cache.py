# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Diagnostic zeroing of cache storage; never enabled on the normal serving path."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import EP

WORDS = pl.dynamic("DSPARK_DIAGNOSTIC_WORDS")
TILE = 1024


@pl.jit
def zero_storage_rank(storage: pl.Out[pl.Tensor[[1, WORDS], pl.FP32]]):
    storage.bind_dynamic(1, WORDS)
    words = pl.tensor.dim(storage, 1)
    for core in pl.spmd(48, name_hint="dspark_diagnostic_zero_cache"):
        for offset in pl.range(core * TILE, words, 48 * TILE):
            zeros = pl.full([1, TILE], dtype=pl.FP32, value=0.0)
            valid = pl.set_validshape(zeros, 1, pl.min(TILE, words - offset))
            storage[0:1, offset:offset + TILE] = valid
    return storage


@pl.jit.host
def l3_zero_storage(storage: pl.Out[pl.Tensor[[EP, 1, WORDS], pl.FP32]]):
    storage.bind_dynamic(2, WORDS)
    for rank in pl.range(pld.world_size()):
        zero_storage_rank(storage[rank], device=rank)
