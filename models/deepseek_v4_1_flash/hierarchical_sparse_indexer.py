# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""First-level candidate-block selection for the decoder hierarchical sparse indexer."""

import pypto.language as pl
import torch

from models.deepseek_v4_1_flash.config import FLASH
from models.deepseek_v4_1_flash.config import CMP_POSITIONS_DYN, T_DYN
from models.deepseek_v4_1_flash.golden import select_candidate_blocks


def golden_hierarchical_sparse_indexer(
    index_scores: torch.Tensor,
    compressed_lens: torch.Tensor,
) -> torch.Tensor:
    return select_candidate_blocks(
        index_scores, compressed_lens, FLASH.candidate_topk_blocks, FLASH.candidate_block_size
    )


@pl.jit.inline
def hierarchical_sparse_indexer(
    index_scores: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.FP32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.BOOL],
):
    raise NotImplementedError("hierarchical sparse indexer kernel body is assigned independently")


__all__ = ["golden_hierarchical_sparse_indexer", "hierarchical_sparse_indexer"]


if __name__ == "__main__":
    from models.deepseek_v4_1_flash._golden_smoke import run_hierarchical_indexer_golden

    run_hierarchical_indexer_golden(golden_hierarchical_sparse_indexer)
