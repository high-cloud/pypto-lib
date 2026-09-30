# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Expand compact scheduler page lists into device-owned attention tables."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import EP

ROWS_DYN = pl.dynamic("DSPARK_PAGE_TABLE_ROWS_DYN")
PAGES_DYN = pl.dynamic("DSPARK_PAGE_TABLE_COMPACT_DYN")
DEPTH_DYN = pl.dynamic("DSPARK_PAGE_TABLE_DEPTH_DYN")


@pl.jit
def expand_page_tables_rank(
    pages: pl.Tensor[[ROWS_DYN, PAGES_DYN], pl.INT32],
    row_config: pl.Tensor[[ROWS_DYN, 2], pl.INT32],
    tables: pl.Out[pl.Tensor[[ROWS_DYN, DEPTH_DYN], pl.INT32]],
):
    pages.bind_dynamic(0, ROWS_DYN)
    pages.bind_dynamic(1, PAGES_DYN)
    row_config.bind_dynamic(0, ROWS_DYN)
    tables.bind_dynamic(0, ROWS_DYN)
    tables.bind_dynamic(1, DEPTH_DYN)
    rows = pl.tensor.dim(tables, 0)
    depth = pl.tensor.dim(tables, 1)
    capacity = pl.tensor.dim(pages, 1)
    # Each block owns whole 64-byte lines, even across unaligned request rows.
    for core in pl.spmd(48, name_hint="dspark_expand_page_tables"):
        for line in pl.range(core, (rows * depth + 15) // 16, 48):
            for lane in pl.range(16):
                index = line * 16 + lane
                if index < rows * depth:
                    row = index // depth
                    logical = index % depth
                    count = pl.read(row_config, [row, 0])
                    ring = pl.read(row_config, [row, 1])
                    page = pl.cast(-1, pl.INT32)
                    if count > 0 and count <= capacity:
                        if ring == 1:
                            page = pl.read(pages, [row, logical % count])
                        elif logical < count:
                            page = pl.read(pages, [row, logical])
                    pl.write(tables, [row, logical], page)
    return tables


@pl.jit.host
def l3_expand_page_tables(
    pages: pl.Tensor[[EP, ROWS_DYN, PAGES_DYN], pl.INT32],
    row_config: pl.Tensor[[EP, ROWS_DYN, 2], pl.INT32],
    tables: pl.Out[pl.Tensor[[EP, ROWS_DYN, DEPTH_DYN], pl.INT32]],
):
    pages.bind_dynamic(1, ROWS_DYN)
    pages.bind_dynamic(2, PAGES_DYN)
    row_config.bind_dynamic(1, ROWS_DYN)
    tables.bind_dynamic(1, ROWS_DYN)
    tables.bind_dynamic(2, DEPTH_DYN)
    for rank in pl.range(pld.world_size()):
        expand_page_tables_rank(pages[rank], row_config[rank], tables[rank], device=rank)
