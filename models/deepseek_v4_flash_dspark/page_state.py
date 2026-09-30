# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generation- and version-checked compact page updates shared by both phases."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import EP

UPDATES = pl.dynamic("DSPARK_PAGE_UPDATES_DYN")
SOURCE = pl.dynamic("DSPARK_PAGE_SOURCE_DYN")
LEASES = pl.dynamic("DSPARK_PAGE_LEASES_DYN")
CAPACITY = pl.dynamic("DSPARK_PAGE_CAPACITY_DYN")
DELTA_WIDTH = 9
PAGE_META_WIDTH = 4
STATUS_WIDTH = 16
ROWS = pl.dynamic("DSPARK_LEASE_TABLE_ROWS_DYN")
DEPTH = pl.dynamic("DSPARK_LEASE_TABLE_DEPTH_DYN")


@pl.jit.inline
def _materialize_pages_rank(
    selectors: pl.Tensor[[ROWS, 3], pl.INT32],
    request_meta: pl.Tensor[[LEASES, 6], pl.INT32],
    pages: pl.Tensor[[LEASES, CAPACITY], pl.INT32],
    page_meta: pl.Tensor[[LEASES, PAGE_META_WIDTH], pl.INT32],
    tables: pl.Tensor[[ROWS, DEPTH], pl.INT32],
):
    """Resolve lease/generation/version selectors without Host page retransmission."""
    selectors.bind_dynamic(0, ROWS)
    request_meta.bind_dynamic(0, LEASES)
    pages.bind_dynamic(0, LEASES)
    pages.bind_dynamic(1, CAPACITY)
    page_meta.bind_dynamic(0, LEASES)
    tables.bind_dynamic(0, ROWS)
    tables.bind_dynamic(1, DEPTH)
    rows = pl.tensor.dim(tables, 0)
    depth = pl.tensor.dim(tables, 1)
    leases = pl.tensor.dim(pages, 0)
    capacity = pl.tensor.dim(pages, 1)
    for core in pl.spmd(48, name_hint="dspark_materialize_lease_tables"):
        for line in pl.range(core, (rows * depth + 15) // 16, 48):
            for lane in pl.range(16):
                index = line * 16 + lane
                if index < rows * depth:
                    row = index // depth
                    column = index % depth
                    slot = pl.read(selectors, [row, 0])
                    generation = pl.read(selectors, [row, 1])
                    version = pl.read(selectors, [row, 2])
                    page = pl.cast(-1, pl.INT32)
                    if slot >= 0 and slot < leases and generation > 0 and version > 0:
                        granted = pl.read(request_meta, [slot, 1])
                        resident_generation = pl.read(page_meta, [slot, 0])
                        resident_version = pl.read(page_meta, [slot, 1])
                        count = pl.read(page_meta, [slot, 2])
                        ring = pl.read(page_meta, [slot, 3])
                        if granted == generation and resident_generation == generation and resident_version == version:
                            if count > 0 and count <= capacity:
                                if ring == 1:
                                    page = pl.read(pages, [slot, column % count])
                                elif ring == 0 and column < count:
                                    page = pl.read(pages, [slot, column])
                    pl.write(tables, [row, column], page)
    return tables



@pl.jit
def materialize_pages_rank(
    selectors: pl.Tensor[[ROWS, 3], pl.INT32],
    request_meta: pl.Tensor[[LEASES, 6], pl.INT32],
    pages: pl.Tensor[[LEASES, CAPACITY], pl.INT32],
    page_meta: pl.Tensor[[LEASES, PAGE_META_WIDTH], pl.INT32],
    tables: pl.Out[pl.Tensor[[ROWS, DEPTH], pl.INT32]],
):
    return _materialize_pages_rank(selectors, request_meta, pages, page_meta, tables)

@pl.jit.host
def l3_materialize_pages(
    selectors: pl.Tensor[[EP, ROWS, 3], pl.INT32],
    request_meta: pl.Tensor[[EP, LEASES, 6], pl.INT32],
    pages: pl.Tensor[[EP, LEASES, CAPACITY], pl.INT32],
    page_meta: pl.Tensor[[EP, LEASES, PAGE_META_WIDTH], pl.INT32],
    tables: pl.Out[pl.Tensor[[EP, ROWS, DEPTH], pl.INT32]],
):
    selectors.bind_dynamic(1, ROWS)
    request_meta.bind_dynamic(1, LEASES)
    pages.bind_dynamic(1, LEASES)
    pages.bind_dynamic(2, CAPACITY)
    page_meta.bind_dynamic(1, LEASES)
    tables.bind_dynamic(1, ROWS)
    tables.bind_dynamic(2, DEPTH)
    for rank in pl.range(pld.world_size()):
        materialize_pages_rank(
            selectors[rank], request_meta[rank], pages[rank], page_meta[rank], tables[rank], device=rank,
        )


@pl.jit.inline
def _update_pages_rank(
    deltas: pl.Tensor[[UPDATES, DELTA_WIDTH], pl.INT32],
    source: pl.Tensor[[SOURCE], pl.INT32],
    request_meta: pl.Tensor[[LEASES, 6], pl.INT32],
    pages: pl.Tensor[[LEASES, CAPACITY], pl.INT32],
    page_meta: pl.Tensor[[LEASES, PAGE_META_WIDTH], pl.INT32],
    status: pl.Tensor[[UPDATES, STATUS_WIDTH], pl.INT32],
):
    deltas.bind_dynamic(0, UPDATES)
    source.bind_dynamic(0, SOURCE)
    request_meta.bind_dynamic(0, LEASES)
    pages.bind_dynamic(0, LEASES)
    pages.bind_dynamic(1, CAPACITY)
    page_meta.bind_dynamic(0, LEASES)
    status.bind_dynamic(0, UPDATES)
    updates = pl.tensor.dim(deltas, 0)
    source_length = pl.tensor.dim(source, 0)
    leases = pl.tensor.dim(pages, 0)
    capacity = pl.tensor.dim(pages, 1)
    # Sequential ownership also orders several deltas for the same row.
    for _core in pl.spmd(1, name_hint="dspark_update_lease_pages"):
        for update in pl.range(updates):
            for column in pl.range(STATUS_WIDTH):
                pl.write(status, [update, column], pl.cast(0, pl.INT32))
            slot = pl.read(deltas, [update, 0])
            generation = pl.read(deltas, [update, 1])
            expected = pl.read(deltas, [update, 2])
            version = pl.read(deltas, [update, 3])
            offset = pl.read(deltas, [update, 4])
            count = pl.read(deltas, [update, 5])
            length = pl.read(deltas, [update, 6])
            ring = pl.read(deltas, [update, 7])
            source_offset = pl.read(deltas, [update, 8])
            code = pl.cast(-1, pl.INT32)
            if slot >= 0 and slot < leases and generation > 0 and expected >= 0:
                if offset >= 0 and count >= 0 and length >= 0 and length <= capacity:
                    if offset <= length and count <= length - offset and source_offset >= 0 and source_offset <= source_length:
                        if count <= source_length - source_offset and expected < 2147483647 and version == expected + 1 and (ring == 0 or (ring == 1 and length > 0)):
                            code = pl.cast(-2, pl.INT32)
                            granted = pl.read(request_meta, [slot, 1])
                            if granted == generation:
                                old_generation = pl.read(page_meta, [slot, 0])
                                old_version = pl.read(page_meta, [slot, 1])
                                old_length = pl.read(page_meta, [slot, 2])
                                if old_generation != generation:
                                    old_version = pl.cast(0, pl.INT32)
                                    old_length = pl.cast(0, pl.INT32)
                                code = pl.cast(-3, pl.INT32)
                                if old_version == expected:
                                    code = pl.cast(-4, pl.INT32)
                                    if offset <= old_length and (length <= old_length or offset + count == length):
                                        code = pl.cast(1, pl.INT32)
                                        for index in pl.range(count):
                                            page = pl.read(source, [source_offset + index])
                                            if page < 0:
                                                code = pl.cast(-5, pl.INT32)
                                        if code == 1:
                                            for index in pl.range(count):
                                                pl.write(pages, [slot, offset + index], pl.read(source, [source_offset + index]))
                                            pl.write(page_meta, [slot, 0], generation)
                                            pl.write(page_meta, [slot, 1], version)
                                            pl.write(page_meta, [slot, 2], length)
                                            pl.write(page_meta, [slot, 3], ring)
            pl.write(status, [update, 0], code)
    return pages, page_meta, status



@pl.jit
def update_pages_rank(
    deltas: pl.Tensor[[UPDATES, DELTA_WIDTH], pl.INT32],
    source: pl.Tensor[[SOURCE], pl.INT32],
    request_meta: pl.Tensor[[LEASES, 6], pl.INT32],
    pages: pl.InOut[pl.Tensor[[LEASES, CAPACITY], pl.INT32]],
    page_meta: pl.InOut[pl.Tensor[[LEASES, PAGE_META_WIDTH], pl.INT32]],
    status: pl.Out[pl.Tensor[[UPDATES, STATUS_WIDTH], pl.INT32]],
):
    return _update_pages_rank(deltas, source, request_meta, pages, page_meta, status)

@pl.jit.host
def l3_update_pages(
    deltas: pl.Tensor[[EP, UPDATES, DELTA_WIDTH], pl.INT32],
    source: pl.Tensor[[EP, SOURCE], pl.INT32],
    request_meta: pl.Tensor[[EP, LEASES, 6], pl.INT32],
    pages: pl.InOut[pl.Tensor[[EP, LEASES, CAPACITY], pl.INT32]],
    page_meta: pl.InOut[pl.Tensor[[EP, LEASES, PAGE_META_WIDTH], pl.INT32]],
    status: pl.Out[pl.Tensor[[EP, UPDATES, STATUS_WIDTH], pl.INT32]],
):
    deltas.bind_dynamic(1, UPDATES)
    source.bind_dynamic(1, SOURCE)
    request_meta.bind_dynamic(1, LEASES)
    pages.bind_dynamic(1, LEASES)
    pages.bind_dynamic(2, CAPACITY)
    page_meta.bind_dynamic(1, LEASES)
    status.bind_dynamic(1, UPDATES)
    for rank in pl.range(pld.world_size()):
        update_pages_rank(
            deltas[rank], source[rank], request_meta[rank], pages[rank], page_meta[rank], status[rank], device=rank,
        )


@pl.jit
def update_and_materialize_rank(
    deltas: pl.Tensor[[UPDATES, DELTA_WIDTH], pl.INT32],
    source: pl.Tensor[[SOURCE], pl.INT32],
    selectors: pl.Tensor[[ROWS, 3], pl.INT32],
    request_meta: pl.Tensor[[LEASES, 6], pl.INT32],
    pages: pl.InOut[pl.Tensor[[LEASES, CAPACITY], pl.INT32]],
    page_meta: pl.InOut[pl.Tensor[[LEASES, PAGE_META_WIDTH], pl.INT32]],
    tables: pl.Out[pl.Tensor[[ROWS, DEPTH], pl.INT32]],
    status: pl.Out[pl.Tensor[[UPDATES, STATUS_WIDTH], pl.INT32]],
):
    updated_pages, updated_meta, updated_status = _update_pages_rank(
        deltas, source, request_meta, pages, page_meta, status
    )
    generated_tables = _materialize_pages_rank(selectors, request_meta, updated_pages, updated_meta, tables)
    return updated_pages, updated_meta, generated_tables, updated_status


@pl.jit.host
def l3_update_and_materialize(
    deltas: pl.Tensor[[EP, UPDATES, DELTA_WIDTH], pl.INT32],
    source: pl.Tensor[[EP, SOURCE], pl.INT32],
    selectors: pl.Tensor[[EP, ROWS, 3], pl.INT32],
    request_meta: pl.Tensor[[EP, LEASES, 6], pl.INT32],
    pages: pl.InOut[pl.Tensor[[EP, LEASES, CAPACITY], pl.INT32]],
    page_meta: pl.InOut[pl.Tensor[[EP, LEASES, PAGE_META_WIDTH], pl.INT32]],
    tables: pl.Out[pl.Tensor[[EP, ROWS, DEPTH], pl.INT32]],
    status: pl.Out[pl.Tensor[[EP, UPDATES, STATUS_WIDTH], pl.INT32]],
):
    deltas.bind_dynamic(1, UPDATES)
    source.bind_dynamic(1, SOURCE)
    selectors.bind_dynamic(1, ROWS)
    request_meta.bind_dynamic(1, LEASES)
    pages.bind_dynamic(1, LEASES)
    pages.bind_dynamic(2, CAPACITY)
    page_meta.bind_dynamic(1, LEASES)
    tables.bind_dynamic(1, ROWS)
    tables.bind_dynamic(2, DEPTH)
    status.bind_dynamic(1, UPDATES)
    for rank in pl.range(pld.world_size()):
        update_and_materialize_rank(
            deltas[rank], source[rank], selectors[rank], request_meta[rank], pages[rank], page_meta[rank],
            tables[rank], status[rank], device=rank,
        )
