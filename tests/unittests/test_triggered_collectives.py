# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Direct collectives (all-gather, scatter, gather) on the copy engines, with a device kernel flipping the gates:
# every tile is published by a kernel on the rank that owns it and polled by a kernel on each rank that receives it.

import math

import pytest
import torch
import triton
import triton.language as tl

import iris
from iris.triggered import State, TriggeredView, allgather, alltoall, broadcast, gather, scatter

POISON = -1


def _ctx(lanes=1):
    try:
        ctx = iris.iris(1 << 26, allocator_type="vmem_chunked", engines=("sdma",), sdma_lanes=lanes)
    except RuntimeError as exc:
        if "update rocm-xio" in str(exc):
            pytest.skip("rocm-xio cannot create several host channels per peer")
        raise
    if ctx.get_num_ranks() < 2:
        pytest.skip("triggered collectives need at least two ranks")
    return ctx


@triton.jit
def _drive(view, status, BUDGET: tl.constexpr):
    tile = tl.program_id(0)
    tv = TriggeredView.initialize(view)
    seen = tl.full((), 0, tl.int32)
    if tv.can_publish(tile):
        seen |= 1
        tv.publish(tile)
    if tv.can_consume(tile):
        seen |= 2
        spins = tl.full((), 0, tl.int32)
        landed = tv.arrived(tile).to(tl.int32)
        while (landed == 0) & (spins < BUDGET):
            spins += 1
            landed = tv.arrived(tile).to(tl.int32)
        if landed == 0:
            seen |= 4
    tl.store(status + tile, seen)


def _data(epoch, r, rows, cols, dtype):
    """Slice r's contents: the epoch and rank in the hundreds, the element index below (exact in every dtype used)."""
    idx = torch.arange(rows * cols, dtype=torch.int32, device="cuda") % 100
    return ((epoch * 16 + r + 1) * 100 + idx).to(dtype).view(rows, cols)


def _tiles_per_slice(rows, cols, tile):
    return math.ceil(rows / tile[0]) * math.ceil(cols / tile[1])


def _run(ctx, plan, buffer, rows, holds, lands, publishes, consumes, epochs, world=None):
    """Seed, drive from the device, and check: slices in ``lands`` hold the data, the others are untouched."""
    rank = ctx.get_rank()
    world = world or ctx.get_num_ranks()  # the number of slices of ``rows`` rows the buffer holds
    cols, dtype = buffer.shape[1], buffer.dtype
    status = torch.zeros(plan.num_batches, dtype=torch.int32, device="cuda")
    try:
        for epoch in range(1, epochs + 1):
            if epoch > 1:
                plan.next_epoch()
            buffer.fill_(POISON)
            for r in holds:
                buffer[r * rows : (r + 1) * rows] = _data(epoch, r, rows, cols, dtype)
            ctx.barrier()
            status.zero_()
            view = plan.begin_device_epoch()
            _drive[(plan.num_batches,)](view, status, BUDGET=1 << 24)
            torch.cuda.synchronize()
            seen = status.cpu()
            assert int(((seen & 1) != 0).sum()) == publishes, f"rank {rank}: tiles published"
            assert int(((seen & 2) != 0).sum()) == consumes, f"rank {rank}: tiles waited for"
            assert not ((seen & 4) != 0).any(), "a tile exceeded the spin budget"
            plan.end_device_epoch()
            assert plan.state == State.COMPLETE
            for r in range(world):
                got = buffer[r * rows : (r + 1) * rows]
                if r in lands:
                    assert torch.equal(got, _data(epoch, r, rows, cols, dtype)), f"epoch {epoch}: slice {r} wrong"
                else:
                    assert bool((got == POISON).all()), f"epoch {epoch}: slice {r} should be untouched"
    finally:
        if plan.state in (State.ARMED, State.ACTIVE, State.FAILED):
            plan.abort(timeout=10)
        plan.destroy()


# (rows per rank, columns, tile shape): whole slice, aligned tiles, edge tiles clipped in rows, columns, or both
SHAPES = [(16, 64, (16, 64)), (16, 64, (4, 64)), (16, 64, (4, 16)), (15, 70, (4, 32)), (7, 33, (7, 33)), (3, 5, (2, 4))]
DTYPES = [torch.int32, torch.int16, torch.float32]


def _allgather(ctx, rows, cols, tile, dtype, epochs=2, lane=0):
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    buffer = ctx.zeros((world * rows, cols), dtype=dtype)
    plan = allgather(ctx, buffer, tile, lane=lane)
    n = _tiles_per_slice(rows, cols, tile)
    _run(ctx, plan, buffer, rows, [rank], range(world), n, n * (world - 1), epochs)


@pytest.mark.parametrize("rows,cols,tile", SHAPES)
def test_allgather_shapes(rows, cols, tile):
    _allgather(_ctx(), rows, cols, tile, torch.int32)


@pytest.mark.parametrize("dtype", DTYPES)
def test_allgather_dtypes(dtype):
    _allgather(_ctx(), 8, 48, (4, 48), dtype, epochs=3)


def test_allgather_on_the_second_lane():
    _allgather(_ctx(lanes=2), 16, 64, (4, 64), torch.int32, lane=1)


@pytest.mark.parametrize("root", [0, 1, 2, 3])
@pytest.mark.parametrize("rows,cols,tile", [(16, 64, (4, 64)), (15, 70, (4, 32))])
def test_scatter(root, rows, cols, tile):
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    if root >= world:
        pytest.skip("root beyond the ranks")
    buffer = ctx.zeros((world * rows, cols), dtype=torch.int32)
    plan = scatter(ctx, buffer, root, tile)
    n = _tiles_per_slice(rows, cols, tile)
    if rank == root:
        _run(ctx, plan, buffer, rows, range(world), range(world), n * (world - 1), 0, 2)
    else:
        _run(ctx, plan, buffer, rows, [], [rank], 0, n, 2)


@pytest.mark.parametrize("root", [0, 1, 2, 3])
@pytest.mark.parametrize("rows,cols,tile", [(16, 64, (4, 64)), (15, 70, (4, 32))])
def test_gather(root, rows, cols, tile):
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    if root >= world:
        pytest.skip("root beyond the ranks")
    buffer = ctx.zeros((world * rows, cols), dtype=torch.int32)
    plan = gather(ctx, buffer, root, tile)
    n = _tiles_per_slice(rows, cols, tile)
    if rank == root:
        _run(ctx, plan, buffer, rows, [rank], range(world), 0, n * (world - 1), 2)
    else:
        _run(ctx, plan, buffer, rows, [rank], [rank], n, 0, 2)


@pytest.mark.parametrize("root", [0, 1, 3])
def test_broadcast(root):
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    if root >= world:
        pytest.skip("root beyond the ranks")
    rows, cols, tile = 10, 40, (4, 16)  # the buffer is not split by rank; the edge tiles are clipped
    buffer = ctx.zeros((rows, cols), dtype=torch.int32)
    plan = broadcast(ctx, buffer, root, tile)
    n = _tiles_per_slice(rows, cols, tile)
    # one "slice" spanning the whole buffer: slice 0 is the root's data, held at the root, landed everywhere
    # (_run indexes slices by rank, so check with a world of one slice)
    _run(
        ctx,
        plan,
        buffer,
        rows,
        [0] if rank == root else [],
        [0],
        n if rank == root else 0,
        0 if rank == root else n,
        2,
        world=1,
    )


def test_alltoall():
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    rows, cols = 4, 24  # rows per block
    buffer = ctx.zeros((world * world * rows, cols), dtype=torch.int32)
    plan = alltoall(ctx, buffer, (2, cols))
    n = _tiles_per_slice(rows, cols, (2, cols))
    # block (r, j) is slice r * world + j: rank r holds its row of blocks and receives block (r', rank) from every r'
    holds = [rank * world + j for j in range(world)]
    lands = holds + [r * world + rank for r in range(world)]
    _run(ctx, plan, buffer, rows, holds, lands, n * (world - 1), n * (world - 1), 2, world=world * world)


def test_two_collectives_on_one_context_in_arming_order():
    """An all-gather armed first and a scatter armed second share the queues; run and abort them in that order."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    rows, cols = 8, 32
    a = ctx.zeros((world * rows, cols), dtype=torch.int32)
    b = ctx.zeros((world * rows, cols), dtype=torch.int32)
    first = allgather(ctx, a, (4, cols))
    second = scatter(ctx, b, 0, (4, cols))
    n = _tiles_per_slice(rows, cols, (4, cols))
    _run(ctx, first, a, rows, [rank], range(world), n, n * (world - 1), 1)
    if rank == 0:
        _run(ctx, second, b, rows, range(world), range(world), n * (world - 1), 0, 1)
    else:
        _run(ctx, second, b, rows, [], [rank], 0, n, 1)


def test_misuse_raises():
    ctx = _ctx()
    world = ctx.get_num_ranks()
    buffer = ctx.zeros((world * 4, 8), dtype=torch.int32)
    with pytest.raises(ValueError, match="root"):
        scatter(ctx, buffer, world, (2, 8))
    with pytest.raises(ValueError, match="evenly"):
        allgather(ctx, ctx.zeros((world * 4 + 1, 8), dtype=torch.int32), (2, 8))
    with pytest.raises(ValueError, match="lane"):
        allgather(ctx, buffer, (2, 8), lane=5)
