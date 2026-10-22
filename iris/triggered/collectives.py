# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Direct (single-hop) collectives as triggered plans: all-gather, scatter, gather, broadcast and all-to-all.

Each builder takes a symmetric 2D ``buffer`` of ``world_size * rows_per_rank`` rows and a ``tile_shape``, splits
rank ``r``'s slice (rows ``r * rows_per_rank`` to ``(r + 1) * rows_per_rank``) into tiles, and returns a
``TriggeredPlan`` with one batch per tile. Data always lands at the same offset in the receiver's buffer, so
rank ``r``'s slice is where rank ``r`` keeps it and where everyone else expects it.

| collective | who publishes | who receives | data moved |
|---|---|---|---|
| ``allgather`` | every rank, its own slice | every other rank | slice ``r`` to all ranks |
| ``scatter`` | the root, every slice | rank ``r`` | slice ``r`` of the root to rank ``r`` only |
| ``gather`` | every rank except the root | the root | slice ``r`` to the root only |
| ``broadcast`` | the root, the whole buffer | every other rank | the root's buffer to all ranks |
| ``alltoall`` | every rank, its row of blocks | block ``(r, j)`` to rank ``j`` | see below |

``alltoall`` needs ``world_size ** 2`` equal blocks, laid out ``[source][destination]``: rank ``r`` holds block ``(r, j)``
for every ``j`` and sends it to rank ``j``, where it lands in the same place, so afterwards rank ``j``'s column ``j`` of
blocks holds what every rank sent it. Blocks a rank does not receive are not touched. ``broadcast`` needs no slicing.

The plan is the ordinary one: ``begin_device_epoch()`` returns the view, a kernel calls ``TriggeredView.publish``
for the tiles its rank owns and waits for the tiles it receives, ``end_device_epoch()`` checks completion.
``can_publish(tile)`` and ``can_consume(tile)`` say which role a rank has for a tile, and a rank with neither must
neither publish nor wait for it. Batch ids are global: rank ``r``'s tiles come after those of ranks ``0..r-1``
(``plan.batch_id(rank, index)``).
"""

from iris.triggered.plan import TriggeredPlan
from iris.triggered.schedule import tiles


def _tile_shape(tile_shape):
    shape = tuple(tile_shape)
    if len(shape) != 2 or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in shape):
        raise ValueError(f"tile_shape must be two positive integers (rows, columns), got {tile_shape!r}")
    return shape


def _slices(buffer, world_size):
    """The per-rank slices of ``buffer``: views of ``rows_per_rank`` rows each."""
    if buffer.dim() != 2 or buffer.stride(1) != 1 or not buffer.is_contiguous():
        raise ValueError("a collective's buffer must be a contiguous 2D tensor")
    rows, rem = divmod(buffer.shape[0], world_size)
    if rem or rows == 0:
        raise ValueError(f"buffer has {buffer.shape[0]} rows, which {world_size} ranks cannot split evenly")
    return [buffer[r * rows : (r + 1) * rows] for r in range(world_size)]


def _root(root, world_size):
    if not isinstance(root, int) or isinstance(root, bool) or not 0 <= root < world_size:
        raise ValueError(f"root must be a rank in 0..{world_size - 1}, got {root!r}")
    return root


def allgather(ctx, buffer, tile_shape, lane=0, timeout=30.0):
    """Every rank's slice of ``buffer`` ends up in every rank's ``buffer``."""
    shape = _tile_shape(tile_shape)
    slices = _slices(buffer, ctx.get_num_ranks())
    schedules = [tiles(s, shape, base=buffer, lane=lane) for s in slices]
    return TriggeredPlan(ctx, buffer, schedules, timeout=timeout)


def scatter(ctx, buffer, root, tile_shape, lane=0, timeout=30.0):
    """Slice ``r`` of the root's ``buffer`` ends up in rank ``r``'s ``buffer``; other slices are not touched."""
    world = ctx.get_num_ranks()
    root = _root(root, world)
    shape = _tile_shape(tile_shape)
    slices = _slices(buffer, world)
    schedules = [[] for _ in range(world)]
    for r, s in enumerate(slices):
        if r != root:
            schedules[root] += tiles(s, shape, base=buffer, peers=(r,), lane=lane)
    return TriggeredPlan(ctx, buffer, schedules, timeout=timeout)


def gather(ctx, buffer, root, tile_shape, lane=0, timeout=30.0):
    """Every rank's slice of ``buffer`` ends up in the root's ``buffer``; other ranks' buffers are not touched."""
    world = ctx.get_num_ranks()
    root = _root(root, world)
    shape = _tile_shape(tile_shape)
    slices = _slices(buffer, world)
    schedules = [
        [] if r == root else tiles(s, shape, base=buffer, peers=(root,), lane=lane) for r, s in enumerate(slices)
    ]
    return TriggeredPlan(ctx, buffer, schedules, timeout=timeout)


def broadcast(ctx, buffer, root, tile_shape, lane=0, timeout=30.0):
    """The root's whole ``buffer`` ends up in every other rank's ``buffer``."""
    world = ctx.get_num_ranks()
    root = _root(root, world)
    shape = _tile_shape(tile_shape)
    _slices(buffer, 1)  # the layout check only
    schedules = [tiles(buffer, shape, base=buffer, lane=lane) if r == root else [] for r in range(world)]
    return TriggeredPlan(ctx, buffer, schedules, timeout=timeout)


def alltoall(ctx, buffer, tile_shape, lane=0, timeout=30.0):
    """Block ``(r, j)`` of rank ``r``'s ``buffer`` ends up as block ``(r, j)`` of rank ``j``'s ``buffer``."""
    world = ctx.get_num_ranks()
    shape = _tile_shape(tile_shape)
    blocks = _slices(buffer, world * world)
    schedules = [
        [
            b
            for j in range(world)
            if j != r
            for b in tiles(blocks[r * world + j], shape, base=buffer, peers=(j,), lane=lane)
        ]
        for r in range(world)
    ]
    return TriggeredPlan(ctx, buffer, schedules, timeout=timeout)


class Collectives:
    """The collectives declared on one Iris context: ``ctx.triggered.allgather(buffer, tile_shape)``."""

    def __init__(self, ctx):
        self._ctx = ctx

    def allgather(self, buffer, tile_shape, **kwargs):
        return allgather(self._ctx, buffer, tile_shape, **kwargs)

    def scatter(self, buffer, root, tile_shape, **kwargs):
        return scatter(self._ctx, buffer, root, tile_shape, **kwargs)

    def gather(self, buffer, root, tile_shape, **kwargs):
        return gather(self._ctx, buffer, root, tile_shape, **kwargs)

    def broadcast(self, buffer, root, tile_shape, **kwargs):
        return broadcast(self._ctx, buffer, root, tile_shape, **kwargs)

    def alltoall(self, buffer, tile_shape, **kwargs):
        return alltoall(self._ctx, buffer, tile_shape, **kwargs)
