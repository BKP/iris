# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# What the direct collectives (all-gather, scatter, gather) ask of a triggered plan: per-batch destinations, the
# roles and masks in the device view, and the schedules the builders produce (host only; no context or GPU).

from types import SimpleNamespace

import pytest

from iris.mem.triton import triggered as dv
from iris.triggered import Batch, Segment, collectives
from iris.triggered.plan import view_words
from iris.triggered.schedule import destinations, tiles, validate

ELEM = 4


class FakeTensor:
    """The few tensor methods the schedule builders use: a 2D row-major tensor of 4-byte elements."""

    def __init__(self, rows, cols, ptr=0x1000, storage=0x1000):
        self.shape, self._ptr, self._storage = (rows, cols), ptr, storage

    def dim(self):
        return 2

    def stride(self, i):
        return (self.shape[1], 1)[i]

    def element_size(self):
        return ELEM

    def data_ptr(self):
        return self._ptr

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self._storage)

    def is_contiguous(self):
        return True

    def __getitem__(self, rows):
        return FakeTensor(
            rows.stop - rows.start, self.shape[1], self._ptr + rows.start * self.shape[1] * ELEM, self._storage
        )


class FakeCtx:
    def __init__(self, world):
        self.world = world

    def get_num_ranks(self):
        return self.world


def build(builder, world, *args, **kwargs):
    """Run a collective builder with the TriggeredPlan replaced by a recorder of the schedules."""
    original = collectives.TriggeredPlan
    collectives.TriggeredPlan = lambda ctx, buffer, schedules, timeout=30.0: SimpleNamespace(schedules=schedules)
    try:
        return builder(FakeCtx(world), *args, **kwargs).schedules
    finally:
        collectives.TriggeredPlan = original


def test_a_batch_goes_to_every_other_rank_by_default():
    batch = Batch((Segment(0, 64),))
    assert batch.peers is None
    assert destinations(batch, owner=1, world_size=4) == (0, 2, 3)
    assert destinations(Batch((Segment(0, 64),), peers=(3, 0)), owner=1, world_size=4) == (0, 3)


def test_validate_accepts_peers_that_are_other_ranks():
    batches = [Batch((Segment(0, 64),), peers=(0, 2))]
    assert validate(batches, 64, owner=1, world_size=3) == batches
    assert validate([Batch((Segment(0, 64),), peers=(0, 2))], 64) is not None  # unchecked without a world


def test_validate_rejects_bad_peers():
    for peers in ((), (1,), (3,), (-1,), (0, 0), (True,), ("0",)):
        with pytest.raises(ValueError, match="peers"):
            validate([Batch((Segment(0, 64),), peers=peers)], 64, owner=1, world_size=3)


def test_view_words_give_each_rank_its_role_and_the_destination_mask():
    # rank 0 sends a batch to rank 2 only
    schedules = [[Batch((Segment(0, 64),), peers=(2,))], [], []]
    for rank, role in ((0, dv.ROLE_PUBLISH.value), (1, 0), (2, dv.ROLE_CONSUME.value)):
        words = view_words(rank, 3, 0x9000, schedules)
        record = dv.HEADER_WORDS.value
        assert words[record + dv.R_ROLE.value] == role
        assert words[record + dv.R_OWNER.value] == 0
        assert words[record + dv.R_PEERS.value] == 0b100


def test_a_default_batch_has_every_other_rank_in_its_mask_and_everyone_a_role():
    schedules = [[], [Batch((Segment(0, 64),))], []]
    words = [view_words(rank, 3, 0x9000, schedules)[dv.HEADER_WORDS.value + dv.R_ROLE.value] for rank in range(3)]
    assert words == [dv.ROLE_CONSUME.value, dv.ROLE_PUBLISH.value, dv.ROLE_CONSUME.value]
    assert view_words(1, 3, 0x9000, schedules)[dv.HEADER_WORDS.value + dv.R_PEERS.value] == 0b101


def test_tiles_makes_one_batch_per_tile_in_row_major_order():
    view = FakeTensor(4, 8)
    batches = tiles(view, (2, 4), peers=(1,), lane=2)
    assert [b.peers for b in batches] == [(1,)] * 4
    assert {b.lane for b in batches} == {2}
    assert [b.segments[0].offset for b in batches] == [0, 16, 64, 80]  # (0,0) (0,4) (2,0) (2,4), 4-byte elements
    assert batches[0].segments[0] == Segment(0, 16, 2, 32)


def test_tiles_clips_the_edge_tiles():
    batches = tiles(FakeTensor(3, 5), (2, 4))
    sizes = [(b.segments[0].height, b.segments[0].width) for b in batches]
    assert sizes == [(2, 16), (2, 4), (1, 16), (1, 4)]


def test_allgather_has_every_rank_send_its_own_slice_to_everyone():
    schedules = build(collectives.allgather, 3, FakeTensor(6, 8), (1, 8))
    assert [len(s) for s in schedules] == [2, 2, 2]
    for rank, schedule in enumerate(schedules):
        assert all(b.peers is None for b in schedule)
        assert [b.segments[0].offset for b in schedule] == [rank * 2 * 32 + i * 32 for i in range(2)]


def test_scatter_has_the_root_send_each_slice_to_that_rank_only():
    schedules = build(collectives.scatter, 3, FakeTensor(6, 8), 1, (2, 8))
    assert schedules[0] == [] and schedules[2] == []
    root = schedules[1]
    assert [b.peers for b in root] == [(0,), (2,)]
    assert [b.segments[0].offset for b in root] == [0, 128]  # rank 0's slice and rank 2's slice, root's own is skipped


def test_gather_has_every_other_rank_send_its_slice_to_the_root_only():
    schedules = build(collectives.gather, 3, FakeTensor(6, 8), 2, (2, 8))
    assert schedules[2] == []
    assert [b.peers for b in schedules[0]] == [(2,)] and [b.peers for b in schedules[1]] == [(2,)]
    assert schedules[0][0].segments[0].offset == 0 and schedules[1][0].segments[0].offset == 64


def test_builders_refuse_what_they_cannot_split():
    with pytest.raises(ValueError, match="evenly"):
        build(collectives.allgather, 3, FakeTensor(7, 8), (1, 8))
    with pytest.raises(ValueError, match="tile_shape"):
        build(collectives.allgather, 2, FakeTensor(4, 8), (0, 8))
    with pytest.raises(ValueError, match="tile_shape"):
        build(collectives.allgather, 2, FakeTensor(4, 8), (1, 2, 3))
    with pytest.raises(ValueError, match="root"):
        build(collectives.scatter, 2, FakeTensor(4, 8), 2, (1, 8))
    with pytest.raises(ValueError, match="root"):
        build(collectives.gather, 2, FakeTensor(4, 8), -1, (1, 8))


def test_broadcast_has_only_the_root_send_its_whole_buffer_to_everyone():
    schedules = build(collectives.broadcast, 3, FakeTensor(5, 8), 1, (2, 8))
    assert schedules[0] == [] and schedules[2] == []
    assert [b.peers for b in schedules[1]] == [None] * 3
    assert [b.segments[0].height for b in schedules[1]] == [2, 2, 1]  # 5 rows need not split across ranks


def test_alltoall_sends_block_r_j_from_rank_r_to_rank_j_only():
    schedules = build(collectives.alltoall, 3, FakeTensor(9, 8), (1, 8))  # nine one-row blocks
    for r, schedule in enumerate(schedules):
        assert [b.peers for b in schedule] == [(j,) for j in range(3) if j != r]
        assert [b.segments[0].offset for b in schedule] == [(r * 3 + j) * 32 for j in range(3) if j != r]


def test_alltoall_needs_world_squared_blocks():
    with pytest.raises(ValueError, match="evenly"):
        build(collectives.alltoall, 3, FakeTensor(6, 8), (1, 8))
