# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Lanes of triggered plans: Batch.lane validation, per-lane armed bytes, the lane -> host queue map and the
# per-(peer, queue) ledger (host only; no context, copy engine or GPU).

from types import SimpleNamespace

import pytest

from iris.triggered import Batch, Segment
from iris.triggered.armer import POLL_REGMEM_BYTES, chain_bytes_by_lane
from iris.triggered.plan import State, _QueueLedger
from iris.triggered.queues import queue_index
from iris.triggered.schedule import validate

SEGMENT = Segment(0, 64)


class FakeSdmaEp:
    COPY_LINEAR_COMMAND_BYTES = 32
    COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES = 48
    ATOMIC_COMMAND_BYTES = 36


def plan(state=State.ARMED):
    return SimpleNamespace(state=state)


def test_batch_defaults_to_lane_zero():
    assert Batch((SEGMENT,)).lane == 0
    batch = Batch((SEGMENT,), 3)
    assert (batch.contributors, batch.lane) == (3, 0)
    assert Batch((SEGMENT,), 3, 1).lane == 1


def test_validate_accepts_lanes_the_context_has():
    batches = [Batch((SEGMENT,), lane=0), Batch((SEGMENT,), lane=1)]
    assert validate(batches, 64, lanes=2) == batches
    assert validate([Batch((SEGMENT,))], 64) == [Batch((SEGMENT,))]  # one lane by default


def test_validate_rejects_a_lane_the_context_does_not_have():
    with pytest.raises(ValueError, match="lane 1 .*1 lane"):
        validate([Batch((SEGMENT,), lane=1)], 64)
    with pytest.raises(ValueError, match="lane 2"):
        validate([Batch((SEGMENT,), lane=2)], 64, lanes=2)
    with pytest.raises(ValueError, match="lane -1"):
        validate([Batch((SEGMENT,), lane=-1)], 64, lanes=2)


def test_validate_rejects_a_lane_that_is_not_an_integer():
    for lane in (True, 1.0, "1", None):
        with pytest.raises(ValueError, match="lane"):
            validate([Batch((SEGMENT,), lane=lane)], 64, lanes=2)


def test_chain_bytes_by_lane_sums_each_lane_separately():
    ep = FakeSdmaEp()
    batches = [
        Batch((Segment(0, 64), Segment(64, 64)), lane=0),
        Batch((Segment(128, 64, height=4, pitch=128),), lane=1),
        Batch((Segment(256, 64),), lane=0),
    ]
    first = POLL_REGMEM_BYTES + 2 * 32 + 36
    second = POLL_REGMEM_BYTES + 48 + 36
    third = POLL_REGMEM_BYTES + 32 + 36
    assert chain_bytes_by_lane(ep, batches) == {0: first + third, 1: second}
    assert chain_bytes_by_lane(ep, []) == {}


def test_ledger_budget_is_per_peer_and_queue():
    ledger = _QueueLedger(100)
    ledger.reserve(plan(), 1, {(1, 0): 90})
    ledger.reserve(plan(), 1, {(1, 1): 90, (2, 0): 90})  # another queue, another peer: own budget
    with pytest.raises(RuntimeError, match=r"arming 20 bytes on the queue to rank 1 would exceed 100"):
        ledger.reserve(plan(), 1, {(1, 0): 20})
    with pytest.raises(RuntimeError, match=r"arming 20 bytes on the queue to rank 1 \(host queue 1\) would exceed"):
        ledger.reserve(plan(), 1, {(1, 1): 20})


def test_a_refused_reservation_reserves_nothing():
    ledger = _QueueLedger(100)
    ledger.reserve(plan(), 1, {(1, 0): 90})
    with pytest.raises(RuntimeError, match="would exceed"):
        ledger.reserve(plan(), 1, {(2, 0): 10, (1, 0): 20})
    assert not ledger.entries[(2, 0)]


def test_plans_on_different_queues_do_not_wait_for_each_other():
    ledger = _QueueLedger(1000)
    first, second = plan(), plan()
    ledger.reserve(first, 1, {(1, 0): 10})
    ledger.reserve(second, 1, {(1, 1): 10})
    ledger.check_order(second, (State.ACTIVE,), "begin")  # first sits on queue 0 only: not ahead of second


def test_plans_on_one_queue_keep_arming_order():
    ledger = _QueueLedger(1000)
    first, second = plan(), plan()
    ledger.reserve(first, 1, {(1, 1): 10})
    ledger.reserve(second, 1, {(1, 1): 10})
    with pytest.raises(RuntimeError, match=r"armed earlier on the queue to rank 1 \(host queue 1\) .* is armed"):
        ledger.check_order(second, (State.ACTIVE,), "begin")
    with pytest.raises(RuntimeError, match="abort in the order"):
        ledger.check_order(second, (State.ACTIVE, State.ABORTED), "abort")
    first.state = State.ACTIVE
    ledger.check_order(second, (State.ACTIVE,), "begin")


def test_host_work_is_refused_only_on_a_queue_with_armed_chains():
    ledger = _QueueLedger(1000)
    armed = plan()
    ledger.reserve(armed, 1, {(1, 1): 10})
    ledger.check_idle(1)  # queue 0, the default
    ledger.check_idle(2, 1)  # another peer
    with pytest.raises(RuntimeError, match=r"armed on the copy-engine queue to rank 1 \(host queue 1\)"):
        ledger.check_idle(1, 1)
    ledger.release(armed)
    ledger.check_idle(1, 1)


def test_the_one_queue_messages_do_not_change():
    ledger = _QueueLedger(10)
    armed = plan()
    ledger.reserve(armed, 1, {(1, 0): 10})
    with pytest.raises(RuntimeError) as busy:
        ledger.check_idle(1)
    assert str(busy.value) == (
        "triggered chains are armed on the copy-engine queue to rank 1; "
        "complete, abort or destroy those plans before using it"
    )
    with pytest.raises(RuntimeError) as full:
        ledger.reserve(plan(), 1, {(1, 0): 1})
    assert str(full.value) == (
        "arming 1 bytes on the queue to rank 1 would exceed 10 armed bytes (10 already armed); "
        "complete or destroy other plans first"
    )


def test_release_frees_every_queue():
    ledger = _QueueLedger(1000)
    armed = plan()
    ledger.reserve(armed, 1, {(1, 0): 10, (1, 1): 10, (2, 1): 10})
    ledger.release(armed)
    for key in ((1, 0), (1, 1), (2, 1)):
        ledger.check_idle(*key)


def test_a_lane_runs_on_the_queue_of_the_same_index():
    assert [queue_index(lane) for lane in range(4)] == [0, 1, 2, 3]
