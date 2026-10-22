# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# What a rank sees when an epoch does not complete: PlanError names the batches and why, the error word of each
# incomplete gate is set so kernels see TriggeredView.failed.


import pytest
import torch
import triton
import triton.language as tl

import iris
from iris.triggered import (
    PlanError,
    State,
    TriggeredPlan,
    TriggeredView,
    allgather,
    chunks,
)

EMPTY = -1


def _ctx(**kwargs):
    try:
        ctx = iris.iris(1 << 26, allocator_type="vmem_chunked", engines=("sdma",), **kwargs)
    except RuntimeError as exc:
        if "update rocm-xio" in str(exc):
            pytest.skip("rocm-xio cannot create several host channels per peer")
        raise
    if ctx.get_num_ranks() < 2:
        pytest.skip("triggered plans need at least two ranks")
    return ctx


def _finish(ctx, plan):
    """Abort what is armed, then wait for every rank's chains to land before anyone frees a gate they write to."""
    ctx.barrier()  # every rank has finished looking at the failed epoch: an abort completes the peers' batches
    try:
        if plan.state in (State.ARMED, State.ACTIVE, State.FAILED):
            plan.abort(timeout=10)
    finally:
        ctx.barrier()
        plan.destroy()


@triton.jit
def _publish_kernel(view, batch):
    TriggeredView.initialize(view).publish(batch)


@triton.jit
def _failed_kernel(view, out):
    batch = tl.program_id(0)
    tl.store(out + batch, TriggeredView.initialize(view).failed(batch).to(tl.int32))


def _timed_out(ctx, plan):
    """end_device_epoch must raise on every rank; returns once all ranks have timed out."""
    with pytest.raises(PlanError) as err:
        plan.end_device_epoch(timeout=1)
    ctx.barrier()
    return err


def _failed(plan):
    out = torch.zeros(plan.num_batches, dtype=torch.int32, device="cuda")
    _failed_kernel[(plan.num_batches,)](plan.view, out)
    torch.cuda.synchronize()
    return out.cpu().tolist()


def test_unpublished_batches_are_reported_and_flagged():
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    buffer = ctx.zeros((world * 2, 16), dtype=torch.int32)
    plan = allgather(ctx, buffer, (1, 16), timeout=20)  # two tiles per rank
    try:
        view = plan.begin_device_epoch()
        _publish_kernel[(1,)](view, plan.batch_id(rank, 0))  # the first tile only
        assert _failed(plan) == [0] * plan.num_batches
        ctx.barrier()
        err = _timed_out(ctx, plan)
        assert plan.state == State.FAILED
        assert isinstance(err.value, TimeoutError)
        late = [plan.batch_id(r, 1) for r in range(world)]
        assert [f.batch for f in plan.failures] == late
        for f in plan.failures:
            assert f.owner == [r for r in range(world) if plan.batch_id(r, 1) == f.batch][0]
            assert f.reason == ("unpublished" if f.owner == rank else "not arrived here")
        assert str(late[0]) in str(err.value)
        # the error word is this epoch exactly for the incomplete gates, and kernels see it
        assert _failed(plan) == [1 if b in late else 0 for b in range(plan.num_batches)]
    finally:
        _finish(ctx, plan)


def test_a_released_batch_that_never_lands_names_the_peers():
    """Own batch released but its chain cannot run (an earlier batch on the queue is unreleased): peers missing."""
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    if world < 2:
        pytest.skip("needs peers")
    buffer = ctx.zeros(world * 64, dtype=torch.int32)
    first, second = chunks(buffer[:32], 2)  # both from rank 0, same lane: second waits behind first
    schedules = [[first, second]] + [[] for _ in range(world - 1)]
    plan = TriggeredPlan(ctx, buffer, schedules, timeout=20)
    try:
        view = plan.begin_device_epoch()
        if rank == 0:
            _publish_kernel[(1,)](view, 1)  # released, but head-of-line blocked behind batch 0
        ctx.barrier()
        _timed_out(ctx, plan)
        by_batch = {f.batch: f for f in plan.failures}
        if rank == 0:
            assert by_batch[0].reason == "unpublished"
            assert by_batch[1].reason == "not arrived at peers"
            assert by_batch[1].peers == tuple(range(1, world))
        else:
            assert by_batch[0].reason == by_batch[1].reason == "not arrived here"
    finally:
        _finish(ctx, plan)


def test_a_failure_is_for_its_epoch_only():
    ctx = _ctx()
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    buffer = ctx.zeros((world, 16), dtype=torch.int32)
    plan = allgather(ctx, buffer, (1, 16), timeout=20)
    try:
        plan.begin_device_epoch()
        _timed_out(ctx, plan)
        assert all(_failed(plan))
    finally:
        _finish(ctx, plan)
    # a new plan starts clean: the error word of the old one does not carry over
    plan = allgather(ctx, buffer, (1, 16), timeout=20)
    try:
        plan.run()  # failed() is read after the epoch begins: the view's epoch is 0 before that
        assert _failed(plan) == [0] * plan.num_batches
    finally:
        _finish(ctx, plan)
