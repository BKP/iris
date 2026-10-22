# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Kernels a triggered plan launches itself: releasing its own batches and checking completion."""

import triton
import triton.language as tl

from iris.mem.triton.context import __translate
from iris.mem.triton.triggered import (
    G_ARRIVAL,
    G_COMPLETION,
    G_READY,
    GATE_WORDS,
    HEADER_WORDS,
    R_OWNER,
    R_PEERS,
    R_ROLE,
    RECORD_WORDS,
    ROLE_CONSUME,
    TriggeredView,
)


STATUS_DONE = tl.constexpr(1)
STATUS_UNPUBLISHED = tl.constexpr(2)
STATUS_NOT_HERE = tl.constexpr(4)


@triton.jit
def release_kernel(view, batches):
    """Open the gate of batch ``batches[pid]``, as its last contributor would."""
    tv = TriggeredView.initialize(view)
    batch = tl.load(batches + tl.program_id(0))
    tl.atomic_xchg(tv.gates + batch * GATE_WORDS + G_READY, tv.epoch, sem="release", scope="sys")


@triton.jit
def _reached(ptr, epoch, budget: tl.constexpr):
    i = 0
    while (tl.atomic_add(ptr, 0, sem="acquire", scope="sys") < epoch) & (i < budget):
        i += 1
    return tl.atomic_add(ptr, 0, sem="acquire", scope="sys") >= epoch


@triton.jit
def progress_kernel(
    view, heap_bases, status, missing, BUDGET: tl.constexpr, WORLD_SIZE: tl.constexpr, OWN_ONLY: tl.constexpr
):
    """
    Mark batch ``pid`` complete once it is done this epoch: for this rank's own batches, published
    and arrived at every destination (each chain ends with that signal, so it has fully run); for the
    others, arrived here if this rank is a destination, and nothing to wait for if it is not.
    ``status[pid]`` is ``STATUS_DONE`` when complete. If ``BUDGET`` ran out it says why instead:
    ``STATUS_UNPUBLISHED`` (own batch never released), ``STATUS_NOT_HERE`` (a batch for this rank has not
    arrived), or neither flag with the destinations that have not received an own batch in ``missing[pid]``
    (bit ``r`` set for rank ``r``).
    """
    batch = tl.program_id(0)
    tv = TriggeredView.initialize(view)
    gate = tv.gates + batch * GATE_WORDS
    record = view + HEADER_WORDS + batch * RECORD_WORDS
    own = tl.load(record + R_OWNER) == tv.rank
    destinations = tl.load(record + R_PEERS)
    consume = (tl.load(record + R_ROLE) & ROLE_CONSUME) != 0
    done = tl.full((), True, tl.int1)
    flags = tl.full((), 0, tl.int32)
    absent = tl.full((), 0, tl.int64)
    if own:
        done = _reached(gate + G_READY, tv.epoch, BUDGET)
        flags = tl.where(done, 0, STATUS_UNPUBLISHED)
        published = done
        for peer in tl.static_range(WORLD_SIZE):
            if published & (peer != tv.rank) & (((destinations >> peer) & 1) != 0):
                landed = _reached(__translate(gate + G_ARRIVAL, tv.rank, peer, heap_bases), tv.epoch, BUDGET)
                done = done & landed
                absent = absent | (tl.where(landed, 0, 1).to(tl.int64) << peer)
    elif not OWN_ONLY:
        if consume:
            done = _reached(gate + G_ARRIVAL, tv.epoch, BUDGET)
            flags = tl.where(done, 0, STATUS_NOT_HERE)
    if done:
        tl.atomic_xchg(gate + G_COMPLETION, tv.epoch, sem="release", scope="sys")
    tl.store(status + batch, flags | done.to(tl.int32))
    tl.store(missing + batch, absent)
