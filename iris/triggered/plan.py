# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Triggered SDMA plans: the host arms copy-engine chains, a kernel releases them.

Lifecycle:

    plan = TriggeredPlan(ctx, buffer, schedules)    # collective: allocate gates, arm epoch 1
    view = plan.begin_device_epoch()
    producer[grid](..., view)                       # TriggeredView.publish(batch)
    plan.end_device_epoch()                         # every batch complete, or TimeoutError
    plan.next_epoch()                               # barrier -> re-arm -> barrier
    plan.destroy()

All of a rank's chains to one peer on one lane share that lane's host-initiated copy-engine queue,
and run in the order they were armed. Plans armed on the same queues must therefore be released in
the order they were armed, and nothing else may use those queues while chains are armed on them.
Batches default to lane 0; ``Batch.lane`` puts a batch on another of the context's ``sdma_lanes``
lanes, each on its own host queue, and lanes do not wait for each other.
"""

import enum
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from iris.host.distributed.helpers import distributed_allgather
from iris.host.memory.allocators.torch_allocator import TorchAllocator
from iris.mem.triton import triggered as dv
from iris.triggered import armer, kernels
from iris.triggered.queues import queue_index
from iris.triggered.rdma import RdmaKicks
from iris.triggered.schedule import destinations, validate

UINT32_MAX = (1 << 32) - 1
_MAX_RANKS = 63
_PROGRESS_BUDGET = 100_000


class State(enum.Enum):
    ARMED = "armed"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"
    ABORTED = "aborted"
    DESTROYED = "destroyed"


def _queue_name(peer, queue):
    return f"queue to rank {peer}" + (f" (host queue {queue})" if queue else "")


@dataclass(frozen=True)
class BatchFailure:
    """
    One batch that did not complete in time, as this rank saw it: ``batch`` (global id), ``owner`` rank, and
    ``reason``, one of ``"unpublished"`` (the owner never released it; owner only), ``"not arrived at peers"``
    (released, but ``peers`` have not received it; owner only), ``"not arrived here"`` (this rank is a
    destination and it has not arrived) or ``"send not acknowledged by the NIC"`` (an RDMA send to ``peers`` has not
    completed).
    """

    batch: int
    owner: int
    reason: str
    peers: tuple = ()


class PlanError(TimeoutError):
    """
    The epoch did not complete in time. ``failures`` lists every incomplete batch this rank is part of (see
    ``BatchFailure``). Each rank reports what it observed: it also stores the epoch in the ``G_ERROR`` word of
    those batches' gates, so a kernel on this rank sees ``TriggeredView.failed(batch)`` become true. Peers learn of
    a failure from their own timeout, not from this rank.
    """

    def __init__(self, epoch, failures):
        self.epoch, self.failures = epoch, failures
        ids = [f.batch for f in failures]
        shown = "; ".join(
            f"batch {f.batch} (rank {f.owner}) {f.reason}" + (f" {list(f.peers)}" if f.peers else "")
            for f in failures[:8]
        )
        super().__init__(f"epoch {epoch}: batches {ids[:16]} not complete: {shown}")


class _QueueLedger:
    """
    What is armed on each of this process's host queues, in arming order.

    A queue is one host queue to one peer, keyed ``(peer, queue)`` with ``queue`` the index rocm-xio
    calls a channel (see ``iris.triggered.queues``). Each queue has its own ring, so its byte budget
    is its own. A lane runs on the queue ``queue_index(lane)``.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = defaultdict(list)  # (peer, queue) -> [(plan, epoch, nbytes)]

    def reserve(self, plan, epoch, nbytes):
        """Reserve ``nbytes``, a dict ``{(peer, queue): bytes}``, or raise if any queue would overflow."""
        for (peer, queue), n in nbytes.items():
            used = sum(e[2] for e in self.entries[(peer, queue)])
            if used + n > self.capacity:
                raise RuntimeError(
                    f"arming {n} bytes on the {_queue_name(peer, queue)} would exceed {self.capacity} armed bytes "
                    f"({used} already armed); complete or destroy other plans first"
                )
        for key, n in nbytes.items():
            self.entries[key].append((plan, epoch, n))

    def release(self, plan):
        for key in self.entries:
            self.entries[key] = [e for e in self.entries[key] if e[0] is not plan]

    def check_order(self, plan, allowed, action):
        for (peer, queue), entries in self.entries.items():
            if not any(e[0] is plan for e in entries):
                continue  # nothing of this plan is queued there: its order does not matter to it
            for other, epoch, _ in entries:
                if other is plan:
                    break
                if other.state not in allowed:
                    raise RuntimeError(
                        f"plan armed earlier on the {_queue_name(peer, queue)} (epoch {epoch}) is {other.state.value}; "
                        f"plans sharing queues must {action} in the order they were armed"
                    )

    def check_idle(self, peer, queue=0):
        """Refuse host copy-engine work on host queue ``queue`` to ``peer`` while triggered chains are armed on it."""
        if self.entries.get((peer, queue)):
            raise RuntimeError(
                f"triggered chains are armed on the copy-engine {_queue_name(peer, queue)}; "
                "complete, abort or destroy those plans before using it"
            )


def ledger(ctx):
    if getattr(ctx, "_triggered_ledger", None) is None:
        # Leave half the ring for wrap padding and work queued before the plan
        ctx._triggered_ledger = _QueueLedger(ctx._require_copy_engine().SDMA_QUEUE_SIZE // 2)
    return ctx._triggered_ledger


def _digest(buffer_bytes, schedules):
    text = repr((buffer_bytes, schedules)).encode()
    return np.frombuffer(hashlib.sha256(text).digest()[:8], dtype=np.int64).copy()


def view_words(rank, world_size, gates_ptr, schedules, host_peers=()):
    """
    The device view (see ``iris.mem.triton.triggered``) as a list of int64 words. ``host_peers`` are ranks reached by
    RDMA: a kernel cannot read their gates, so they are left out of the owner's destination mask and a batch sent to
    them is marked ``ROLE_HOST_COMPLETES`` (the host publishes its completion once the NIC acknowledged the send).
    """
    num_batches = sum(len(s) for s in schedules)
    words = [0] * (dv.HEADER_WORDS.value + num_batches * dv.RECORD_WORDS.value)
    words[dv.H_VERSION.value] = dv.ABI_VERSION
    words[dv.H_NUM_BATCHES.value] = num_batches
    words[dv.H_RANK.value] = rank
    words[dv.H_GATES.value] = gates_ptr
    words[dv.H_WORLD_SIZE.value] = world_size
    rec = dv.HEADER_WORDS.value
    for owner, schedule in enumerate(schedules):
        for batch in schedule:
            peers = destinations(batch, owner, world_size)
            mask_peers = peers
            if owner == rank:
                role = dv.ROLE_PUBLISH.value
                mask_peers = tuple(p for p in peers if p not in host_peers)
                if len(mask_peers) != len(peers):
                    role |= dv.ROLE_HOST_COMPLETES.value
            elif rank in peers:
                role = dv.ROLE_CONSUME.value
            else:
                role = 0  # not this rank's batch: it neither publishes nor waits for it
            words[rec + dv.R_ROLE.value] = role
            words[rec + dv.R_CONTRIBUTORS.value] = batch.contributors
            words[rec + dv.R_OWNER.value] = owner
            words[rec + dv.R_PEERS.value] = sum(1 << p for p in mask_peers)
            rec += dv.RECORD_WORDS.value
    return words


class TriggeredPlan:
    """
    Copies ``buffer`` regions between ranks with pre-armed copy-engine chains.

    Rank ``r`` owns the batches in ``schedules[r]`` and sends each of them to every other rank, at
    the same offsets in the peer's ``buffer``. Batch ids are global: rank ``r``'s batches come after
    those of ranks ``0..r-1`` (see ``batch_id``).

    Collective: every rank passes the same ``schedules`` for the same symmetric ``buffer``.

    Args:
        ctx: Iris instance.
        buffer: Contiguous tensor on the symmetric heap. The heap must keep peer mappings stable while
            chains are armed, which rules out the default torch allocator: use ``allocator_type="vmem_chunked"`` (or ``"vmem"``).
        schedules: One list of ``Batch`` per rank, each in the order that rank completes them. Batches
            on different lanes (``Batch.lane``; the context needs ``sdma_lanes`` above 1) do not wait for
            each other.
        timeout: Seconds ``end_device_epoch()`` waits before raising.
    """

    def __init__(self, ctx, buffer, schedules, timeout=30.0):
        self.ctx = ctx
        self._sdma_ep = ctx._require_copy_engine()
        self.rank = ctx.get_rank()
        self.world_size = ctx.get_num_ranks()
        self.timeout = timeout
        self.state = None

        # Armed chains hold peer addresses; the torch allocator re-maps peer heaps on every allocation
        if isinstance(ctx.heap.allocator, TorchAllocator):
            raise ValueError(
                "triggered plans need stable peer mappings: create the Iris context with allocator_type='vmem_chunked' (or 'vmem')"
            )
        if not ctx.heap.is_symmetric(buffer) or not buffer.is_contiguous():
            raise ValueError("buffer must be a contiguous tensor on the symmetric heap")
        buffer_bytes = buffer.numel() * buffer.element_size()
        if len(schedules) != self.world_size:
            raise ValueError(f"need one schedule per rank ({self.world_size}), got {len(schedules)}")
        if self.world_size > _MAX_RANKS:
            raise ValueError(f"a plan supports at most {_MAX_RANKS} ranks (batch destinations are a 64-bit mask)")
        self.schedules = [
            list(validate(s, buffer_bytes, lanes=ctx.sdma_lanes, owner=r, world_size=self.world_size))
            for r, s in enumerate(schedules)
        ]
        digests = distributed_allgather(_digest(buffer_bytes, self.schedules))
        if not (digests == digests[0]).all():
            raise ValueError("ranks passed different buffers sizes or schedules")

        self._first = np.cumsum([0] + [len(s) for s in self.schedules]).tolist()
        self.num_batches = self._first[-1]
        if self.num_batches == 0:
            raise ValueError("schedules contain no batches")
        # Lanes the plan's batches use; the schedules (hence this) are the same on every rank
        self.lanes = 1 + max(b.lane for s in self.schedules for b in s)

        self.buffer = buffer
        self.gates = ctx.zeros(self.num_batches, dv.GATE_WORDS.value, dtype=torch.int64)
        # FIXME: remove once symmetric allocation is fixed. The vmem_chunked heap frees on each rank's
        # GC, so its free lists can diverge and place the same allocation at different offsets
        heap_base = int(ctx.heap.heap_bases_cpu[self.rank])
        offsets = np.array([buffer.data_ptr() - heap_base, self.gates.data_ptr() - heap_base], dtype=np.int64)
        if not (distributed_allgather(offsets) == offsets).all():
            raise RuntimeError("buffer or gates sit at different heap offsets on different ranks")
        # Peers on other hosts are reached by RDMA (the context's rdma engine), the rest by copy-engine chains
        self._rdma_peers = frozenset(ctx._rdma_endpoints)
        uses_rdma = bool(distributed_allgather(np.array([int(bool(self._rdma_peers))], dtype=np.int64)).any())
        words = view_words(
            self.rank, self.world_size, self.gates.data_ptr(), self.schedules, host_peers=self._rdma_peers
        )
        self.view = torch.tensor(words, dtype=torch.int64, device=buffer.device)
        self._own = torch.arange(
            self._first[self.rank], self._first[self.rank + 1], dtype=torch.int64, device=buffer.device
        )
        self._status = torch.zeros(self.num_batches, dtype=torch.int32, device=buffer.device)
        self._missing = torch.zeros(self.num_batches, dtype=torch.int64, device=buffer.device)
        self.failures = []
        self._ledger = ledger(ctx)
        self._rdma = RdmaKicks(self, self._rdma_peers) if uses_rdma else None
        self.epoch = 0
        self._arm(1)
        ctx.barrier()

    def batch_id(self, rank, index):
        """Global id of batch ``index`` in ``schedules[rank]``."""
        return self._first[rank] + index

    def _arm(self, epoch):
        if epoch > UINT32_MAX:
            raise RuntimeError("epoch exceeds 32 bits; destroy the plan and create a new one")
        own = self.schedules[self.rank]
        dests = [destinations(b, self.rank, self.world_size) for b in own]
        reserve = {}
        for peer in range(self.world_size):
            if peer in self._rdma_peers:
                continue
            to_peer = [b for b, d in zip(own, dests) if peer in d]
            for lane, n in armer.chain_bytes_by_lane(self._sdma_ep, to_peer).items():
                reserve[(peer, queue_index(lane))] = n
        if self._rdma is not None:
            for key, n in self._rdma.reserve_bytes().items():
                reserve[key] = reserve.get(key, 0) + n
        self._ledger.reserve(self, epoch, reserve)
        self.epoch = epoch
        self.state = State.ARMED

        heap = self.ctx.heap
        self._heap_bases = heap.heap_bases_cpu.copy()
        base = self.buffer.data_ptr()
        gate_row = self.gates.element_size() * dv.GATE_WORDS.value
        try:
            for i, batch in enumerate(own):
                gate = self.gates.data_ptr() + self.batch_id(self.rank, i) * gate_row
                for peer in dests[i]:
                    if peer in self._rdma_peers:
                        continue
                    copies = [
                        armer.Copy(
                            base + s.offset,
                            heap.translate(base + s.offset, self.rank, peer),
                            s.width,
                            s.height,
                            s.row_pitch,
                        )
                        for s in batch.segments
                    ]
                    arrival = heap.translate(gate + 8 * dv.G_ARRIVAL.value, self.rank, peer)
                    armer.arm_chain(
                        self._sdma_ep,
                        self.rank,
                        peer,
                        gate,
                        self.epoch,
                        copies,
                        arrival,
                        batch.is_2d,
                        channel=queue_index(batch.lane),
                    )
            if self._rdma is not None:
                self._rdma.arm(self.epoch)
        except Exception:
            # Chains armed so far stay parked: abort() releases them
            self.state = State.FAILED
            raise

    def _require(self, *states):
        if self.state not in states:
            names = " or ".join(s.value for s in states)
            raise RuntimeError(f"plan is {self.state.value if self.state else 'uninitialized'}, needs {names}")

    def begin_device_epoch(self):
        """Start the armed epoch and return the view tensor its kernels take."""
        self._require(State.ARMED)
        if not np.array_equal(self.ctx.heap.heap_bases_cpu, self._heap_bases):
            raise RuntimeError("symmetric heap was re-mapped since the plan was armed")
        self._ledger.check_order(self, (State.ACTIVE,), "begin")
        self.view[dv.H_EPOCH.value] = self.epoch
        self.state = State.ACTIVE
        return self.view

    def _wait(self, own_only, timeout):
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        done = kernels.STATUS_DONE.value
        while True:
            kernels.progress_kernel[(self.num_batches,)](
                self.view,
                self.ctx.get_heap_bases(),
                self._status,
                self._missing,
                BUDGET=_PROGRESS_BUDGET,
                WORLD_SIZE=self.world_size,
                OWN_ONLY=own_only,
            )
            status = self._status.cpu()
            mine = slice(self._first[self.rank], self._first[self.rank + 1])
            pending = ((status[mine] if own_only else status) & done) == 0
            sent = self._rdma.poll() if self._rdma is not None else True
            if not pending.any() and sent:
                if self._rdma is not None:
                    self._rdma.mark_complete(self.gates, self.epoch)
                return
            if time.monotonic() > deadline:
                first = mine.start if own_only else 0
                ids = (pending.nonzero().flatten() + first).tolist()
                if ids:
                    self._fail(ids, status, self._missing.cpu())
                self._fail_unacknowledged()

    def _fail(self, ids, status, missing):
        """Record why each batch in ``ids`` is incomplete, flag it in its gate's error word, and raise."""
        owner_of = np.searchsorted(self._first, ids, side="right") - 1
        failures = []
        for b, owner in zip(ids, owner_of.tolist()):
            flags, absent = int(status[b]), int(missing[b])
            if flags & kernels.STATUS_UNPUBLISHED.value:
                failures.append(BatchFailure(b, owner, "unpublished"))
            elif flags & kernels.STATUS_NOT_HERE.value:
                failures.append(BatchFailure(b, owner, "not arrived here"))
            else:
                peers = tuple(r for r in range(self.world_size) if absent >> r & 1)
                failures.append(BatchFailure(b, owner, "not arrived at peers", peers))
        self.gates[torch.tensor(ids, device=self.gates.device), dv.G_ERROR.value] = self.epoch
        self.failures = failures
        raise PlanError(self.epoch, failures)

    def _fail_unacknowledged(self):
        """Every batch is done on the kernel's side but the NIC has not acknowledged all RDMA sends: flag and raise."""
        ids = list(self._rdma.host_batches)
        failures = [
            BatchFailure(b, self.rank, "send not acknowledged by the NIC", tuple(sorted(self._rdma_peers))) for b in ids
        ]
        if ids:
            self.gates[torch.tensor(ids, device=self.gates.device), dv.G_ERROR.value] = self.epoch
        self.failures = failures
        raise PlanError(self.epoch, failures)

    def end_device_epoch(self, timeout=None):
        """Wait until every batch is complete this epoch: own ones arrived at every peer, others here."""
        self._require(State.ACTIVE)
        try:
            self._wait(own_only=False, timeout=timeout)
        except TimeoutError:
            self.state = State.FAILED
            raise
        self._ledger.release(self)
        self.state = State.COMPLETE

    def next_epoch(self):
        """Collective: re-arm every chain for the next epoch."""
        self._require(State.COMPLETE)
        self.ctx.barrier()
        self._arm(self.epoch + 1)
        self.ctx.barrier()

    def run(self, timeout=None):
        """Run one epoch with no producer kernel: publish every own batch as it stands, then wait."""
        self.begin_device_epoch()
        if len(self._own):
            kernels.release_kernel[(len(self._own),)](self.view, self._own)
        self.end_device_epoch(timeout)

    def abort(self, timeout=None):
        """
        Release this rank's armed chains and wait for them to drain. The plan can only be destroyed
        afterwards. Peers' chains to this rank are theirs to abort. Plans sharing queues must be
        aborted in the order they were armed: later chains wait behind earlier ones.
        """
        if self.state not in (State.ARMED, State.ACTIVE, State.FAILED):
            return
        self._ledger.check_order(self, (State.ACTIVE, State.ABORTED), "abort")
        self.view[dv.H_EPOCH.value] = self.epoch
        if len(self._own):
            kernels.release_kernel[(len(self._own),)](self.view, self._own)
        self.state = State.ABORTED
        self._wait(own_only=True, timeout=timeout)
        self._ledger.release(self)

    def destroy(self):
        """Forget the plan. Its chains must have drained: abort() an armed or failed plan first."""
        if self.state in (State.ARMED, State.ACTIVE, State.FAILED):
            raise RuntimeError(f"plan is {self.state.value}; abort() it before destroying")
        self._ledger.release(self)
        self.state = State.DESTROYED
        self._rdma = None
        self.gates = self.view = self.buffer = None
