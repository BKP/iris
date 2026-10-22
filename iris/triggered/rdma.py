# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
RDMA destinations of a triggered plan: dormant work in GPU-memory queues, released by an SDMA kick chain.

For each batch a rank owns and each peer on another host, the host posts the batch's RDMA WRITEs into a dedicated send
queue and leaves them **dormant** (the NIC cannot see work whose doorbell was not rung), followed by one more WRITE that
puts the epoch into the peer's arrival gate. It stages the two values that ring the doorbell (the doorbell record, the
control header of the newest work request) in GPU memory and arms an SDMA chain on the local GPU's host queue:

    POLL_REGMEM(ready gate >= epoch) -> copy the record to the send queue's doorbell record -> copy the header to the doorbell

so the producer kernel's ``publish`` opens the gate and the copy engine rings the NIC. A kick releases every work request
posted up to its batch, so kicks must go out in posting order; chains on one SDMA queue run in arming order, and a lane is
one (SDMA host queue, queue pair) pair. The arrival word has exactly one writer per epoch (the batch's owner), so the
epoch is written, not added. The sender learns that a batch is done from the completion of the arrival write, polled on
the host (the NIC reports it once the peer acknowledged the data and the arrival write).

Needs ``engines=("sdma", "rdma")``; queue pairs and their send queues live in GPU memory (rocm-xio ``device_queues``).
"""

import numpy as np
import torch
import torch.distributed as dist

from iris.mem.triton import triggered as dv
from iris.triggered import armer
from iris.triggered.queues import queue_index
from iris.triggered.schedule import destinations

# Work-queue sizes of a triggered queue pair (rocm-xio: sq_depth=256 -> 1024 slots, cq_depth=256 -> 512 entries)
SQ_SLOTS = 1024
CQ_SLOTS = 512


def writes(segment):
    """``(byte offset, byte count)`` of the RDMA WRITEs a segment needs: one if it is contiguous, else one per row."""
    if segment.height == 1 or segment.row_pitch == segment.width:
        return [(segment.offset, segment.width * segment.height)]
    return [(segment.offset + r * segment.row_pitch, segment.width) for r in range(segment.height)]


class RdmaKicks:
    """
    The RDMA side of one ``TriggeredPlan`` on this rank. Collective to build (it connects queue pairs and allocates a
    symmetric word); afterwards ``arm(epoch)`` posts and arms one epoch, ``poll()`` reaps completions.
    """

    def __init__(self, plan, rdma_peers):
        from iris.device import rdma_utils

        self.plan = plan
        ctx = plan.ctx
        me = plan.rank
        self.peers = frozenset(rdma_peers)

        # What this rank sends over RDMA, in posting order, and the queue pairs it needs (also as a receiver:
        # the peer's queue pair must exist and be connected for the data to land)
        self.kicks = []  # (batch id, batch, peer, queue)
        keys = set()
        self.host_batches = []  # batches this rank owns that reach a rank on another host (completion is host-published)
        for owner, schedule in enumerate(plan.schedules):
            for i, batch in enumerate(schedule):
                bid = plan.batch_id(owner, i)
                queue = queue_index(batch.lane)
                receivers = destinations(batch, owner, plan.world_size)
                if owner == me:
                    if any(d in self.peers for d in receivers):
                        self.host_batches.append(bid)
                    for receiver in receivers:
                        if receiver in self.peers:
                            self.kicks.append((bid, batch, receiver, queue))
                            keys.add((receiver, queue))
                elif me in receivers and owner in self.peers:
                    keys.add((owner, queue))
        self.kicks.sort(key=lambda kick: kick[0])  # kicks go out in posting order
        self._check_budget()

        rdma = rdma_utils.bind_rdma_ep()
        base, size = ctx.heap.rdma_region()
        self.endpoints = {}
        mine = {}
        for peer, queue in sorted(keys):
            ep = rdma.Endpoint("mlx5", ctx.rdma_device, ctx.gpu_id, device_queues=True)
            ep.register_buffer(base, size, True)
            self.endpoints[(peer, queue)] = ep
            mine[(peer, queue)] = ep.local_info()
        everyone = [None] * plan.world_size
        dist.all_gather_object(everyone, mine)
        for (peer, queue), ep in self.endpoints.items():
            ep.connect(everyone[peer][(me, queue)])
        self.addresses = {key: ep.host_addresses() for key, ep in self.endpoints.items()}

        # The word the arrival writes read their value from: a WRITE reads its source when it executes, so it holds the
        # epoch (all of an epoch's arrival writes carry the same value; the next epoch starts after this one drained)
        self.epoch_word = ctx.zeros(1, dtype=torch.int64)
        self.stage_rec = torch.zeros(max(1, len(self.kicks)), dtype=torch.int32, device=plan.buffer.device)
        self.stage_val = torch.zeros(max(1, len(self.kicks)), dtype=torch.int64, device=plan.buffer.device)

    def _check_budget(self):
        used = {}
        for _, batch, peer, queue in self.kicks:
            slots, cqes = used.get((peer, queue), (0, 0))
            used[(peer, queue)] = (slots + sum(len(writes(s)) for s in batch.segments) + 1, cqes + 1)
        for (peer, queue), (slots, cqes) in used.items():
            if slots > SQ_SLOTS or cqes > CQ_SLOTS:
                raise ValueError(
                    f"the batches to rank {peer} on lane queue {queue} need {slots} send-queue slots and {cqes} "
                    f"completions per epoch; a triggered queue pair has {SQ_SLOTS} and {CQ_SLOTS}. Use fewer or larger "
                    "batches, or spread them over more lanes"
                )

    def reserve_bytes(self):
        """SDMA ring bytes this rank's kick chains take, keyed like the ledger: ``{(rank, queue): bytes}``."""
        sdma = self.plan._sdma_ep
        per_kick = armer.POLL_REGMEM_BYTES + 2 * sdma.COPY_LINEAR_COMMAND_BYTES
        out = {}
        for _, _, _, queue in self.kicks:
            key = (self.plan.rank, queue)
            out[key] = out.get(key, 0) + per_kick
        return out

    def arm(self, epoch):
        """Post this epoch's work dormant, stage the doorbell values, arm the kick chains."""
        plan = self.plan
        ctx = plan.ctx
        if any(ep.host_outstanding() for ep in self.endpoints.values()):
            raise RuntimeError("the previous epoch's RDMA sends have not completed")
        if not self.kicks:
            return
        self.epoch_word.fill_(epoch)
        n = len(self.kicks)
        recs = np.zeros(n, dtype=np.uint32)
        vals = np.zeros(n, dtype=np.uint64)
        heap = ctx.heap
        base = plan.buffer.data_ptr()
        gate_row = plan.gates.element_size() * dv.GATE_WORDS.value
        for idx, (bid, batch, peer, queue) in enumerate(self.kicks):
            ep = self.endpoints[(peer, queue)]
            for segment in batch.segments:
                for offset, nbytes in writes(segment):
                    ep.host_post_write(
                        heap.translate(base + offset, plan.rank, peer), base + offset, nbytes, signaled=False
                    )
            gate = plan.gates.data_ptr() + bid * gate_row
            arrival = heap.translate(gate + 8 * dv.G_ARRIVAL.value, plan.rank, peer)
            ep.host_post_write(arrival, self.epoch_word.data_ptr(), 8, signaled=True)
            recs[idx], vals[idx] = ep.host_staged()
        # The CPU wrote the work requests into GPU-memory queues: drain them to the device before any chain that
        # will ring the doorbell is armed. The drain is per GPU, so one endpoint does it for all of them.
        next(iter(self.endpoints.values())).host_flush()
        self.stage_rec[:n].copy_(torch.from_numpy(recs.view(np.int32)))
        self.stage_val[:n].copy_(torch.from_numpy(vals.view(np.int64)))
        torch.cuda.synchronize()

        sdma = plan._sdma_ep
        gpu = ctx.gpu_id
        for idx, (bid, batch, peer, queue) in enumerate(self.kicks):
            where = self.addresses[(peer, queue)]
            gate = plan.gates.data_ptr() + bid * gate_row
            sdma.wait_flag_then_put(
                gpu, gpu, queue, gate, epoch, where["dbrec"], self.stage_rec.data_ptr() + 4 * idx, 4, 32
            )
            sdma.put(gpu, gpu, queue, where["gpu_doorbell"], self.stage_val.data_ptr() + 8 * idx, 8)

    def outstanding(self):
        return sum(ep.host_outstanding() for ep in self.endpoints.values())

    def poll(self):
        """Reap completions; True once every send of this epoch has been acknowledged."""
        for ep in self.endpoints.values():
            if ep.host_outstanding():
                ep.host_poll(64)
        return self.outstanding() == 0

    def mark_complete(self, gates, epoch):
        """The sends are acknowledged: publish completion of the batches whose destinations the kernel cannot see."""
        if self.host_batches:
            ids = torch.tensor(self.host_batches, dtype=torch.int64, device=gates.device)
            gates[ids, dv.G_COMPLETION.value] = epoch
