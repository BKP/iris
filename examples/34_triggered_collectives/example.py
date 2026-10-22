#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Triggered collectives driven from a device kernel: a kernel flips the gates.

Declare a collective, open an epoch, and launch a kernel that does what ``plan.run()`` would do for you:

    plan = ctx.triggered.allgather(buffer, tile_shape)    # arms one copy-engine chain per (tile, receiver)
    view = plan.begin_device_epoch()                       # the device descriptor
    drive[(plan.num_batches,)](view, status, ...)          # your kernel: publish the tiles you own, poll the ones you receive
    plan.end_device_epoch()                                # every tile arrived, or TimeoutError

``publish(tile)`` is one release store to the tile's ready word: that is the whole trigger, the armed chains have been
waiting on it since the plan was declared. ``arrived(tile)`` is an acquire load of the tile's arrival word. A kernel
that spins on ``arrived`` with no budget hangs the GPU, so this one counts its polls.

Roles are per tile and per rank. The kernel does not assume them: it asks ``can_publish`` and ``can_consume`` and writes
back what it saw, which is what the ``roles:`` lines report. In an all-gather each rank publishes its own tiles and
consumes everyone else's; in a scatter the root publishes every tile and each other rank consumes only its own; in a
gather everyone but the root publishes and only the root consumes.

Run it (2 or more GPUs, one rank per GPU):
    torchrun --nproc_per_node=4 --standalone example.py --collective allgather
    torchrun --nproc_per_node=4 --standalone example.py --collective scatter --root 1
    torchrun --nproc_per_node=4 --standalone example.py --collective gather --root 2
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import iris
from iris.triggered import TriggeredView

SEEN_PUBLISH, SEEN_CONSUME, SEEN_TIMEOUT = 1, 2, 4
POISON = -1
SPIN_BUDGET = 1 << 24  # polls a tile may spin before giving up


@triton.jit
def drive(view, status, BUDGET: tl.constexpr):
    """One program per tile: publish it if this rank owns it, poll for it if this rank receives it."""
    tile = tl.program_id(0)
    tv = TriggeredView.initialize(view)
    seen = tl.full((), 0, tl.int32)
    if tv.can_publish(tile):
        seen |= 1
        tv.publish(tile)  # the trigger
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


def payload(rows, cols, tag, device):
    """Slice contents: the tag in the high half, the element index in the low half."""
    return (torch.arange(rows * cols, dtype=torch.int32, device=device) & 0xFFFF | (tag << 16)).view(rows, cols)


def expected_slices(collective, root, rank, world, rows, cols, epoch, device):
    """For each slice index, what this rank's buffer must hold afterwards: a tensor, or None for 'untouched poison'."""
    out = []
    for r in range(world):
        data = payload(rows, cols, epoch * 16 + r + 1, device)
        if collective == "allgather":
            out.append(data)
        else:  # scatter or gather: the root sees every slice, the others only their own
            out.append(data if (rank == root or r == rank) else None)
    return out


def seed(buffer, collective, root, rank, world, rows, cols, epoch):
    """Put the input where each rank keeps it: own slice (all-gather, gather), every slice at the root (scatter)."""
    buffer.fill_(POISON)
    for r in range(world):
        mine = r == rank if collective != "scatter" else rank == root
        if mine:
            buffer[r * rows : (r + 1) * rows] = payload(rows, cols, epoch * 16 + r + 1, buffer.device)


def verify(buffer, want, rows):
    for r, expected in enumerate(want):
        got = buffer[r * rows : (r + 1) * rows]
        if expected is None:
            if not bool((got == POISON).all()):
                return f"slice {r} should have been left untouched"
        elif not torch.equal(got, expected):
            bad = (got != expected).nonzero()[0].tolist()
            return f"slice {r} differs at {bad}: got {int(got[tuple(bad)])}, expected {int(expected[tuple(bad)])}"
    return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--collective", choices=["allgather", "scatter", "gather"], default="allgather")
    parser.add_argument("--root", type=int, default=0, help="Root rank (scatter and gather)")
    parser.add_argument("-m", type=int, default=64, help="Rows per rank")
    parser.add_argument("-n", type=int, default=128, help="Columns")
    parser.add_argument("--tile_rows", type=int, default=8, help="Rows per tile (tiles span the full width)")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs to run (the plan re-arms between them)")
    parser.add_argument("--heap_size", type=int, default=1 << 28)
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="gloo")
    ctx = iris.iris(args.heap_size, allocator_type="vmem_chunked", engines=("sdma",))
    rank, world = ctx.get_rank(), ctx.get_num_ranks()
    rows, cols = args.m, args.n

    buffer = ctx.zeros((world * rows, cols), dtype=torch.int32)
    tile_shape = (args.tile_rows, cols)
    if args.collective == "allgather":
        plan = ctx.triggered.allgather(buffer, tile_shape)
    else:
        plan = getattr(ctx.triggered, args.collective)(buffer, args.root, tile_shape)
    if rank == 0:
        print(f"armed: {plan.num_batches} tiles, {world} rank(s), collective={args.collective}", flush=True)

    status = torch.zeros(plan.num_batches, dtype=torch.int32, device=buffer.device)
    ok = True
    try:
        for epoch in range(1, args.epochs + 1):
            if epoch > 1:
                plan.next_epoch()
            seed(buffer, args.collective, args.root, rank, world, rows, cols, epoch)
            ctx.barrier()
            status.zero_()
            view = plan.begin_device_epoch()
            drive[(plan.num_batches,)](view, status, BUDGET=SPIN_BUDGET)
            torch.cuda.synchronize()
            seen = status.cpu()
            publish = int(((seen & SEEN_PUBLISH) != 0).sum())
            consume = int(((seen & SEEN_CONSUME) != 0).sum())
            timeouts = int(((seen & SEEN_TIMEOUT) != 0).sum())
            print(
                f"rank {rank} epoch {epoch} roles: {publish} publish, {consume} consume (of {plan.num_batches} tiles)",
                flush=True,
            )
            if timeouts:
                ok = False
                print(f"rank {rank} FAIL: {timeouts} tile(s) exceeded the spin budget", flush=True)
            try:
                plan.end_device_epoch()
            except TimeoutError as exc:
                ok = False
                print(f"rank {rank} FAIL: end_device_epoch: {exc}", flush=True)
            want = expected_slices(args.collective, args.root, rank, world, rows, cols, epoch, buffer.device)
            problem = verify(buffer, want, rows)
            if problem:
                ok = False
                print(f"rank {rank} FAIL (epoch {epoch}): {problem}", flush=True)
            if not ok:
                break
    finally:
        plan.abort(timeout=10)
        plan.destroy()
    if ok:
        print(
            f"rank {rank} OK: {args.collective} correct over {args.epochs} epoch(s), driven from the device", flush=True
        )
    ctx.barrier()
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
