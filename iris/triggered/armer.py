# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Arms triggered chains on the host-initiated copy-engine queue through rocm-xio.

Per (batch, peer): POLL_REGMEM(ready >= epoch) -> one copy per segment -> ATOMIC ADD_RTN_64(arrival += 1).
"""

from dataclasses import dataclass

CHANNEL = 0
POLL_REGMEM_BYTES = 24


@dataclass(frozen=True)
class Copy:
    src: int
    dst: int
    width: int
    height: int
    pitch: int


def chain_bytes(sdma_ep, batch):
    copy_bytes = sdma_ep.COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES if batch.is_2d else sdma_ep.COPY_LINEAR_COMMAND_BYTES
    return POLL_REGMEM_BYTES + len(batch.segments) * copy_bytes + sdma_ep.ATOMIC_COMMAND_BYTES


def arm_chain(sdma_ep, src_device, dst_device, ready_addr, epoch, copies, arrival_addr, is_2d, channel=CHANNEL):
    """Arm one chain on the host queue from ``src_device`` to ``dst_device``."""
    if is_2d:
        # data_size=1 keeps width and strides in bytes
        sdma_ep.wait_flag_then_put_tiles(
            src_device,
            dst_device,
            channel,
            ready_addr,
            epoch,
            [c.dst for c in copies],
            [c.src for c in copies],
            1,
            [c.height for c in copies],
            [c.width for c in copies],
            [c.pitch for c in copies],
            [c.pitch for c in copies],
        )
    else:
        first, *rest = copies
        sdma_ep.wait_flag_then_put(
            src_device, dst_device, channel, ready_addr, epoch, first.dst, first.src, first.width, 32
        )
        for c in rest:
            sdma_ep.put(src_device, dst_device, channel, c.dst, c.src, c.width)
    sdma_ep.signal(src_device, dst_device, channel, arrival_addr, 1, 64)
