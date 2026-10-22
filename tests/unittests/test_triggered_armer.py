# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Call convention of the triggered armer against a recording stand-in for ``xio.sdma_ep`` (host only).
# rocm-xio 5c3bc93 takes ``(dst, src)`` and raw tile arguments; a swapped pair is silent on hardware.

from types import SimpleNamespace

from iris.triggered import armer

SRC, DST = 0x1000, 0x9000


class FakeSdmaEp:
    COPY_LINEAR_COMMAND_BYTES = 32
    COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES = 48
    ATOMIC_COMMAND_BYTES = 36

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, args))

        return record


def test_linear_chain_passes_dst_before_src():
    ep = FakeSdmaEp()
    copies = [armer.Copy(SRC, DST, 256, 1, 256), armer.Copy(SRC + 256, DST + 256, 128, 1, 128)]
    armer.arm_chain(ep, 0, 1, 0xA000, 7, copies, 0xB000, False)
    assert ep.calls == [
        ("wait_flag_then_put", (0, 1, 0, 0xA000, 7, DST, SRC, 256, 32)),
        ("put", (0, 1, 0, DST + 256, SRC + 256, 128)),
        ("signal", (0, 1, 0, 0xB000, 1, 64)),
    ]


def test_2d_chain_uses_raw_tile_arguments_in_bytes():
    ep = FakeSdmaEp()
    copies = [armer.Copy(SRC, DST, 64, 8, 256), armer.Copy(SRC + 64, DST + 64, 32, 4, 256)]
    armer.arm_chain(ep, 2, 3, 0xA000, 5, copies, 0xB000, True)
    wait, signal = ep.calls
    assert wait == (
        "wait_flag_then_put_tiles",
        (2, 3, 0, 0xA000, 5, [DST, DST + 64], [SRC, SRC + 64], 1, [8, 4], [64, 32], [256, 256], [256, 256]),
    )
    assert signal == ("signal", (2, 3, 0, 0xB000, 1, 64))


def test_channel_is_passed_through():
    ep = FakeSdmaEp()
    armer.arm_chain(ep, 0, 1, 0xA000, 1, [armer.Copy(SRC, DST, 64, 1, 64)], 0xB000, False, channel=3)
    assert {args[2] for _, args in ep.calls} == {3}


def test_chain_bytes_uses_the_context_constants():
    ep = FakeSdmaEp()
    linear = SimpleNamespace(is_2d=False, segments=[0, 1])
    tiled = SimpleNamespace(is_2d=True, segments=[0, 1, 2])
    assert armer.chain_bytes(ep, linear) == armer.POLL_REGMEM_BYTES + 2 * 32 + 36
    assert armer.chain_bytes(ep, tiled) == armer.POLL_REGMEM_BYTES + 3 * 48 + 36
