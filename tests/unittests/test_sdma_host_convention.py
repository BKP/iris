# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# Call convention of the host put methods against a recording stand-in for ``xio.sdma_ep`` (host only).
# rocm-xio 5c3bc93 takes ``(dst, src)`` and raw tile arguments; a swapped pair is silent on hardware, so
# these check that the destination is passed first and the source second.

from types import SimpleNamespace

import pytest

from iris import Iris

SRC, DST = 0x1000, 0x9000
REMOTE = 0x4000_0000  # what the fake heap adds when translating to a peer


class FakeSdmaEp:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, args))

        return record


class FakeTensor:
    def __init__(self, ptr, nbytes):
        self._ptr, self._nbytes = ptr, nbytes

    def data_ptr(self):
        return self._ptr

    def numel(self):
        return self._nbytes

    def element_size(self):
        return 1


def make_ctx():
    ep = FakeSdmaEp()
    ctx = SimpleNamespace(
        get_rank=lambda: 0,
        _require_copy_engine=lambda: ep,
        _check_host_queue_idle=lambda rank, channel=0: None,
        heap=SimpleNamespace(translate=lambda ptr, src_rank, dst_rank: ptr + REMOTE),
    )
    ctx._flag_pointer_and_bits = lambda flag, **kw: Iris._flag_pointer_and_bits(ctx, flag, **kw)
    return ctx, ep


def test_put_passes_dst_before_src():
    ctx, ep = make_ctx()
    Iris.put(ctx, FakeTensor(SRC, 256), to_rank=1)
    assert ep.calls == [("put", (0, 1, 0, SRC + REMOTE, SRC, 256)), ("quiet", (0, 1, 0))]


def test_put_with_wait_and_signal_passes_dst_before_src():
    ctx, ep = make_ctx()
    Iris.put(ctx, FakeTensor(SRC, 256), to_rank=1, wait_flag=0xA000, wait_value=3, signal_flag=0xB000, async_op=True)
    assert ep.calls == [
        ("wait_flag_then_put", (0, 1, 0, 0xA000, 3, SRC + REMOTE, SRC, 256, 32)),
        ("signal", (0, 1, 0, 0xB000 + REMOTE, 1, 32)),
    ]


def test_put_signal_passes_dst_before_src():
    ctx, ep = make_ctx()
    Iris.put(ctx, FakeTensor(SRC, 256), to_rank=1, signal_flag=0xB000, async_op=True)
    assert ep.calls == [("put_signal", (0, 1, 0, SRC + REMOTE, SRC, 256, 0xB000 + REMOTE, 1, 32))]


def test_put_tile_takes_raw_arguments():
    ctx, ep = make_ctx()
    Iris.put_tile(ctx, SRC, 1, DST, 4, 8, 16, 64, 128, async_op=True)
    assert ep.calls == [("put_tile", (0, 1, 0, DST, SRC, 4, 8, 16, 128, 64))]  # ..., dst_stride, src_stride


def test_put_tile_with_wait_takes_raw_arguments():
    ctx, ep = make_ctx()
    Iris.put_tile(ctx, SRC, 1, DST, 4, 8, 16, 64, 128, wait_flag=0xA000, wait_value=2, async_op=True)
    assert ep.calls == [("wait_flag_then_put_tile", (0, 1, 0, 0xA000, 2, DST, SRC, 4, 8, 16, 128, 64, 32))]


def test_put_tiles_takes_raw_arguments():
    ctx, ep = make_ctx()
    Iris.put_tiles(ctx, [SRC, SRC + 64], 1, [DST, DST + 64], 4, [8, 4], [16, 8], [64, 64], [128, 128], async_op=True)
    assert ep.calls == [
        (
            "put_tiles",
            (0, 1, 0, [DST, DST + 64], [SRC, SRC + 64], 4, [8, 4], [16, 8], [128, 128], [64, 64]),
        )
    ]


def assert_refused_while_triggered_chains_are_armed(call):
    ctx, ep = make_ctx()

    def busy(rank, channel=0):
        raise RuntimeError(f"triggered chains are armed on the copy-engine queue to rank {rank}")

    ctx._check_host_queue_idle = busy
    with pytest.raises(RuntimeError, match="triggered chains are armed"):
        call(ctx)
    assert ep.calls == []


def test_put_refuses_a_peer_with_armed_triggered_chains():
    assert_refused_while_triggered_chains_are_armed(lambda ctx: Iris.put(ctx, FakeTensor(SRC, 256), to_rank=1))


def test_put_tile_refuses_a_peer_with_armed_triggered_chains():
    assert_refused_while_triggered_chains_are_armed(lambda ctx: Iris.put_tile(ctx, SRC, 1, DST, 4, 8, 16, 64, 128))


def test_put_tiles_refuses_a_peer_with_armed_triggered_chains():
    assert_refused_while_triggered_chains_are_armed(
        lambda ctx: Iris.put_tiles(ctx, [SRC], 1, [DST], 4, [8], [16], [64], [128])
    )


def test_put_tiles_rejects_mismatched_lengths():
    ctx, ep = make_ctx()
    with pytest.raises(ValueError, match="same length"):
        Iris.put_tiles(ctx, [SRC], 1, [DST, DST + 64], 4, [8], [16], [64], [128])
    assert ep.calls == []
