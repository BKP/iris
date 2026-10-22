# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

# sdma_lanes on the Iris context: argument checks, creation of one host channel per lane and the channel
# that host puts check for armed triggered chains (host only; a stand-in for ``xio.sdma_ep``, no GPU).

from types import SimpleNamespace

import pytest

from iris import Iris
from iris.host.iris import _require_multi_channel, _validate_sdma_lanes

SRC, DST = 0x1000, 0x9000


class FakeSdmaEp:
    def __init__(self, channels=1, create_error=None):
        self.calls = []
        self.channels = channels
        self.create_error = create_error

    def create_host_queue(self, *args, **kwargs):
        self.calls.append(("create_host_queue", args, kwargs))
        if self.create_error:
            raise self.create_error

    def get_host_channel_count(self, *args):
        self.calls.append(("get_host_channel_count", args))
        return self.channels

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


def make_ctx(lanes=1, **ep_kwargs):
    ep = FakeSdmaEp(**ep_kwargs)
    idle_checks = []
    ctx = SimpleNamespace(
        sdma_lanes=lanes,
        _sdma_ep=ep,
        get_rank=lambda: 0,
        get_num_ranks=lambda: 2,
        _require_copy_engine=lambda: ep,
        _check_host_queue_idle=lambda rank, channel=0: idle_checks.append((rank, channel)),
        heap=SimpleNamespace(translate=lambda ptr, src_rank, dst_rank: ptr + 0x4000_0000),
    )
    ctx._flag_pointer_and_bits = lambda flag, **kw: Iris._flag_pointer_and_bits(ctx, flag, **kw)
    return ctx, ep, idle_checks


def test_sdma_lanes_must_be_a_positive_integer():
    for lanes in (0, -1, True, 2.0, "2", None):
        with pytest.raises(ValueError, match="positive integer"):
            _validate_sdma_lanes(lanes, ("sdma",))


def test_sdma_lanes_above_one_need_the_sdma_engine():
    _validate_sdma_lanes(1, ())
    _validate_sdma_lanes(3, ("sdma",))
    with pytest.raises(ValueError, match='engines=\\("sdma",\\)'):
        _validate_sdma_lanes(2, ())


def test_several_lanes_need_a_rocm_xio_that_can_create_channels():
    _require_multi_channel(FakeSdmaEp())
    with pytest.raises(RuntimeError, match="update rocm-xio"):
        _require_multi_channel(SimpleNamespace(create_host_queue=None))


def test_one_lane_creates_the_single_channel_every_rocm_xio_can():
    ctx, ep, _ = make_ctx(lanes=1)
    Iris._create_host_queues(ctx, 0, 1)
    assert ep.calls == [("create_host_queue", (0, 1), {})]  # no num_channels: works with the old binding


def test_several_lanes_create_one_channel_each():
    ctx, ep, _ = make_ctx(lanes=3, channels=3)
    Iris._create_host_queues(ctx, 0, 1)
    assert ep.calls == [
        ("create_host_queue", (0, 1), {"num_channels": 3}),
        ("get_host_channel_count", (0, 1)),
    ]


def test_missing_channels_are_an_error():
    ctx, _, _ = make_ctx(lanes=3, channels=2)
    with pytest.raises(RuntimeError, match="has 2 host channels, sdma_lanes=3"):
        Iris._create_host_queues(ctx, 0, 1)


def test_a_failed_channel_creation_names_the_request():
    ctx, _, _ = make_ctx(lanes=4, create_error=RuntimeError("initHostQueue failed"))
    message = "could not create sdma_lanes=4 host channels from device 0 to device 1"
    with pytest.raises(RuntimeError, match=message) as e:
        Iris._create_host_queues(ctx, 0, 1)
    assert "initHostQueue failed" in str(e.value)


def test_idle_check_forwards_the_queue_index_to_the_ledger():
    checked = []
    ledger = SimpleNamespace(check_idle=lambda peer, queue: checked.append((peer, queue)))
    ctx = SimpleNamespace(_triggered_ledger=ledger)
    Iris._check_host_queue_idle(ctx, 2, 1)
    Iris._check_host_queue_idle(ctx, 3)
    assert checked == [(2, 1), (3, 0)]
    Iris._check_host_queue_idle(SimpleNamespace(_triggered_ledger=None), 2, 1)  # no plan yet: nothing to check


def test_host_puts_check_the_channel_they_use():
    ctx, ep, idle = make_ctx(lanes=3)
    Iris.put(ctx, FakeTensor(SRC, 256), to_rank=1, channel=2, async_op=True)
    Iris.put_tile(ctx, SRC, 1, DST, 4, 8, 16, 64, 128, channel=1, async_op=True)
    Iris.put_tiles(ctx, [SRC], 1, [DST], 4, [8], [16], [64], [128], channel=2, async_op=True)
    assert idle == [(1, 2), (1, 1), (1, 2)]
    assert [call[1][2] for call in ep.calls] == [2, 1, 2]  # and the copy went to that channel


def test_quiet_checks_the_channel_it_waits_on():
    ctx, ep, idle = make_ctx(lanes=3)
    Iris.quiet(ctx, to_rank=1, channel=2)
    assert idle == [(1, 2)]
    idle.clear()
    Iris.quiet(ctx, channel=1)
    assert idle == [(0, 1), (1, 1)]
    quiets = [call for call in ep.calls if call[0] == "quiet"]
    assert quiets == [("quiet", (0, 1, 2)), ("quiet", (0, 0, 1)), ("quiet", (0, 1, 1))]
