# Iris Class

## Factory Function

Prefer using the convenience factory over calling the constructor directly:

```{eval-rst}
.. autofunction:: iris.host.iris.iris
```

## Optional SDMA Copy Engine

The `engines` parameter defaults to `()`. In this mode, Iris skips SDMA
initialization and queue creation, and `get_copy_engine_ctx()` returns `None`.
Regular Triton and Gluon memory operations and collectives work without SDMA.

Pass `engines=("sdma",)` to initialize both host- and device-initiated SDMA queues.
This needs the `sdma` extra ([Installation](../../getting-started/installation.md)):

```python
ctx = iris.iris(heap_size=1 << 30, engines=("sdma",))
copy_engine_ctx = ctx.get_copy_engine_ctx()
```

This is required for the host methods `put`, `put_tile`, `put_tiles`, and `quiet`,
and for device operations using `use_copy_engine=True`. Host SDMA methods raise
`RuntimeError` if the copy engine is disabled. Pass the returned context to
device operations through their `copy_engine_ctx` argument.

### SDMA lanes

Host-initiated SDMA work to one peer runs on an ordered queue: a copy that waits on a flag holds
back everything queued behind it. `sdma_lanes` creates several independent lanes to each peer, so
unrelated work does not wait on it. Each lane runs on its own host queue; `put` and `quiet` name a
queue with `channel=`, which is rocm-xio's word for the index of a queue within a GPU pair (it is
not a routing channel as in collective libraries). Lane `i` currently uses queue `i`. The default
is one lane.

```python
ctx = iris.iris(heap_size=1 << 30, engines=("sdma",), sdma_lanes=2)
```

`sdma_lanes` needs `engines=("sdma",)`. Values above 1 need a rocm-xio whose `sdma_ep` can create
several host queues per peer (`create_host_queue(..., num_channels=N)`); Iris raises
`RuntimeError` at creation if the installed one cannot. All lanes to one peer share the same copy
engine, so lanes remove blocking but add no bandwidth. Each lane holds its own queue, and the driver
limits how many a process can create. [Triggered SDMA](../../conceptual/triggered-sdma.md) plans
assign batches to lanes with `Batch.lane`.

```{eval-rst}
.. automethod:: iris.host.iris.Iris.get_copy_engine_ctx
```

## Core Methods

```{eval-rst}
.. automethod:: iris.host.iris.Iris.get_heap_bases
.. automethod:: iris.host.iris.Iris.get_device_context
.. automethod:: iris.host.iris.Iris.barrier
.. automethod:: iris.host.iris.Iris.get_device
.. automethod:: iris.host.iris.Iris.get_cu_count
.. automethod:: iris.host.iris.Iris.get_rank
.. automethod:: iris.host.iris.Iris.get_num_ranks
```

## Logging Helpers

Use Iris-aware logging that automatically annotates each message with the current rank and world size. This is helpful when debugging multi-rank programs.

```{eval-rst}
.. autofunction:: iris.host.logging.logging.set_logger_level
.. automethod:: iris.host.iris.Iris.debug
.. automethod:: iris.host.iris.Iris.info
.. automethod:: iris.host.iris.Iris.warning
.. automethod:: iris.host.iris.Iris.error
```

## Utility Functions

```{eval-rst}
.. autofunction:: iris.host.platform.utils.do_bench
```

## Broadcast Helper

Broadcast data from a source rank to all ranks. This method automatically detects whether the value is a tensor/array or a scalar and uses the appropriate broadcast mechanism.

```{eval-rst}
.. automethod:: iris.host.iris.Iris.broadcast
```

## allocate_symmetric

```{eval-rst}
.. automethod:: iris.host.iris.Iris.allocate_symmetric
```

## device_barrier

```{eval-rst}
.. automethod:: iris.host.iris.Iris.device_barrier
```

## get_device_id

```{eval-rst}
.. automethod:: iris.host.iris.Iris.get_device_id
```

## ops

```{eval-rst}
.. autoproperty:: iris.host.iris.Iris.ops
```
