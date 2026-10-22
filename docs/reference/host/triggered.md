# Triggered SDMA

Pre-armed copy-engine transfers released by kernels. See
[Triggered SDMA](../../conceptual/triggered-sdma.md) for the model and its ordering rules.

## TriggeredPlan
```{eval-rst}
.. autoclass:: iris.triggered.TriggeredPlan
   :members: begin_device_epoch, end_device_epoch, next_epoch, run, abort, destroy, batch_id
```

## Schedules
```{eval-rst}
.. autoclass:: iris.triggered.Segment
.. autoclass:: iris.triggered.Batch
.. autofunction:: iris.triggered.whole
.. autofunction:: iris.triggered.chunks
.. autofunction:: iris.triggered.waves
.. autofunction:: iris.triggered.tiles
```

## Failures
```{eval-rst}
.. autoclass:: iris.triggered.PlanError
.. autoclass:: iris.triggered.BatchFailure
```

## Collectives
```{eval-rst}
.. autofunction:: iris.triggered.allgather
.. autofunction:: iris.triggered.scatter
.. autofunction:: iris.triggered.gather
.. autofunction:: iris.triggered.broadcast
.. autofunction:: iris.triggered.alltoall
```
The same builders are available as `ctx.triggered.allgather(...)`, `.scatter(...)`, `.gather(...)`.

## TriggeredView (device)
```{eval-rst}
.. autoclass:: iris.triggered.TriggeredView
   :members: initialize, publish, arrived, wait, reusable, failed, can_publish, can_consume
```
