# GPU-initiated RDMA

A Triton kernel can write to a rank on **another host** through an RDMA NIC, without involving the CPU: the kernel
claims a send-queue slot, builds the NIC's work request (WQE) in memory, rings the doorbell and polls the completion
queue. It is the RDMA counterpart of the SDMA device-initiated path, and it is separate from [triggered
SDMA](triggered-sdma.md): there the host pre-arms the copies and a kernel only flips a gate; here the kernel posts the
work itself.

Status: NVIDIA/Mellanox `mlx5` NICs only (rocm-xio's other providers have no Triton-side implementation yet). It needs a
rocm-xio built with `-DGDA_MLX5=ON -DXIO_BUILD_PYTHON=ON`.

## Setting it up

```python
ctx = iris.iris(heap_size, allocator_type="vmem_chunked", engines=("rdma",), rdma_device="mlx5_0")
```

`engines=("rdma",)` changes how the context is built:

- The job may span hosts without a GPU fabric domain, but peers on other hosts are **not mapped**: there is no
  load/store to them, only RDMA.
- The heap is **one physical chunk** (`heap_size` must be a power of two with `vmem_chunked`), registered with the NIC
  as a single region by dmabuf. (A heap made of several physical chunks cannot be registered whole.)
- Every rank's real heap base stays in `heap_bases`, so Iris's address translation yields the **remote address** the
  NIC writes to.
- The send and completion queues live in **GPU memory** (created through DevX and registered by dmabuf) on an
  InfiniBand port; on a port that cannot do that rocm-xio keeps them in host memory. The NIC reads work requests from
  and writes completions to the queues, and the kernels read and write them locally.
- One connected queue pair per peer is created; connection information moves through the process group.
  `ctx.rdma_device_ctx` is the tensor of words kernels read (one row per peer).

## In a kernel

```python
from iris.mem.triton.ops import rdma_put_signal, rdma_quiet

@triton.jit
def send(data, flag, rank, peer, heap_bases, nbytes, rdma_ctx, status):
    index = rdma_put_signal(data, data, flag, rank, peer, heap_bases, nbytes, 1, rdma_ctx)
    error = rdma_quiet(peer, index, rdma_ctx)       # 0 = fine, else 1 + the NIC's error syndrome
    tl.store(status, error.to(tl.int32))
```

`rdma_put` writes `nbytes` contiguous bytes to the same symmetric address on the peer; `rdma_atomic_add` does a remote
fetch-add; `rdma_put_signal` is a write followed, on the same queue pair, by an atomic add to a flag, so a peer that
sees the flag sees the data (a Triton kernel on the peer polls the flag with an acquire atomic). The lower level is
`iris.device.rdma_utils` (`put`, `atomic_add`, `quiet`, `poll_cq`).

## Triggered plans over RDMA

A `TriggeredPlan` (see [triggered SDMA](triggered-sdma.md)) can send to a peer on another host: create the context with
`engines=("sdma", "rdma")` and use the plan, or the collectives, as usual. For each batch and each remote peer the host
posts the batch's RDMA writes **dormant** into a dedicated queue pair (one per peer and lane, queues in GPU memory), then
a write of the epoch into the peer's arrival gate, and arms a copy-engine chain: wait for the batch's ready gate, then
write the send queue's doorbell record and the NIC's doorbell. `publish` in the producer kernel opens the gate and the
copy engine rings the NIC; no CPU is involved while the epoch runs. The receiving side is unchanged: it waits on its own
arrival gate.

- The arrival gate has exactly one writer per epoch, so the epoch is **written**; a design with several writers into one
  gate would need an atomic add.
- A rank learns that a batch it sent to an RDMA peer is done from the completion the NIC posts once the peer acknowledged
  the data and the arrival write. The host polls those in `end_device_epoch()`, then marks the batch complete, so
  `reusable(batch)` on the sending rank becomes true then (not before the host call).
- A triggered queue pair holds 1,024 work requests and 512 completions per epoch: a batch of S contiguous segments takes
  S + 1 slots and one completion (a strided segment takes one slot per row). Plans that need more are refused at creation.
- The work requests are written by the CPU into GPU memory; Iris drains those writes to the device before arming any
  chain that can ring the doorbell.
- Limits: `mlx5` and InfiniBand, and one rank per host, so every peer is reached over RDMA.

## Things to know

- Use `dist.barrier()`, not `ctx.barrier()`, while a kernel is spinning for a signal: `ctx.barrier()` first waits for
  the local GPU to go idle.
- A queue pair is driven either by the host or by kernels, not both. Iris posts from kernels only.
- Work requests are claimed in order and their doorbells are rung in claim order; many programs may post on one queue
  pair. If the queue is full a program drains the completion queue itself.
- A failed work request leaves the error word set (`rdma_quiet` returns it): `0x04` local protection error, `0x13`
  remote access error, `0x15` retries exhausted (often the peer's queue pair died: read both sides).
- Every peer is reached over the NIC, so use one rank per host. Same-host peers are not mapped in this mode.
- Only mlx5 is supported, and only an InfiniBand port is tested.
- Wait with `rdma_quiet` regularly. A single kernel that posted 300 write requests without waiting completed, but
  350 or more did not complete in testing on two hosts (also with the completion queue drained earlier); the cause is
  not known yet, so do not post more than a few hundred requests between waits.
