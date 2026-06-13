# Capturing peer-to-peer copies in CUDA graphs

A short technical note, with a runnable demonstration, on a CUDA stream-capture
detail that is easy to hit and not spelled out in the documentation:
**`cudaMemcpyPeerAsync` (and `cudaMemcpy3DPeerAsync`) cannot be captured into a
CUDA graph.** Issuing one while a stream is capturing returns `operation not
permitted when stream is capturing`. To put a cross-device copy inside a captured
graph, express it in a capturable form instead.

This is CUDA runtime behaviour, confirmed byte-for-byte identical on WSL2 and
native Linux (same GPUs, same toolkit). It is not a WSL bug and not a defect to
report; this note exists because the constraint is undocumented and the fix is
not obvious.

## TL;DR

| Form of the peer copy | Inside stream capture? |
|---|---|
| `cudaMemcpyPeerAsync` / `cudaMemcpy3DPeerAsync` | **rejected** (`operation not permitted`) |
| `cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, stream)` over peer-enabled UVA pointers | **captured** (memcpy node) |
| a kernel that dereferences the remote device pointer | **captured** (kernel node) |

Peer access must be enabled (`cudaDeviceEnablePeerAccess`) for the two capturable
forms to take the direct P2P path rather than staging through host memory.

## The behaviour

`demo.cu` isolates the constraint by showing that everything around it works, and
only the one combination fails:

```
[1]  peer access enabled            cudaDeviceEnablePeerAccess -> no error
[2]  single-GPU stream capture      begin / end / instantiate / launch -> no error
[3]  cudaMemcpyPeerAsync OUTSIDE capture -> no error
[4]  cudaMemcpyPeerAsync INSIDE  capture -> operation not permitted when stream is capturing
        cudaStreamEndCapture            -> operation failed due to a previous error during capture
```

[1]–[3] pass, [4] fails. P2P works, capture works; only a peer copy *recorded
during* capture fails, and that failure then poisons the capture so
`cudaStreamEndCapture` cannot build the graph.

## Why it happens

Stream capture records each enqueued async operation as a graph node. A graph
memcpy node is described by `cudaMemcpy3DParms`, which addresses memory by UVA
pointer and has no source/destination *device id* fields. `cudaMemcpyPeerAsync`
is defined in terms of explicit device ids, and there is no explicit-device-id
"peer memcpy node" for the runtime to translate it into. With no node to record,
capture rejects the call.

The capturable forms sidestep this because both *do* have node representations: a
plain `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, ...)` over peer-enabled UVA
pointers records as an ordinary memcpy node (the peer pointer is directly
addressable, so no device ids are needed), and a kernel that reads the remote
pointer records as a kernel node.

This reading is consistent with the documentation rather than contradicted by it:
the CUDA Programming Guide's list of prohibited operations during capture does
not mention peer memcpy, and nothing in the guide states that
`cudaMemcpyPeerAsync` is capturable. The only "peer" material in the graphs
chapter concerns peer accessibility of graph-owned memory-pool allocations, which
is a different feature.

## The capturable forms

`capturable.cu` performs the same cross-device transfer two ways, and verifies
each captures, instantiates, replays, and produces the correct result across the
peer boundary (`1.0 + 2.0 == 3.0`).

**(A) Device-to-device memcpy over UVA — recommended drop-in.** Captured as a
memcpy node. Keep `cudaMemcpyPeerAsync` on the non-capture path and switch to
this form while a capture is active:

```c
if (capture_active) {
    // capturable: direct peer DtoD copy via UVA, peer access already enabled
    cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream);
} else {
    cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, nbytes, stream);
}
```

Preconditions: `cudaDeviceEnablePeerAccess` has been called for the device pair
(otherwise the DtoD copy stages through the host instead of taking the direct
peer path), and both pointers are plain UVA device allocations.

**(B) A peer-access kernel.** A kernel on the local device that dereferences the
remote pointer directly, captured as a kernel node. Useful when the transfer is
already fused with computation (for example a reduce or accumulate), avoiding a
separate copy.

## Why it matters

Graph capture and replay collapse many individual launches into a single graph
submission, which removes per-launch CPU dispatch overhead. A single
`cudaMemcpyPeerAsync` on a multi-GPU reduce or exchange path forces that path
back to eager dispatch and forfeits the gain, with no error until you try to
capture. The overhead is larger on WSL2, where every launch crosses the
paravirtualisation boundary, so the constraint is most visible there even though
it is not WSL-specific. Form (A) restores capture with no change to the
non-capture path.

## Run it

Requires two P2P-capable GPUs.

```sh
make run          # builds and runs all three
# or individually:
nvcc -o demo demo.cu               && ./demo          # the behaviour, cases [1]-[4]
nvcc -o capturable capturable.cu   && ./capturable    # the two capturable forms
nvcc -o matrix matrix.cu           && ./matrix        # full sweep (see below)
```

`demo` exits 0 when the system behaves as described; `capturable` exits 0 when
both forms work. Captured reference output is in
[`EXPECTED-OUTPUT.txt`](EXPECTED-OUTPUT.txt).

## Reproducibility across platforms

`matrix.cu` is a platform-neutral sweep: every peer-copy form
(`peerAsync` / `3DPeerAsync` / D2D / Default-UVA / kernel) crossed with every
capture mode and with peer access on and off, each row reporting enqueue result,
end-capture result, and replay correctness. `matrix-wsl2.txt` is the captured
WSL2 sweep; running the same binary on native Linux produces a byte-for-byte
identical table. See [`NATIVE-TEST.md`](NATIVE-TEST.md) for that comparison,
which is what establishes the behaviour as general CUDA rather than platform
specific.

| | |
|---|---|
| GPUs | 2× Quadro RTX 6000 (Turing TU102), NVLink, P2P enabled |
| Platforms | WSL2 (`6.18.33.1-microsoft-standard-WSL2`) and native Linux (`7.0.10-arch1-1`) |
| Driver | WSL display 610.47 · native 610.43.02 |
| CUDA toolkit | 13.3 (V13.3.33); same behaviour under 13.2 |

## Files

- `demo.cu` — the behaviour, cases [1]–[4]; prints a `RESULT:` line.
- `capturable.cu` — the two capturable forms (A) and (B), with correctness checks.
- `matrix.cu` — full peer-copy × capture-mode × peer-access sweep.
- `matrix-wsl2.txt` — captured sweep on WSL2; identical on native Linux.
- `EXPECTED-OUTPUT.txt` — captured output of `demo` and `capturable`.
- `NATIVE-TEST.md` — the cross-platform comparison and its verdict.
- `Makefile` — `make`, `make run`, `make clean`.
