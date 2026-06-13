# `cudaMemcpyPeerAsync` rejected during CUDA stream capture on WSL2

Minimal reproducer for a CUDA stream-capture limitation observed on WSL2.

> **Status (pending native-Linux confirmation):** the failure below is confirmed
> on WSL2. Whether it is *WSL-specific* (a driver bug) or *general CUDA
> behaviour* is decided by running `matrix.cu` on native Linux — see
> [`NATIVE-TEST.md`](NATIVE-TEST.md). NVIDIA docs state peer copies *can* be
> captured, so native behaviour is the deciding evidence. Do not treat the
> "driver bug" framing as settled until that diff is in.

**The crux:** on WSL2, `cudaMemcpyPeerAsync` issued inside a stream capture fails
with `operation not permitted when stream is capturing`, while the
*semantically identical* peer transfer expressed as
`cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)` over peer-enabled pointers is
**accepted** during capture. Same two GPUs, same NVLink, same bytes, yet one API
is capturable and the other is not. Peer access is enabled, the peer copy
succeeds outside capture, and single-GPU graph capture works. Only
`cudaMemcpyPeerAsync` issued during capture fails.

## Environment

| | |
|---|---|
| Platform | WSL2 (`6.18.33.1-microsoft-standard-WSL2`) |
| GPUs | 2× Quadro RTX 6000 (Turing TU102), NVLink, P2P enabled |
| WSL display driver | 610.47 |
| CUDA toolkit | 13.3 (V13.3.33); reproduced identically under 13.2 |

## Build & run

```sh
make run          # builds and runs both binaries
# or individually:
nvcc -o repro repro.cu             && ./repro
nvcc -o workaround workaround.cu   && ./workaround
```

Requires two P2P-capable GPUs. `repro` exits 0 when the failure reproduces;
`workaround` exits 0 when both mitigations succeed. Captured reference output is
in [`EXPECTED-OUTPUT.txt`](EXPECTED-OUTPUT.txt).

## What `repro` shows

It confirms everything *around* the failure works, so the failure is isolated to
one API/context combination:

```
[1]  canAccessPeer 0<->1 = 1   ; cudaDeviceEnablePeerAccess -> no error
[2]  single-GPU stream capture -> no error  (begin / end / instantiate / launch)
[3]  cudaMemcpyPeerAsync OUTSIDE capture -> no error
[4]  cudaMemcpyPeerAsync INSIDE  capture -> operation not permitted when stream is capturing
       cudaStreamEndCapture           -> operation failed due to a previous error during capture
```

Only case **[4]** fails.

## Mitigation

With peer access enabled, the remote device pointer is directly addressable on
the local device via UVA, so the same transfer can be issued in a capturable
form. `workaround.cu` demonstrates two, both verified to capture, instantiate,
replay, and produce the correct cross-device result (`1.0 + 2.0 == 3.0`):

**(A) `cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, stream)`** —
recommended drop-in. Captured as a memcpy node. Issue it only while a capture is
active and keep `cudaMemcpyPeerAsync` on the non-capture path:

```c
if (capture_active) {
    // capturable: direct peer DtoD copy via UVA, peer access already enabled
    cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, src_stream);
} else {
    cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, nbytes, src_stream);
}
```

Preconditions: `cudaDeviceEnablePeerAccess` has been called for the device pair
(otherwise the DtoD copy stages through the host instead of taking the direct
peer path), and both pointers are plain UVA device allocations.

**(B) a peer-access kernel** — a kernel on the local device that dereferences the
remote pointer directly. Captured as a kernel node. Useful when the transfer is
already fused with computation (e.g. a reduce/accumulate), avoiding a separate
copy.

## Expected behaviour / why this looks like a driver bug

Both forms in `workaround.cu` produce the same peer transfer as the rejected
`cudaMemcpyPeerAsync`, and the WSL2 driver captures both without complaint. The rejection in case [4] is
therefore an inconsistency in capture support for one API, not a fundamental P2P
or graph-capture limitation. Per the CUDA Programming Guide, peer memcpy
operations are representable as graph memcpy nodes and are expected to be
capturable.

> Not verified in this repo's environment: behaviour on native (non-WSL) Linux.
> This reproducer was run only on WSL2. The claim above rests on the CUDA
> Programming Guide and on the internal inconsistency shown here, both of which
> are independent of a native-Linux comparison.

## Why it matters

CUDA graph capture/replay collapses many individual launches into a single graph
submission. On WSL2 each CUDA launch crosses the paravirtualization boundary, so
per-launch overhead is *higher* than native, which makes capture more valuable on
WSL2, not less. A single `cudaMemcpyPeerAsync` in a multi-GPU decode/reduce path
forces that path back to eager dispatch and forfeits the gain. Mitigation (A)
restores it with no change to the non-capture path.

## Files

- `repro.cu` — isolates the failure (cases [1]–[4]); prints a `RESULT:` verdict.
- `workaround.cu` — demonstrates the two capturable mitigations (A) and (B).
- `EXPECTED-OUTPUT.txt` — captured output of both binaries.
- `Makefile` — `make`, `make run`, `make clean`.
