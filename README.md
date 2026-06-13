# `cudaMemcpyPeerAsync` rejected during CUDA stream capture (WSL2 and native Linux)

Minimal reproducer for a CUDA stream-capture limitation. First observed on WSL2;
**confirmed identical on native Linux**, so it is general CUDA behaviour, not a
WSL bug.

> **Status — RESOLVED (native-Linux confirmed):** the rejection reproduces
> *identically* on native (non-WSL) Linux with the same 2× Quadro RTX 6000 + P2P
> (kernel `7.0.10-arch1-1`, driver 610.43.02, CUDA 13.3 V13.3.33). The native
> `matrix.cu` table is **byte-for-byte identical** to
> the WSL2 run (`diff matrix-wsl2.txt matrix-native.txt` → no output) — see
> [`NATIVE-TEST.md`](NATIVE-TEST.md) and
> [issue #1](https://github.com/connollydavid/wsl2-cuda-peercopy-capture/issues/1).
> Verdict: this is **general CUDA behaviour, not WSL-specific** — `cudaMemcpyPeerAsync`
> (and `cudaMemcpy3DPeerAsync`) are simply not capturable; the capturable
> mitigations below are the supported path. Not an NVIDIA/WSL bug report.

**The crux:** `cudaMemcpyPeerAsync` issued inside a stream capture fails
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
| Platform | WSL2 (`6.18.33.1-microsoft-standard-WSL2`) **and** native Linux (`7.0.10-arch1-1`, Arch) |
| GPUs | 2× Quadro RTX 6000 (Turing TU102), NVLink, P2P enabled |
| Driver | WSL display 610.47 · native 610.43.02 |
| CUDA toolkit | 13.3 (V13.3.33) — same on both platforms |

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

## Why one API and not the other

Both forms in `workaround.cu` produce the same peer transfer as the rejected
`cudaMemcpyPeerAsync`, and both capture without complaint — identically on WSL2
and native Linux. The rejection is therefore an API-level distinction in CUDA's
capture support (`cudaMemcpyPeerAsync` / `cudaMemcpy3DPeerAsync` are not
capturable; the explicit-peer copy expressed via UVA *is*), not a fundamental P2P
or graph-capture limitation and not a WSL-specific defect.

> **Native-Linux comparison (the deciding evidence):** running `matrix.cu` on
> native Linux yields a table identical to the WSL2 run — the
> `cudaMemcpyPeerAsync`-inside-capture row reads `operation not permitted` on
> both, and built under matching CUDA 13.3 the two captures are byte-for-byte
> identical — the *same* CUDA 13.3 (V13.3.33) userspace install run under two
> kernel/boot contexts (WSL2's paravirtualized kernel and the native Linux
> kernel). Identical userspace, identical result: the rejection lives in CUDA's
> runtime, not the kernel/driver layer. Per `NATIVE-TEST.md`'s interpretation,
> native = `operation not permitted` ⇒ general CUDA behaviour, not a WSL bug.
> Treat the capturable forms above as the supported way to express a peer copy
> inside a graph.

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
- `matrix.cu` — platform-neutral peer-copy × capture-mode sweep; one table row per
  case. Run on a second platform and diff against `matrix-wsl2.txt`.
- `matrix-wsl2.txt` — captured sweep on WSL2; reproduced byte-for-byte on native Linux.
- `EXPECTED-OUTPUT.txt` — captured output of `repro` and `workaround`.
- `NATIVE-TEST.md` — the native-Linux confirmation procedure and its (now resolved) verdict.
- `Makefile` — `make`, `make run`, `make clean`.
