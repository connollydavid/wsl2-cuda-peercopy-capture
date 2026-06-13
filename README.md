# `cudaMemcpyPeerAsync` rejected during CUDA stream capture on WSL2

Minimal reproducer for a WSL2-specific CUDA limitation: **`cudaMemcpyPeerAsync`
fails with `operation not permitted when stream is capturing` when issued inside
a stream capture**, even though peer-to-peer access is enabled, the same copy
succeeds outside capture, and single-GPU graph capture works. The
*semantically identical* transfer expressed as `cudaMemcpyAsync(...,
cudaMemcpyDeviceToDevice)` over peer-enabled pointers **is** accepted during
capture — so this is an inconsistency in the WSL2 driver's capture support, not
a fundamental P2P or graph limitation.

On native Linux with the same hardware, `cudaMemcpyPeerAsync` captures fine (it
becomes a memcpy node in the graph).

## Environment

| | |
|---|---|
| Platform | WSL2 (`6.18.33.1-microsoft-standard-WSL2`) |
| GPUs | 2× Quadro RTX 6000 (Turing TU102), NVLink, P2P enabled |
| WSL display driver | 610.47 |
| CUDA toolkit | 13.3 (V13.3.33); also reproduced under 13.2 |

## Build & run

```sh
make run         # builds and runs both binaries
# or:
nvcc -o repro repro.cu && ./repro
nvcc -o workaround workaround.cu && ./workaround
```

Requires two P2P-capable GPUs.

## What the reproducer shows

`repro.cu` isolates the failure by confirming everything *around* it works:

```
[1]  canAccessPeer 0<->1 = 1          ; cudaDeviceEnablePeerAccess -> no error
[2]  single-GPU stream capture        -> no error  (begin/end/instantiate/launch)
[3]  cudaMemcpyPeerAsync OUTSIDE capture -> no error
[4]  cudaMemcpyPeerAsync INSIDE capture  -> operation not permitted when stream is capturing
       cudaStreamEndCapture            -> operation failed due to a previous error during capture
```

Only case **[4]** fails. P2P works, capture works, peer copy works — only the
*combination* is rejected.

## Mitigation

Peer access being enabled means the remote device pointer is directly
addressable on the local device via UVA. The same byte-for-byte transfer can
therefore be expressed in two capture-compatible forms, both demonstrated in
`workaround.cu` and both verified on the environment above:

**(A) `cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, stream)`** —
recommended. The drop-in replacement. With peer access enabled the runtime
performs the identical direct peer transfer over NVLink/PCIe; it is captured as
a memcpy node. Apply it only while a capture is active and keep
`cudaMemcpyPeerAsync` on the non-capture path:

```c
if (capture_active) {
    // capturable: direct peer DtoD copy via UVA, peer access already enabled
    cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, src_stream);
} else {
    cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, nbytes, src_stream);
}
```

Preconditions: `cudaDeviceEnablePeerAccess` has been called for the relevant
device pair (otherwise the DtoD copy degrades to a host-staged copy instead of a
direct peer transfer), and both pointers are plain UVA device allocations.

**(B) a peer-access kernel** — a kernel on the local device that dereferences the
remote pointer directly (loads/stores over enabled peer access). Captured as a
kernel node. Useful when the transfer is already fused with computation (e.g. a
reduce/accumulate), since it avoids a separate copy entirely.

Both forms capture, instantiate, replay, and produce correct results
(`workaround.cu` checks `1.0 + 2.0 == 3.0` across the peer boundary).

## Why it matters

Graph capture/replay collapses many individual kernel/copy launches into a
single graph submission. Under WSL2 each CUDA launch crosses the
paravirtualization boundary, so per-launch overhead is *higher* than native —
making graph capture more valuable on WSL2, not less. A single non-capturable
`cudaMemcpyPeerAsync` in a multi-GPU decode/reduce path forces the entire path
back to eager dispatch, forfeiting that gain. Mitigation (A) restores it.

## Files

- `repro.cu` — isolates the failure (cases [1]–[4]).
- `workaround.cu` — demonstrates the two capturable mitigations (A) and (B).
- `Makefile` — `make`, `make run`, `make clean`.
