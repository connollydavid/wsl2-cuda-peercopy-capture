# Cross-platform confirmation: WSL2 vs native Linux

This note records the comparison that establishes the behaviour described in the
[README](README.md) as general CUDA, not a WSL-specific quirk. Background:
[issue #1](https://github.com/connollydavid/capturable-peer-copy-cuda/issues/1).

## Result (2026-06-13)

**Native Linux matches WSL2, byte for byte.** `matrix.cu` was run on native
(non-WSL) Arch Linux with the same 2× Quadro RTX 6000 + P2P. The
`cudaMemcpyPeerAsync`-inside-capture row reads `operation not permitted` on
native, exactly as on WSL2. Built under matching CUDA 13.3 (V13.3.33, from
`/opt/cuda-13.3`), `diff matrix-wsl2.txt matrix-native.txt` produces no output.

It is the same CUDA 13.3 (V13.3.33) userspace install exercised under two
kernel/boot contexts: WSL2's paravirtualised kernel and the native Linux kernel.
Identical userspace yielding identical output isolates the behaviour to CUDA's
runtime, independent of the kernel/driver context. That independence is why it is
not WSL-specific.

| | |
|---|---|
| Native kernel | `7.0.10-arch1-1` (Arch Linux, non-WSL) |
| GPUs | 2× Quadro RTX 6000, NVLink, P2P enabled |
| Driver | 610.43.02 |
| CUDA toolkit | 13.3 (V13.3.33), `/opt/cuda-13.3` |

## Reproduce the comparison

`matrix.cu` is the platform-neutral peer-copy-during-capture sweep. WSL2 output
is captured in `matrix-wsl2.txt`. Run the same binary on another platform and
diff:

```sh
git clone https://github.com/connollydavid/capturable-peer-copy-cuda
cd capturable-peer-copy-cuda
nvcc -o matrix matrix.cu
./matrix | tee matrix-native.txt
diff matrix-wsl2.txt matrix-native.txt   # no output => identical behaviour
```

The deciding row is `cudaMemcpyPeerAsync` inside capture. On both platforms, all
three capture modes give:

```
cudaMemcpyPeerAsync   peer=1   enqueue=operation not permitted when stream is capturing   end=ERR
```

`no error` on a platform would mean the constraint is specific to that platform's
driver; `operation not permitted` (what both platforms show) means it is general
CUDA behaviour.
