# Native-Linux confirmation (the dispositive test — DONE)

> Tracking ticket for the native run:
> [issue #1](https://github.com/connollydavid/wsl2-cuda-peercopy-capture/issues/1).

## Result — native run completed (2026-06-13)

**Native Linux matches WSL2.** The same binary was run on native (non-WSL) Arch
Linux with the same 2× Quadro RTX 6000 + P2P. The deciding
`cudaMemcpyPeerAsync`-inside-capture row reads `operation not permitted` on
native, exactly as on WSL2. Built under matching CUDA 13.3 (V13.3.33, from
`/opt/cuda-13.3`), `diff matrix-wsl2.txt matrix-native.txt` produces **no output
— byte-for-byte identical**.

This is the **same byte-identical CUDA 13.3 (V13.3.33) userspace install**
exercised under two slightly different kernel/boot contexts — WSL2's
paravirtualized kernel and the native Linux kernel. Identical userspace yielding
identical output isolates the rejection to CUDA's runtime itself, independent of
the kernel/driver context; that independence is precisely why the behaviour is
not WSL-specific.

| | |
|---|---|
| Native kernel | `7.0.10-arch1-1` (Arch Linux, non-WSL) |
| GPUs | 2× Quadro RTX 6000, NVLink, P2P enabled |
| Driver | 610.43.02 |
| CUDA toolkit | 13.3 (V13.3.33), `/opt/cuda-13.3` |

**Verdict:** native = `operation not permitted` ⇒ **general CUDA behaviour, not
WSL-specific. Not a WSL bug** (the second interpretation branch below). The repo
stands as a known-limitation + mitigation reference.

`matrix.cu` is the platform-neutral peer-copy-during-capture matrix. WSL2 output
is captured in `matrix-wsl2.txt`. Run the same binary on **native (non-WSL)
Linux** with the same 2× P2P GPUs and diff.

```sh
git clone https://github.com/connollydavid/wsl2-cuda-peercopy-capture
cd wsl2-cuda-peercopy-capture
nvcc -o matrix matrix.cu
./matrix | tee matrix-native.txt
diff matrix-wsl2.txt matrix-native.txt
```

## Interpretation

The single deciding row is `cudaMemcpyPeerAsync` **inside capture**:

- **Native = `no error` (replay OK)** → the prohibition is **WSL2-specific**. This
  is a WSL CUDA-driver bug (documented as capturable; works native, fails WSL).
  Report it (see below). — *not what happened.*
- **Native = `operation not permitted`** → the prohibition is **general CUDA
  behaviour**, not WSL-specific. Not a WSL bug. The repo stands as a
  known-limitation + mitigation reference; no NVIDIA report warranted (or, at
  most, a docs-consistency note). — **← this is the observed result (see above).**

WSL2 baseline for the deciding row (all three capture modes):

```
cudaMemcpyPeerAsync   peer=1   enqueue=operation not permitted when stream is capturing   end=ERR
```
