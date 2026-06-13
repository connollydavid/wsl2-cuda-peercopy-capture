# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A minimal, self-contained reproducer for a single CUDA stream-capture limitation:
`cudaMemcpyPeerAsync` issued *inside* a stream capture fails with
`operation not permitted when stream is capturing`, even though the same peer
copy succeeds outside capture and the semantically identical
`cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)` over peer-enabled pointers *is*
accepted during capture. First surfaced on WSL2 and **confirmed identical on
native Linux** (same byte-identical CUDA 13.3 install, two kernel/boot contexts),
so it is general CUDA behaviour, not a WSL bug. This is not an application — it is
evidence. Each `.cu` file isolates one facet of the failure or a mitigation, and
the captured text outputs are the artifact.

## Build & run

```sh
make            # builds repro, workaround, matrix
make run        # builds all, runs repro then workaround then matrix
make clean
nvcc -o repro repro.cu && ./repro          # build/run one binary directly
```

Override the compiler with `NVCC=...`. **Requires two P2P-capable GPUs** (the
reference environment is 2× Quadro RTX 6000 over NVLink). On hardware without two
P2P GPUs the binaries exit early with a setup error and prove nothing.

There is no test framework, linter, or CI. "Testing" means running a binary and
comparing its stdout and exit code against the captured reference in
`EXPECTED-OUTPUT.txt`. Exit codes are part of the contract:
`repro` exits **0 = failure reproduced**, **1 = not reproduced**, **2 = setup
error**; `workaround` exits **0 = both mitigations succeeded**.

## The four artifacts and how they relate

- `repro.cu` — proves everything *around* the failure works, so the failure is
  isolated to one API/context combination. Numbered cases [1]–[4]; only [4]
  (peer copy inside capture) fails. Prints a `RESULT:` verdict line.
- `workaround.cu` — the two capturable mitigations: **(A)** a peer-access kernel
  node, **(B)** `cudaMemcpyAsync` device-to-device over UVA peer pointers
  (recommended drop-in). Both capture, instantiate, replay, and produce the
  correct cross-device result.
- `matrix.cu` → `matrix-wsl2.txt` — the platform-neutral sweep: every peer-copy
  API × capture mode × peer-access state, emitting one table row each. Built to be
  run on a *second* platform and diffed; the native-Linux run is done and
  byte-for-byte identical (see below).
- `EXPECTED-OUTPUT.txt` — captured stdout of `repro` and `workaround` on the
  reference WSL2 box. Update this only when the source genuinely changes the
  output, and re-capture on real hardware — do not hand-edit.

## The central question — RESOLVED

Whether this is a **WSL-specific driver bug** or **general CUDA behaviour** was
the repo's one open question. It is now **settled: general CUDA behaviour, not a
WSL bug.** Running `matrix.cu` on native (non-WSL) Linux with the same 2× P2P GPUs
produces output **byte-for-byte identical** to `matrix-wsl2.txt` — the deciding
`cudaMemcpyPeerAsync`-inside-capture row reads `operation not permitted` on both.
It is the same byte-identical CUDA 13.3 (V13.3.33) userspace install under two
kernel/boot contexts, which pins the rejection to CUDA's runtime, independent of
the kernel/driver layer. See `NATIVE-TEST.md` and (closed) GitHub issue #1;
`matrix-native.txt` is gitignored.

Because of this, **do not reintroduce the "WSL driver bug" framing.** Earlier
drafts hedged ("looks like", "pending native-Linux confirmation"); that hedge is
now resolved in the *not-a-WSL-bug* direction — keep it that way unless new
contradicting evidence appears. The repo stands as a known-limitation + mitigation
reference; the capturable forms (D2D / Default-UVA / peer-access kernel) are the
supported path.
