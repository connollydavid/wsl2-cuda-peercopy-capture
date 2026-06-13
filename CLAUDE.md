# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A short technical note, with a runnable demonstration, on one CUDA stream-capture
detail: `cudaMemcpyPeerAsync` (and `cudaMemcpy3DPeerAsync`) cannot be captured
into a CUDA graph. Issued *inside* a stream capture it fails with `operation not
permitted when stream is capturing`, even though the same peer copy succeeds
outside capture and the semantically equivalent
`cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)` over peer-enabled UVA pointers
*is* capturable. First surfaced on WSL2 and confirmed identical on native Linux
(same byte-identical CUDA 13.3 install, two kernel/boot contexts), so it is
general CUDA behaviour, not a WSL bug. This is not an application; it is a note
plus evidence. Each `.cu` file demonstrates one facet, and the captured text
outputs are the artifact.

## Build & run

```sh
make            # builds demo, capturable, matrix
make run        # builds all, runs demo then capturable then matrix
make clean
nvcc -o demo demo.cu && ./demo             # build/run one binary directly
```

Override the compiler with `NVCC=...`. **Requires two P2P-capable GPUs** (the
reference environment is 2× Quadro RTX 6000 over NVLink). On hardware without two
P2P GPUs the binaries exit early with a setup error and prove nothing.

There is no test framework, linter, or CI. "Testing" means running a binary and
comparing its stdout and exit code against the captured reference in
`EXPECTED-OUTPUT.txt`. Exit codes are part of the contract:
`demo` exits **0 = behaves as the note describes**, **1 = does not**, **2 = setup
error**; `capturable` exits **0 = both capturable forms succeeded**.

## The artifacts and how they relate

- `demo.cu` — shows everything *around* the constraint works, so it is isolated to
  one API/context combination. Numbered cases [1]–[4]; only [4] (peer copy inside
  capture) fails. Prints a `RESULT:` line.
- `capturable.cu` — the two capturable forms: **(A)** a peer-access kernel node,
  **(B)** `cudaMemcpyAsync` device-to-device over UVA peer pointers (recommended
  drop-in). Both capture, instantiate, replay, and produce the correct
  cross-device result.
- `matrix.cu` → `matrix-wsl2.txt` — the platform-neutral sweep: every peer-copy
  API × capture mode × peer-access state, one table row each. Built to be run on a
  *second* platform and diffed; the native-Linux run is done and byte-for-byte
  identical (see below).
- `EXPECTED-OUTPUT.txt` — captured stdout of `demo` and `capturable` on the
  reference WSL2 box. Update this only when the source genuinely changes the
  output, and re-capture on real hardware — do not hand-edit.

## Framing — settled

This started as a suspected WSL driver bug. It is not. Running `matrix.cu` on
native (non-WSL) Linux with the same 2× P2P GPUs produces output byte-for-byte
identical to `matrix-wsl2.txt`: the `cudaMemcpyPeerAsync`-inside-capture row reads
`operation not permitted` on both. Same CUDA 13.3 (V13.3.33) userspace install
under two kernel/boot contexts, which pins the behaviour to CUDA's runtime,
independent of the kernel/driver layer. See `NATIVE-TEST.md` and (closed) GitHub
issue #1; `matrix-native.txt` is gitignored.

**Do not reintroduce "WSL driver bug" or "report to NVIDIA" framing**, and do not
claim the docs state peer memcpy is capturable (they do not; the Programming
Guide's prohibited-ops list does not mention peer memcpy, and its only "peer"
content concerns graph memory-pool peer accessibility). This is a technical note:
the capturable forms (D2D over UVA, or a peer-access kernel) are the supported way
to put a peer copy inside a graph.
