# Native-Linux confirmation (the dispositive test)

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
  Report it (see below).
- **Native = `operation not permitted`** → the prohibition is **general CUDA
  behaviour**, not WSL-specific. Not a WSL bug. The repo stands as a
  known-limitation + mitigation reference; no NVIDIA report warranted (or, at
  most, a docs-consistency note).

WSL2 baseline for the deciding row (all three capture modes):

```
cudaMemcpyPeerAsync   peer=1   enqueue=operation not permitted when stream is capturing   end=ERR
```
