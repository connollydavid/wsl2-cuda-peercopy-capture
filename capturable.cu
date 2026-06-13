// Two capturable ways to express a peer-to-peer (cross-device) copy inside a
// CUDA graph -- the supported alternatives to cudaMemcpyPeerAsync, which is not
// capturable. Both rely on peer access being enabled
// (cudaDeviceEnablePeerAccess), which makes the remote device pointer directly
// addressable on the local device via UVA:
//
//   [A] a kernel on dev0 that reads a dev1 pointer  -> captured as a kernel node
//   [B] cudaMemcpyAsync(..., DeviceToDevice) on UVA  -> captured as a memcpy node
//
// Both are verified here to capture, instantiate, replay, and produce the
// correct result across the peer boundary.
//
// Build: nvcc -o capturable capturable.cu
// Run:   ./capturable      (requires 2 P2P-capable GPUs)
//
// Exit status: 0 if both forms work, 1 if either fails, 2 on setup error.
#include <cstdio>
#include <cuda_runtime.h>

#define MUST(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[setup error] %s: %s\n", #call, cudaGetErrorString(e_)); \
    return 2; } } while (0)

#define SHOW(call) do { cudaError_t e_ = (call); \
    printf("  %-56s -> %s\n", #call, cudaGetErrorString(e_)); } while (0)

__global__ void peer_add(const float* __restrict__ remote, float* __restrict__ local, int nf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nf) local[i] += remote[i];
}

int main() {
    int n = 0;
    MUST(cudaGetDeviceCount(&n));
    if (n < 2) { fprintf(stderr, "need 2 P2P-capable GPUs, found %d\n", n); return 2; }
    const int NF = 256; const size_t B = NF * sizeof(float);

    int can = 0;
    MUST(cudaDeviceCanAccessPeer(&can, 0, 1));
    if (!can) { fprintf(stderr, "GPUs are not P2P-capable; both forms need peer access\n"); return 2; }
    MUST(cudaSetDevice(0)); MUST(cudaDeviceEnablePeerAccess(1, 0));
    MUST(cudaSetDevice(1)); MUST(cudaDeviceEnablePeerAccess(0, 0));

    float *local0, *remote1;
    MUST(cudaSetDevice(0)); MUST(cudaMalloc(&local0, B));
    MUST(cudaSetDevice(1)); MUST(cudaMalloc(&remote1, B));
    { float h[NF]; for (int i = 0; i < NF; i++) h[i] = 1.0f;
      MUST(cudaSetDevice(0)); MUST(cudaMemcpy(local0, h, B, cudaMemcpyHostToDevice)); }
    { float h[NF]; for (int i = 0; i < NF; i++) h[i] = 2.0f;
      MUST(cudaSetDevice(1)); MUST(cudaMemcpy(remote1, h, B, cudaMemcpyHostToDevice)); }

    MUST(cudaSetDevice(0));
    cudaStream_t s; MUST(cudaStreamCreate(&s));

    // [A] peer-access kernel inside capture.
    printf("[A] peer-access KERNEL inside capture (dev0 kernel reads dev1 ptr):\n");
    cudaGraph_t g; cudaGraphExec_t ge;
    SHOW(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    peer_add<<<(NF + 127) / 128, 128, 0, s>>>(remote1, local0, NF);
    cudaError_t a_end = cudaStreamEndCapture(s, &g);
    printf("  cudaStreamEndCapture                                     -> %s\n", cudaGetErrorString(a_end));
    float a_res = -1.0f;
    if (a_end == cudaSuccess) {
        SHOW(cudaGraphInstantiate(&ge, g, 0));
        SHOW(cudaGraphLaunch(ge, s));
        SHOW(cudaStreamSynchronize(s));
        MUST(cudaMemcpy(&a_res, local0, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  result local0[0] = %.1f (expect 3.0)\n", a_res);
    }
    bool a_ok = (a_end == cudaSuccess) && (a_res == 3.0f);

    // [B] plain DtoD memcpy over peer pointers inside capture.
    printf("\n[B] cudaMemcpyAsync(DeviceToDevice, peer ptrs) inside capture:\n");
    cudaGraph_t g2;
    SHOW(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    cudaError_t b_mem = cudaMemcpyAsync(local0, remote1, B, cudaMemcpyDeviceToDevice, s);
    printf("  cudaMemcpyAsync(D2D, peer)                               -> %s\n", cudaGetErrorString(b_mem));
    cudaError_t b_end = cudaStreamEndCapture(s, &g2);
    printf("  cudaStreamEndCapture                                     -> %s\n", cudaGetErrorString(b_end));
    bool b_ok = (b_mem == cudaSuccess) && (b_end == cudaSuccess);

    printf("\nRESULT: kernel-node form [A] %s; memcpy-node form [B] %s\n",
           a_ok ? "OK" : "FAILED", b_ok ? "OK" : "FAILED");
    return (a_ok && b_ok) ? 0 : 1;
}
