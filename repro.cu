// Minimal reproducer: cudaMemcpyPeerAsync is rejected during CUDA stream
// capture on WSL2, even though peer access is enabled, the same copy succeeds
// outside capture, and single-GPU graph capture works. Only the *combination*
// (a peer copy issued while a stream is capturing) fails.
//
// Build: nvcc -o repro repro.cu
// Run:   ./repro            (requires 2 P2P-capable GPUs)
//
// Exit status: 0 if the failure reproduced, 1 if it did not, 2 on setup error.
#include <cstdio>
#include <cuda_runtime.h>

// Hard check for setup calls that MUST succeed; bail clearly if they don't.
#define MUST(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[setup error] %s: %s\n", #call, cudaGetErrorString(e_)); \
    return 2; } } while (0)

// Report-and-continue for the calls under test.
#define SHOW(call) do { cudaError_t e_ = (call); \
    printf("  %-56s -> %s\n", #call, cudaGetErrorString(e_)); } while (0)

int main() {
    int rt = 0, drv = 0, n = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    MUST(cudaGetDeviceCount(&n));
    printf("CUDA runtime %d.%d, driver reports %d.%d, %d device(s)\n\n",
           rt / 1000, (rt % 1000) / 10, drv / 1000, (drv % 1000) / 10, n);
    if (n < 2) { fprintf(stderr, "need 2 P2P-capable GPUs, found %d\n", n); return 2; }

    // [1] P2P is available and enable succeeds.
    printf("[1] peer access:\n");
    int can01 = 0, can10 = 0;
    MUST(cudaDeviceCanAccessPeer(&can01, 0, 1));
    MUST(cudaDeviceCanAccessPeer(&can10, 1, 0));
    printf("  canAccessPeer 0->1=%d  1->0=%d\n", can01, can10);
    if (!can01 || !can10) { fprintf(stderr, "GPUs are not P2P-capable; this repro needs P2P\n"); return 2; }
    MUST(cudaSetDevice(0)); SHOW(cudaDeviceEnablePeerAccess(1, 0));
    MUST(cudaSetDevice(1)); SHOW(cudaDeviceEnablePeerAccess(0, 0));

    MUST(cudaSetDevice(0));
    cudaStream_t s; MUST(cudaStreamCreate(&s));
    void *a0, *b0, *d1;
    MUST(cudaMalloc(&a0, 1024));
    MUST(cudaMalloc(&b0, 1024));
    MUST(cudaSetDevice(1)); MUST(cudaMalloc(&d1, 1024));
    MUST(cudaSetDevice(0));

    // [2] single-GPU stream capture round-trips fine.
    printf("\n[2] single-GPU stream capture (begin/end/instantiate/launch):\n");
    cudaGraph_t g; cudaGraphExec_t ge;
    SHOW(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    SHOW(cudaMemcpyAsync(b0, a0, 1024, cudaMemcpyDeviceToDevice, s));
    SHOW(cudaStreamEndCapture(s, &g));
    SHOW(cudaGraphInstantiate(&ge, g, 0));
    SHOW(cudaGraphLaunch(ge, s));
    SHOW(cudaStreamSynchronize(s));

    // [3] peer copy OUTSIDE capture succeeds.
    printf("\n[3] cudaMemcpyPeerAsync OUTSIDE capture:\n");
    cudaError_t out_err = cudaMemcpyPeerAsync(d1, 1, a0, 0, 1024, s);
    SHOW(cudaStreamSynchronize(s));
    printf("  cudaMemcpyPeerAsync                                      -> %s\n",
           cudaGetErrorString(out_err));

    // [4] peer copy INSIDE capture -- the failure under test.
    printf("\n[4] cudaMemcpyPeerAsync INSIDE capture:\n");
    cudaGraph_t g2 = nullptr;
    SHOW(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    cudaError_t in_err  = cudaMemcpyPeerAsync(d1, 1, a0, 0, 1024, s);
    printf("  cudaMemcpyPeerAsync                                      -> %s\n",
           cudaGetErrorString(in_err));
    cudaError_t end_err = cudaStreamEndCapture(s, &g2);
    printf("  cudaStreamEndCapture                                     -> %s\n",
           cudaGetErrorString(end_err));

    // Verdict: the bug is "peer copy ok outside capture, rejected inside".
    bool reproduced = (out_err == cudaSuccess) && (in_err != cudaSuccess);
    printf("\nRESULT: %s\n", reproduced
        ? "REPRODUCED -- cudaMemcpyPeerAsync rejected during capture, accepted outside"
        : "NOT REPRODUCED on this system");
    return reproduced ? 0 : 1;
}
