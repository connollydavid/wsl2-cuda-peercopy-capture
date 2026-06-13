// Demonstrates what is and is not capturable when you want a peer-to-peer
// (cross-device) copy inside a CUDA graph.
//
// The takeaway: cudaMemcpyPeerAsync works fine on its own, and single-GPU
// stream capture works fine, but issuing cudaMemcpyPeerAsync *while a stream is
// capturing* returns "operation not permitted when stream is capturing". It is
// not capturable. This is CUDA runtime behaviour (identical on WSL2 and native
// Linux), not a platform bug. To put a peer copy inside a captured graph, use
// one of the capturable forms in capturable.cu.
//
// Build: nvcc -o demo demo.cu
// Run:   ./demo            (requires 2 P2P-capable GPUs)
//
// Exit status: 0 if the system behaves as this guide describes (peer copy ok
// outside capture, rejected inside), 1 if it does not, 2 on setup error.
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
    if (!can01 || !can10) { fprintf(stderr, "GPUs are not P2P-capable; this guide needs P2P\n"); return 2; }
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

    // [4] peer copy INSIDE capture -- the operation that is not capturable.
    // cudaMemcpyPeerAsync has no graph-node representation: graph memcpy nodes
    // are addressed by UVA pointer (cudaMemcpy3DParms), and there is no
    // explicit-device-id "peer memcpy node" for the runtime to record this call
    // as. So capture rejects it with "operation not permitted when stream is
    // capturing", which then poisons the capture and fails cudaStreamEndCapture.
    printf("\n[4] cudaMemcpyPeerAsync INSIDE capture:\n");
    cudaGraph_t g2 = nullptr;
    SHOW(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    cudaError_t in_err  = cudaMemcpyPeerAsync(d1, 1, a0, 0, 1024, s);
    printf("  cudaMemcpyPeerAsync                                      -> %s\n",
           cudaGetErrorString(in_err));
    cudaError_t end_err = cudaStreamEndCapture(s, &g2);
    printf("  cudaStreamEndCapture                                     -> %s\n",
           cudaGetErrorString(end_err));

    // Expected: peer copy ok outside capture, rejected inside.
    bool as_expected = (out_err == cudaSuccess) && (in_err != cudaSuccess);
    printf("\nRESULT: %s\n", as_expected
        ? "cudaMemcpyPeerAsync is NOT capturable (accepted outside capture, rejected inside). "
          "Use a capturable form for peer copies inside a graph -- see capturable.cu."
        : "this system does not match the guide (peer copy behaved unexpectedly)");
    return as_expected ? 0 : 1;
}
