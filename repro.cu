// Minimal reproducer: cudaMemcpyPeerAsync is rejected during CUDA stream
// capture on WSL2, even though P2P is enabled and the copy succeeds outside
// capture. Single-GPU capture and peer copies in isolation both work; only
// the *combination* (peer copy issued while a stream is capturing) fails.
//
// Build: nvcc -o repro repro.cu
// Run:   ./repro            (requires 2 P2P-capable GPUs)
#include <cstdio>
#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e = (call); \
  printf("  %-55s -> %s\n", #call, cudaGetErrorString(e)); } while(0)

#define CKq(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  printf("    [err] %s: %s\n", #call, cudaGetErrorString(e)); } } while(0)

int main() {
  int n = 0;
  cudaGetDeviceCount(&n);
  printf("device count: %d\n\n", n);

  // 1. P2P capability
  printf("[1] P2P canAccessPeer matrix:\n");
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (i != j) {
    int can = -1; cudaDeviceCanAccessPeer(&can, i, j);
    printf("  %d->%d canAccessPeer=%d\n", i, j, can);
  }
  printf("\n[1b] cudaDeviceEnablePeerAccess:\n");
  if (n >= 2) { cudaSetDevice(0); CK(cudaDeviceEnablePeerAccess(1, 0)); }

  // 2. Single-GPU stream capture round-trip
  printf("\n[2] single-GPU stream capture (instantiate + launch):\n");
  cudaSetDevice(0);
  cudaStream_t s; cudaStreamCreate(&s);
  float *a, *b; cudaMalloc(&a, 1024); cudaMalloc(&b, 1024);
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  CKq(cudaMemcpyAsync(b, a, 1024, cudaMemcpyDeviceToDevice, s));
  CK(cudaStreamEndCapture(s, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  CK(cudaGraphLaunch(ge, s));
  CK(cudaStreamSynchronize(s));

  // 3. Peer copy OUTSIDE capture
  printf("\n[3] cudaMemcpyPeerAsync OUTSIDE capture:\n");
  if (n >= 2) {
    float *d1; cudaSetDevice(1); cudaMalloc(&d1, 1024);
    cudaSetDevice(0);
    CK(cudaMemcpyPeerAsync(d1, 1, a, 0, 1024, s));
    CK(cudaStreamSynchronize(s));
  }

  // 4. Peer copy INSIDE capture -- the failure
  printf("\n[4] cudaMemcpyPeerAsync INSIDE capture:\n");
  if (n >= 2) {
    float *d1; cudaSetDevice(1); cudaMalloc(&d1, 1024);
    cudaSetDevice(0);
    cudaGraph_t g2;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    CKq(cudaMemcpyPeerAsync(d1, 1, a, 0, 1024, s));
    cudaError_t ec = cudaStreamEndCapture(s, &g2);
    printf("  cudaStreamEndCapture -> %s\n", cudaGetErrorString(ec));
  }
  return 0;
}
