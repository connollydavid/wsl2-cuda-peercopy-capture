// Two capture-compatible ways to perform the same peer transfer that
// cudaMemcpyPeerAsync cannot do during capture on WSL2. Both rely on peer
// access being enabled (cudaDeviceEnablePeerAccess), which makes the remote
// device pointer directly addressable via UVA on the local device.
//
//   [A] a kernel on dev0 that reads a dev1 pointer  -> captured as a kernel node
//   [B] cudaMemcpyAsync(..., DeviceToDevice) on UVA  -> captured as a memcpy node
//
// On native Linux cudaMemcpyPeerAsync itself captures fine; only WSL2 rejects
// it, while accepting [B] -- the semantically identical transfer.
//
// Build: nvcc -o workaround workaround.cu
// Run:   ./workaround       (requires 2 P2P-capable GPUs)
#include <cstdio>
#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e = (call); \
  printf("  %-58s -> %s\n", #call, cudaGetErrorString(e)); } while(0)

__global__ void peer_add(const float* __restrict__ remote, float* __restrict__ local, int nf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nf) local[i] += remote[i];
}

int main() {
  int n = 0; cudaGetDeviceCount(&n);
  if (n < 2) { printf("need 2 GPUs\n"); return 0; }
  const int NF = 256; const size_t B = NF * sizeof(float);

  cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0);
  cudaSetDevice(1); cudaDeviceEnablePeerAccess(0, 0);

  float *local0, *remote1;
  cudaSetDevice(0); cudaMalloc(&local0, B);
  cudaSetDevice(1); cudaMalloc(&remote1, B);
  cudaSetDevice(0); { float h[NF]; for (int i=0;i<NF;i++) h[i]=1.0f; cudaMemcpy(local0,h,B,cudaMemcpyHostToDevice);}
  cudaSetDevice(1); { float h[NF]; for (int i=0;i<NF;i++) h[i]=2.0f; cudaMemcpy(remote1,h,B,cudaMemcpyHostToDevice);}

  cudaSetDevice(0);
  cudaStream_t s; cudaStreamCreate(&s);

  printf("[A] peer-access KERNEL inside capture (dev0 kernel reads dev1 ptr):\n");
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  peer_add<<<(NF+127)/128, 128, 0, s>>>(remote1, local0, NF);
  cudaError_t ec = cudaStreamEndCapture(s, &g);
  printf("  cudaStreamEndCapture                                       -> %s\n", cudaGetErrorString(ec));
  if (ec == cudaSuccess) {
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    float h[NF]; cudaMemcpy(h, local0, B, cudaMemcpyDeviceToHost);
    printf("  result local0[0] = %.1f (expect 3.0)\n", h[0]);
  }

  printf("\n[B] cudaMemcpyAsync(DeviceToDevice, peer ptrs) inside capture:\n");
  cudaGraph_t g2;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  cudaError_t em = cudaMemcpyAsync(local0, remote1, B, cudaMemcpyDeviceToDevice, s);
  printf("  cudaMemcpyAsync(D2D, peer)                                 -> %s\n", cudaGetErrorString(em));
  cudaError_t ec2 = cudaStreamEndCapture(s, &g2);
  printf("  cudaStreamEndCapture                                       -> %s\n", cudaGetErrorString(ec2));
  return 0;
}
