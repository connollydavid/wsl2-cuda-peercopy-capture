// Comprehensive peer-copy-during-capture matrix. Run on WSL2 and native Linux;
// diff the two output tables. Requires 2 P2P-capable GPUs.
#include <cstdio>
#include <cuda_runtime.h>

#define NF 256
#define B  (NF*sizeof(float))

static cudaStream_t s;
static float *dst0, *src1;     // dst0 on dev0, src1 on dev1

static void seed() {            // dst0=0 on dev0, src1=7 on dev1
  float z[NF], v[NF];
  for (int i=0;i<NF;i++){z[i]=0;v[i]=7;}
  cudaSetDevice(0); cudaMemcpy(dst0,z,B,cudaMemcpyHostToDevice);
  cudaSetDevice(1); cudaMemcpy(src1,v,B,cudaMemcpyHostToDevice);
  cudaSetDevice(0);
}
static bool check7(){ float h[NF]; cudaMemcpy(h,dst0,B,cudaMemcpyDeviceToHost); return h[0]==7.0f && h[NF-1]==7.0f; }

__global__ void kcopy(const float* r, float* l, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) l[i]=r[i];}

// op==0 peerAsync, 1 memcpy3DPeer, 2 D2D, 3 Default(UVA), 4 kernel
static cudaError_t enqueue(int op){
  switch(op){
    case 0: return cudaMemcpyPeerAsync(dst0,0,src1,1,B,s);
    case 1: { cudaMemcpy3DPeerParms p={}; p.dstDevice=0; p.srcDevice=1;
              p.dstPtr=make_cudaPitchedPtr(dst0,B,B,1); p.srcPtr=make_cudaPitchedPtr(src1,B,B,1);
              p.extent=make_cudaExtent(B,1,1); return cudaMemcpy3DPeerAsync(&p,s); }
    case 2: return cudaMemcpyAsync(dst0,src1,B,cudaMemcpyDeviceToDevice,s);
    case 3: return cudaMemcpyAsync(dst0,src1,B,cudaMemcpyDefault,s);
    case 4: kcopy<<<(NF+127)/128,128,0,s>>>(src1,dst0,NF); return cudaGetLastError();
  }
  return cudaSuccess;
}

static void run(const char* name, int op, cudaStreamCaptureMode mode, bool peer){
  // (re)set peer access state
  cudaSetDevice(0); cudaDeviceDisablePeerAccess(1); cudaGetLastError();
  cudaSetDevice(1); cudaDeviceDisablePeerAccess(0); cudaGetLastError();
  if(peer){ cudaSetDevice(0); cudaDeviceEnablePeerAccess(1,0); cudaSetDevice(1); cudaDeviceEnablePeerAccess(0,0); }
  cudaSetDevice(0); seed();
  cudaGraph_t g=nullptr; cudaGraphExec_t ge=nullptr;
  cudaError_t be=cudaStreamBeginCapture(s,mode);
  cudaError_t oe=enqueue(op);
  cudaError_t ee=cudaStreamEndCapture(s,&g);
  const char* corr="-";
  if(ee==cudaSuccess && g){
    cudaError_t ie=cudaGraphInstantiate(&ge,g,0);
    if(ie==cudaSuccess){ cudaGraphLaunch(ge,s); cudaStreamSynchronize(s); corr=check7()?"OK":"WRONG"; cudaGraphExecDestroy(ge);}
    else corr="inst-fail";
  }
  if(g) cudaGraphDestroy(g);
  cudaGetLastError();
  printf("  %-34s peer=%d begin=%-8s enqueue=%-46s end=%-8s replay=%s\n",
    name, peer, (be?"ERR":"ok"), cudaGetErrorString(oe), (ee?"ERR":"ok"), corr);
}

int main(){
  int n=0,rt=0,dv=0; cudaGetDeviceCount(&n); cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&dv);
  cudaDeviceProp prop={}; if(n>0) cudaGetDeviceProperties(&prop,0);
  printf("=== ENV: %d x %s | CUDA rt %d.%d drv %d.%d ===\n", n, prop.name, rt/1000,(rt%1000)/10, dv/1000,(dv%1000)/10);
  if(n<2){printf("need 2 GPUs\n");return 2;}
  cudaSetDevice(0); cudaMalloc(&dst0,B); cudaSetDevice(1); cudaMalloc(&src1,B); cudaSetDevice(0); cudaStreamCreate(&s);

  printf("\n-- peer copy INSIDE capture, mode sweep (peer access ON) --\n");
  run("cudaMemcpyPeerAsync",      0, cudaStreamCaptureModeThreadLocal, true);
  run("cudaMemcpyPeerAsync",      0, cudaStreamCaptureModeGlobal,      true);
  run("cudaMemcpyPeerAsync",      0, cudaStreamCaptureModeRelaxed,     true);
  run("cudaMemcpy3DPeerAsync",    1, cudaStreamCaptureModeRelaxed,     true);

  printf("\n-- capturable alternatives INSIDE capture (Relaxed) --\n");
  run("cudaMemcpyAsync D2D",       2, cudaStreamCaptureModeRelaxed,    true);
  run("cudaMemcpyAsync Default",   3, cudaStreamCaptureModeRelaxed,    true);
  run("peer-access kernel",        4, cudaStreamCaptureModeRelaxed,    true);

  printf("\n-- peer access DISABLED (does D2D still capture? correct?) --\n");
  run("cudaMemcpyAsync D2D",       2, cudaStreamCaptureModeRelaxed,    false);
  run("cudaMemcpyPeerAsync",       0, cudaStreamCaptureModeRelaxed,    false);

  printf("\n-- baselines --\n");
  // peer copy OUTSIDE capture
  cudaSetDevice(0); cudaDeviceEnablePeerAccess(1,0); seed();
  cudaError_t o=cudaMemcpyPeerAsync(dst0,0,src1,1,B,s); cudaStreamSynchronize(s);
  printf("  %-34s            enqueue=%-46s replay=%s\n","cudaMemcpyPeerAsync OUTSIDE",cudaGetErrorString(o),check7()?"OK":"WRONG");
  return 0;
}
