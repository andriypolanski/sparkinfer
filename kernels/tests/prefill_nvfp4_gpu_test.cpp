#include "sparkinfer/kernels/prefill_nvfp4.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    using namespace sparkinfer::kernels;
    constexpr int M=128, N=128, K=128;
    if (!prefill_nvfp4_supported(M,N,K)) return 77;
    std::vector<__nv_bfloat16> hA(M*K, __float2bfloat16(1.f));
    std::vector<__nv_bfloat16> hB(N*K, __float2bfloat16(1.f)), hD(M*N);
    void *A0=nullptr,*B0=nullptr,*A=nullptr,*B=nullptr,*SA=nullptr,*SB=nullptr,*D=nullptr,*W=nullptr;
    cudaMalloc(&A0,hA.size()*2); cudaMalloc(&B0,hB.size()*2); cudaMalloc(&D,hD.size()*2);
    cudaMalloc(&A,prefill_nvfp4_data_bytes(M,K)); cudaMalloc(&B,prefill_nvfp4_data_bytes(N,K));
    cudaMalloc(&SA,prefill_nvfp4_scale_bytes_a(M,K)); cudaMalloc(&SB,prefill_nvfp4_scale_bytes_b(N,K));
    size_t ws=prefill_nvfp4_workspace_bytes(M,N,K); if(ws) cudaMalloc(&W,ws);
    cudaMemcpy(A0,hA.data(),hA.size()*2,cudaMemcpyHostToDevice);
    cudaMemcpy(B0,hB.data(),hB.size()*2,cudaMemcpyHostToDevice);
    bool ok=launch_prefill_nvfp4_quant_a(A0,A,SA,M,K) &&
            launch_prefill_nvfp4_quant_b(B0,B,SB,N,K) &&
            launch_prefill_nvfp4_gemm(A,SA,B,SB,D,M,N,K,W) &&
            cudaDeviceSynchronize()==cudaSuccess;
    cudaMemcpy(hD.data(),D,hD.size()*2,cudaMemcpyDeviceToHost);
    float lo=1e9f,hi=-1e9f;
    for(auto v:hD){ float x=__bfloat162float(v); lo=fminf(lo,x); hi=fmaxf(hi,x); ok &= std::isfinite(x) && x>115.f && x<141.f; }
    std::printf("prefill_nvfp4_gpu_test: %s range=[%.3f,%.3f]\n",ok?"OK":"FAIL",lo,hi);
    cudaFree(A0);cudaFree(B0);cudaFree(A);cudaFree(B);cudaFree(SA);cudaFree(SB);cudaFree(D);if(W)cudaFree(W);
    return ok?0:1;
}
