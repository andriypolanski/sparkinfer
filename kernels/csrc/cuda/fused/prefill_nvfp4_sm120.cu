#include "sparkinfer/kernels/prefill_nvfp4.h"

#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/float_subbyte.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>

namespace sparkinfer::kernels {
namespace {
using namespace cute;
using E4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using BF = cutlass::bfloat16_t;
using Tile = Shape<_128, _128, _128>;
using Cluster = Shape<_1, _1, _1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    BF, cutlass::layout::RowMajor, 8, BF, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    E4, cutlass::layout::RowMajor, 32, E4, cutlass::layout::ColumnMajor, 32, float,
    Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, Mainloop, Epilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using StrideA = typename Kernel::StrideA;
using StrideB = typename Kernel::StrideB;
using StrideC = typename Kernel::StrideC;
using StrideD = typename Kernel::StrideD;
using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;

auto shape(int m, int n, int k) { return cute::make_shape(m, n, k, 1); }
auto sfa_layout(int m, int n, int k) { return ScaleConfig::tile_atom_to_shape_SFA(shape(m,n,k)); }
auto sfb_layout(int m, int n, int k) { return ScaleConfig::tile_atom_to_shape_SFB(shape(m,n,k)); }

template <class Layout>
__global__ void quant_rows(const __nv_bfloat16* src, unsigned char* dst,
                           cutlass::float_ue4m3_t* sf, int rows, int cols, Layout layout) {
    constexpr int V = 16;
    constexpr int WARPS = 8;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int total = rows * (cols / V);
    for (int vec = blockIdx.x * WARPS + warp; vec < total; vec += gridDim.x * WARPS) {
        int row = vec / (cols / V), k0 = (vec % (cols / V)) * V;
        float x = lane < V ? __bfloat162float(src[(size_t)row * cols + k0 + lane]) : 0.f;
        float a = fabsf(x);
        #pragma unroll
        for (int d = 16; d; d >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, d));
        cutlass::float_ue4m3_t qs(fmaxf(a * (1.f / 6.f), 0x1p-9f));
        cutlass::float_e2m1_t q(x / float(qs));
        unsigned nibble = q.raw() & 15u;
        unsigned hi = __shfl_down_sync(0xffffffffu, nibble, 1);
        if (lane < V && !(lane & 1))
            dst[((size_t)row * cols + k0 + lane) >> 1] = (unsigned char)(nibble | (hi << 4));
        if (lane == 0) {
            auto scales = cute::make_tensor(sf, layout);
            scales(row, k0, 0) = qs;
        }
    }
}

Gemm::Arguments args(const void* a, const void* sa, const void* b, const void* sb,
                     void* d, int m, int n, int k) {
    auto as = cutlass::make_cute_packed_stride(StrideA{}, {m,k,1});
    auto bs = cutlass::make_cute_packed_stride(StrideB{}, {n,k,1});
    auto cs = cutlass::make_cute_packed_stride(StrideC{}, {m,n,1});
    auto ds = cutlass::make_cute_packed_stride(StrideD{}, {m,n,1});
    return {cutlass::gemm::GemmUniversalMode::kGemm, shape(m,n,k),
            {static_cast<const cutlass::float_e2m1_t*>(a), as, static_cast<const cutlass::float_e2m1_t*>(b), bs, static_cast<const cutlass::float_ue4m3_t*>(sa), sfa_layout(m,n,k), static_cast<const cutlass::float_ue4m3_t*>(sb), sfb_layout(m,n,k)},
            {{1.f, 0.f}, static_cast<const BF*>(nullptr), cs, static_cast<BF*>(d), ds}};
}
} // namespace

bool prefill_nvfp4_supported(int m, int n, int k) {
    int dev=0, major=0, minor=0;
    return cudaGetDevice(&dev) == cudaSuccess &&
           cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
           cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
           major == 12 && minor == 0 && m > 0 && !(m & 7) && !(n & 127) && !(k & 127);
}
size_t prefill_nvfp4_data_bytes(int r, int c) { return ((size_t)r*c + 1)/2; }
size_t prefill_nvfp4_scale_bytes_a(int m, int k) {
    return (size_t)cute::size(cute::filter_zeros(sfa_layout(m,128,k)));
}
size_t prefill_nvfp4_scale_bytes_b(int n, int k) {
    return (size_t)cute::size(cute::filter_zeros(sfb_layout(128,n,k)));
}
size_t prefill_nvfp4_workspace_bytes(int m, int n, int k) {
    return Gemm::get_workspace_size(args(nullptr,nullptr,nullptr,nullptr,nullptr,m,n,k));
}
bool launch_prefill_nvfp4_quant_a(const void* s, void* d, void* sf, int m, int k, cudaStream_t st) {
    if (!s || !d || !sf || !prefill_nvfp4_supported(m,128,k)) return false;
    auto l = sfa_layout(m,128,k);
    int blocks = (m * (k / 16) + 7) / 8; if (blocks > 4096) blocks = 4096;
    quant_rows<<<blocks,256,0,st>>>((const __nv_bfloat16*)s,(unsigned char*)d,
                                     (cutlass::float_ue4m3_t*)sf,m,k,l);
    return cudaPeekAtLastError() == cudaSuccess;
}
bool launch_prefill_nvfp4_quant_b(const void* s, void* d, void* sf, int n, int k, cudaStream_t st) {
    if (!s || !d || !sf || !prefill_nvfp4_supported(128,n,k)) return false;
    auto l = sfb_layout(128,n,k);
    int blocks = (n * (k / 16) + 7) / 8; if (blocks > 4096) blocks = 4096;
    quant_rows<<<blocks,256,0,st>>>((const __nv_bfloat16*)s,(unsigned char*)d,
                                     (cutlass::float_ue4m3_t*)sf,n,k,l);
    return cudaPeekAtLastError() == cudaSuccess;
}
bool launch_prefill_nvfp4_gemm(const void* a,const void* sa,const void* b,const void* sb,
                               void* d,int m,int n,int k,void* ws,cudaStream_t st) {
    if (!a||!sa||!b||!sb||!d||!prefill_nvfp4_supported(m,n,k)) return false;
    Gemm gemm; auto ar = args(a,sa,b,sb,d,m,n,k);
    return gemm.can_implement(ar) == cutlass::Status::kSuccess &&
           gemm.initialize(ar,ws,st) == cutlass::Status::kSuccess &&
           gemm.run(st) == cutlass::Status::kSuccess;
}
} // namespace sparkinfer::kernels
