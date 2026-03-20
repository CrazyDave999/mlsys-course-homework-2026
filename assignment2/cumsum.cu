#include <cuda_runtime.h>
#include <cstdint>

// Each warp (32 threads) processes one row independently.
// Within each chunk of 32 elements, a warp-level inclusive scan is performed
// using shuffle intrinsics, then the running total from previous chunks is added.
extern "C" __global__ void cumsum_2d_kernel(
    const int64_t* __restrict__ d_A,
    int64_t* __restrict__ d_Out,
    int64_t m,
    int64_t n)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = global_tid >> 5;
    int lane = global_tid & 31;

    if (row >= m) return;

    int64_t row_off = (int64_t)row * n;
    int64_t running = 0;
    int num_chunks = (int)((n + 31) >> 5);

    for (int c = 0; c < num_chunks; c++) {
        int64_t col = (int64_t)c * 32 + lane;
        int64_t val = (col < n) ? d_A[row_off + col] : 0;

        // Warp inclusive scan (Hillis-Steele)
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            int64_t t = __shfl_up_sync(0xFFFFFFFF, val, off);
            if (lane >= off) val += t;
        }

        val += running;
        if (col < n) d_Out[row_off + col] = val;
        running = __shfl_sync(0xFFFFFFFF, val, 31);
    }
}
