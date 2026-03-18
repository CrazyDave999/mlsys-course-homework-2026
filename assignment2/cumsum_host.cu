#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdint>

// ============================================================================
// CUDA Error Checking Macro
// ============================================================================
#define checkCudaErrors(call)                                                  \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// Kernel Declarations (from cumsum.cu)
// ============================================================================

// TODO: Add the function prototypes for the CUDA kernels you implement in `cumsum.cu`.
// The `extern "C"` keyword is necessary to ensure the C++ compiler can link them correctly.
//
// Example:
// extern "C" __global__ void my_custom_kernel(...);


// ============================================================================
// CPU Reference Implementation for Verification
// ============================================================================
void cpu_2d_cumsum(int64_t* in, int64_t* out, int64_t m, int64_t n) {
    for (int64_t row = 0; row < m; ++row) {
        int64_t cumsum = 0;
        for (int64_t col = 0; col < n; ++col) {
            cumsum += in[row * n + col];
            out[row * n + col] = cumsum;
        }
    }
}

// ============================================================================
// Verification Function
// ============================================================================
bool verify_results(int64_t* gpu_out, int64_t* cpu_out, int64_t m, int64_t n) {
    bool success = true;
    int64_t num_errors = 0;
    const int64_t max_errors_to_print = 10;

    for (int64_t i = 0; i < m * n; ++i) {
        if (gpu_out[i] != cpu_out[i]) {
            if (num_errors < max_errors_to_print) {
                int64_t row = i / n;
                int64_t col = i % n;
                std::cerr << "Mismatch at [" << row << "," << col << "]: "
                          << "GPU = " << gpu_out[i] << ", CPU = " << cpu_out[i]
                          << std::endl;
            }
            num_errors++;
            success = false;
        }
    }

    if (num_errors > max_errors_to_print) {
        std::cerr << "... and " << (num_errors - max_errors_to_print)
                  << " more errors (total: " << num_errors << ")" << std::endl;
    }
    return success;
}

// ============================================================================
// GPU Kernel Orchestration Function
// ============================================================================
/**
 * @brief Orchestrates the execution of your 2D cumsum algorithm.
 *
 * You will launch the CUDA kernels you wrote in `cumsum.cu` from within this function.
 * The goal is to correctly execute your algorithm to compute the cumulative sum of the
 * input matrix `d_A` and store the final result in `d_Out`.
 *
 * @param d_A The input matrix on the device (m x n).
 * @param d_Out The output matrix on the device (m x n).
 * @param d_Tmp A temporary buffer on the device. You can use this for any
 * intermediate computations, such as storing block sums.
 * @param m The number of rows in the matrix.
 * @param n The number of columns in the matrix.
 *
 * @task
 * 1. Define your thread block and grid dimensions (`dim3`).
 * 2. Launch your kernel(s) in the correct order using the `<<<...>>>` syntax.
 * 3. Ensure synchronization between kernel launches if necessary (e.g., with `cudaDeviceSynchronize`).
 */
void launch_2d_cumsum_kernels(int64_t* d_A, int64_t* d_Out, int64_t* d_Tmp,
                               int64_t m, int64_t n)
{
    // TODO: Implement the logic to launch your CUDA kernels here.
    // 1. Define grid and block dimensions.
    // 2. Launch the kernel(s) you implemented in cumsum.cu.
    // 3. Synchronize if needed.
}

// ============================================================================
// Main Program (DO NOT MODIFY)
// ============================================================================
int main(int argc, char** argv) {
    // Configuration Parameters
    int64_t m = 1024;
    int64_t n = 1024;
    int numRuns = 100;

    if (argc >= 3) {
        m = std::atoll(argv[1]);
        n = std::atoll(argv[2]);
    }
    if (argc >= 4) {
        numRuns = std::atoi(argv[3]);
    }

    std::cout << "============================================================" << std::endl;
    std::cout << "2D Continuous Cumsum Benchmark" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Matrix dimensions: " << m << " x " << n << std::endl;
    std::cout << "Total elements: " << (m * n) << std::endl;
    std::cout << "Benchmark iterations: " << numRuns << std::endl;
    std::cout << "============================================================" << std::endl;

    // Host Memory Allocation
    size_t matrix_size = m * n * sizeof(int64_t);
    int64_t* h_A = new int64_t[m * n];
    int64_t* h_Out_gpu = new int64_t[m * n];
    int64_t* h_Out_cpu = new int64_t[m * n];
    for (int64_t i = 0; i < m * n; ++i) h_A[i] = 1;
    std::cout << "Initialized input matrix " << std::endl;

    // Device Memory Allocation
    int64_t* d_A = nullptr;
    int64_t* d_Out = nullptr;
    int64_t* d_Tmp = nullptr;
    checkCudaErrors(cudaMalloc(&d_A, matrix_size));
    checkCudaErrors(cudaMalloc(&d_Out, matrix_size));
    size_t tmp_size = m * n * 2 * sizeof(int64_t);
    checkCudaErrors(cudaMalloc(&d_Tmp, tmp_size));
    std::cout << "Allocated device memory..." << std::endl;

    // Data Transfer: Host to Device
    checkCudaErrors(cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice));
    std::cout << "Transferred input data to device" << std::endl;

    // Warm-up Run
    std::cout << "\nPerforming warm-up run..." << std::endl;
    checkCudaErrors(cudaMemset(d_Out, 0, matrix_size));
    checkCudaErrors(cudaMemset(d_Tmp, 0, tmp_size));
    launch_2d_cumsum_kernels(d_A, d_Out, d_Tmp, m, n);
    std::cout << "Warm-up complete" << std::endl;

    // Performance Benchmarking
    std::cout << "\nStarting benchmark (" << numRuns << " iterations)..." << std::endl;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float total_time_ms = 0.0f;

    for (int run = 0; run < numRuns; ++run) {
        checkCudaErrors(cudaMemset(d_Out, 0, matrix_size));
        checkCudaErrors(cudaMemset(d_Tmp, 0, tmp_size));
        checkCudaErrors(cudaEventRecord(start));
        launch_2d_cumsum_kernels(d_A, d_Out, d_Tmp, m, n);
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&elapsed_ms, start, stop));
        total_time_ms += elapsed_ms;
    }

    // Performance Metrics Calculation
    float avg_time_ms = total_time_ms / numRuns;
    float avg_time_s = avg_time_ms / 1000.0f;
    double total_elements = static_cast<double>(m) * static_cast<double>(n);
    double throughput_gelem_s = total_elements / avg_time_s / 1e9;
    double bytes_transferred = 2.0 * total_elements * sizeof(int64_t);
    double bandwidth_gb_s = bytes_transferred / avg_time_s / 1e9;

    std::cout << "\n============================================================" << std::endl;
    std::cout << "BENCHMARK RESULTS" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Average Algorithm Latency: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Effective Throughput: " << throughput_gelem_s << " G-Elements/s" << std::endl;
    std::cout << "Effective Memory Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << "============================================================" << std::endl;

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // Verification
    std::cout << "\nVerifying results..." << std::endl;
    checkCudaErrors(cudaMemcpy(h_Out_gpu, d_Out, matrix_size, cudaMemcpyDeviceToHost));
    std::cout << "Computing CPU reference..." << std::endl;
    cpu_2d_cumsum(h_A, h_Out_cpu, m, n);
    bool verification_passed = verify_results(h_Out_gpu, h_Out_cpu, m, n);

    std::cout << "\n============================================================" << std::endl;
    if (verification_passed) {
        std::cout << "VERIFICATION: SUCCESS ✓" << std::endl;
    } else {
        std::cout << "VERIFICATION: FAILED ✗" << std::endl;
    }
    std::cout << "============================================================" << std::endl;

    // Cleanup
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_Out));
    checkCudaErrors(cudaFree(d_Tmp));
    delete[] h_A;
    delete[] h_Out_gpu;
    delete[] h_Out_cpu;
    std::cout << "\nCleanup complete. Exiting." << std::endl;

    return verification_passed ? 0 : 1;
}