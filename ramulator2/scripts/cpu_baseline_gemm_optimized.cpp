#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <omp.h>

// Cache-blocked, OpenMP-parallelized GEMM with AVX2 inner kernel
// C[M x N] += A[M x K] * B[K x N], row-major float32
static void gemm_optimized(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int K, int N) {
    // Tile sizes tuned for L1/L2 cache (32KB L1, 256KB L2)
    constexpr int BM = 64, BK = 64, BN = 256;

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int ii = 0; ii < M; ii += BM) {
        for (int jj = 0; jj < N; jj += BN) {
            for (int kk = 0; kk < K; kk += BK) {
                int i_end = std::min(ii + BM, M);
                int k_end = std::min(kk + BK, K);
                int j_end = std::min(jj + BN, N);
                int j_vec_end = jj + ((j_end - jj) / 8) * 8; // AVX2: 8 floats

                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        __m256 a_val = _mm256_set1_ps(A[i * K + k]);
                        int j;
                        for (j = jj; j < j_vec_end; j += 8) {
                            __m256 b_val = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 c_val = _mm256_loadu_ps(&C[i * N + j]);
                            c_val = _mm256_fmadd_ps(a_val, b_val, c_val);
                            _mm256_storeu_ps(&C[i * N + j], c_val);
                        }
                        // Scalar tail
                        float a_s = A[i * K + k];
                        for (; j < j_end; j++) {
                            C[i * N + j] += a_s * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::vector<int> sizes = {256, 512, 1024, 2048};

    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; i++) {
            sizes.push_back(std::atoi(argv[i]));
        }
    }

    int num_threads = omp_get_max_threads();
    std::cout << "=== Optimized CPU GEMM Baseline ===" << std::endl;
    std::cout << "Compiler: -O3 -march=native -mavx2 -mfma -fopenmp" << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Blocking: 64x64x256, AVX2 FMA inner kernel" << std::endl;
    std::cout << std::endl;

    std::cout << std::left
              << std::setw(10) << "Size"
              << std::setw(18) << "Total FLOPs"
              << std::setw(18) << "Time (ms)"
              << std::setw(18) << "GFLOPS"
              << std::setw(18) << "Mem Footprint"
              << std::endl;
    std::cout << std::string(82, '-') << std::endl;

    for (int N : sizes) {
        int M = N, K = N;
        size_t mem_bytes = (size_t)(M * K + K * N + M * N) * sizeof(float);

        float* A = (float*)aligned_alloc(32, (size_t)M * K * sizeof(float));
        float* B = (float*)aligned_alloc(32, (size_t)K * N * sizeof(float));
        float* C = (float*)aligned_alloc(32, (size_t)M * N * sizeof(float));

        // Initialize
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (float)(rand() % 100) / 100.0f;
        for (size_t i = 0; i < (size_t)K * N; i++) B[i] = (float)(rand() % 100) / 100.0f;
        memset(C, 0, (size_t)M * N * sizeof(float));

        // Warm up
        if (N <= 512) {
            gemm_optimized(A, B, C, M, K, N);
            memset(C, 0, (size_t)M * N * sizeof(float));
        }

        // Run (multiple iterations for small sizes)
        int iterations = (N <= 256) ? 5 : ((N <= 512) ? 3 : 1);

        auto start = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; it++) {
            memset(C, 0, (size_t)M * N * sizeof(float));
            gemm_optimized(A, B, C, M, K, N);
        }
        auto end = std::chrono::high_resolution_clock::now();

        volatile float sink = C[0] + C[M * N - 1];
        (void)sink;

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        double total_flops = 2.0 * M * K * N;
        double gflops = (total_flops / 1e9) / (elapsed_ms / 1000.0);

        std::string mem_str;
        if (mem_bytes < 1024 * 1024)
            mem_str = std::to_string(mem_bytes / 1024) + " KB";
        else
            mem_str = std::to_string(mem_bytes / (1024 * 1024)) + " MB";

        std::cout << std::left
                  << std::setw(10) << N
                  << std::setw(18) << (long long)(total_flops)
                  << std::setw(18) << std::fixed << std::setprecision(2) << elapsed_ms
                  << std::setw(18) << std::fixed << std::setprecision(3) << gflops
                  << std::setw(18) << mem_str
                  << std::endl;

        free(A);
        free(B);
        free(C);
    }

    return 0;
}
