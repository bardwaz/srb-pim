#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <omp.h>

// Optimized GEMV: y[M] = A[M x N] * x[N]
// OpenMP parallelized across rows, AVX2 FMA inner loop
static void gemv_optimized(const float* __restrict__ A,
                           const float* __restrict__ x,
                           float* __restrict__ y,
                           int M, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        const float* row = &A[i * N];
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        int j = 0;
        int j_end = (N / 32) * 32;

        // Unrolled 4x8 = 32 floats per iteration
        for (; j < j_end; j += 32) {
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(&row[j]),    _mm256_loadu_ps(&x[j]),    sum0);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(&row[j+8]),  _mm256_loadu_ps(&x[j+8]),  sum1);
            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(&row[j+16]), _mm256_loadu_ps(&x[j+16]), sum2);
            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(&row[j+24]), _mm256_loadu_ps(&x[j+24]), sum3);
        }

        // Reduce 4 accumulators
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);

        // Handle remaining 8-wide chunks
        for (; j + 8 <= N; j += 8) {
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(&row[j]), _mm256_loadu_ps(&x[j]), sum0);
        }

        // Horizontal sum of sum0
        __m128 lo = _mm256_castps256_ps128(sum0);
        __m128 hi = _mm256_extractf128_ps(sum0, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float result = _mm_cvtss_f32(lo);

        // Scalar tail
        for (; j < N; j++) {
            result += row[j] * x[j];
        }
        y[i] = result;
    }
}

int main(int argc, char* argv[]) {
    std::vector<int> sizes = {1024, 2048, 4096, 8192};

    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; i++) {
            sizes.push_back(std::atoi(argv[i]));
        }
    }

    int num_threads = omp_get_max_threads();
    std::cout << "=== Optimized CPU GEMV Baseline ===" << std::endl;
    std::cout << "Compiler: -O3 -march=native -mavx2 -mfma -fopenmp" << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "4x unrolled AVX2 FMA inner kernel" << std::endl;
    std::cout << std::endl;

    std::cout << std::left
              << std::setw(10) << "Size (N)"
              << std::setw(18) << "Total FLOPs"
              << std::setw(18) << "Time (ms)"
              << std::setw(18) << "GFLOPS"
              << std::setw(18) << "Bandwidth (GB/s)"
              << std::setw(18) << "Matrix Size"
              << std::endl;
    std::cout << std::string(98, '-') << std::endl;

    for (int N : sizes) {
        int M = N;
        size_t mat_bytes = (size_t)M * N * sizeof(float);
        size_t total_bytes = mat_bytes + (size_t)N * sizeof(float) + (size_t)M * sizeof(float);

        float* A = (float*)aligned_alloc(32, mat_bytes);
        float* x = (float*)aligned_alloc(32, (size_t)N * sizeof(float));
        float* y = (float*)aligned_alloc(32, (size_t)M * sizeof(float));

        for (size_t i = 0; i < (size_t)M * N; i++) A[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < N; i++) x[i] = (float)(rand() % 100) / 100.0f;
        memset(y, 0, M * sizeof(float));

        // Warm up
        gemv_optimized(A, x, y, M, N);

        // Multiple iterations for stable timing
        int iterations = (N <= 2048) ? 50 : ((N <= 4096) ? 20 : 5);

        auto start = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; it++) {
            gemv_optimized(A, x, y, M, N);
        }
        auto end = std::chrono::high_resolution_clock::now();

        volatile float sink = y[0] + y[M - 1];
        (void)sink;

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        double total_flops = 2.0 * M * N;
        double gflops = (total_flops / 1e9) / (elapsed_ms / 1000.0);
        double bw_gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);

        std::string mem_str;
        if (mat_bytes < 1024 * 1024)
            mem_str = std::to_string(mat_bytes / 1024) + " KB";
        else if (mat_bytes < (size_t)1024 * 1024 * 1024)
            mem_str = std::to_string(mat_bytes / (1024 * 1024)) + " MB";
        else
            mem_str = std::to_string(mat_bytes / ((size_t)1024 * 1024 * 1024)) + " GB";

        std::cout << std::left
                  << std::setw(10) << N
                  << std::setw(18) << (long long)total_flops
                  << std::setw(18) << std::fixed << std::setprecision(3) << elapsed_ms
                  << std::setw(18) << std::fixed << std::setprecision(3) << gflops
                  << std::setw(18) << std::fixed << std::setprecision(2) << bw_gb_s
                  << std::setw(18) << mem_str
                  << std::endl;

        free(A);
        free(x);
        free(y);
    }

    return 0;
}
