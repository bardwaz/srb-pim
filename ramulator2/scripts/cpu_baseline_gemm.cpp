#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

// Simple row-major GEMM: C[M x N] += A[M x K] * B[K x N]
// Uses float32 to match typical PIM MAC precision.
void gemm_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // Default sizes to sweep
    std::vector<int> sizes = {64, 128, 256, 512, 1024, 2048};

    // Allow override from command line: ./cpu_baseline_gemm 1024 2048 4096
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; i++) {
            sizes.push_back(std::atoi(argv[i]));
        }
    }

    std::cout << std::left
              << std::setw(10) << "Size"
              << std::setw(18) << "Total MACs"
              << std::setw(18) << "Time (ms)"
              << std::setw(18) << "GFLOPS"
              << std::setw(18) << "Mem Footprint"
              << std::endl;
    std::cout << std::string(82, '-') << std::endl;

    for (int N : sizes) {
        int M = N, K = N;
        size_t mem_bytes = (size_t)(M * K + K * N + M * N) * sizeof(float);

        // Allocate and initialize
        float* A = new float[M * K];
        float* B = new float[K * N];
        float* C = new float[M * N];

        // Initialize with small random values
        for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 100) / 100.0f;
        std::memset(C, 0, M * N * sizeof(float));

        // Warm up cache for small sizes
        if (N <= 256) {
            gemm_naive(A, B, C, M, K, N);
            std::memset(C, 0, M * N * sizeof(float));
        }

        // Timed run
        auto start = std::chrono::high_resolution_clock::now();
        gemm_naive(A, B, C, M, K, N);
        auto end = std::chrono::high_resolution_clock::now();

        // Prevent dead-code elimination
        volatile float sink = C[0] + C[M * N - 1];
        (void)sink;

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double total_macs = 2.0 * M * K * N;  // 2 ops per MAC (multiply + add)
        double gflops = (total_macs / 1e9) / (elapsed_ms / 1000.0);

        std::string mem_str;
        if (mem_bytes < 1024 * 1024) {
            mem_str = std::to_string(mem_bytes / 1024) + " KB";
        } else {
            mem_str = std::to_string(mem_bytes / (1024 * 1024)) + " MB";
        }

        std::cout << std::left
                  << std::setw(10) << N
                  << std::setw(18) << (long long)(total_macs)
                  << std::setw(18) << std::fixed << std::setprecision(2) << elapsed_ms
                  << std::setw(18) << std::fixed << std::setprecision(3) << gflops
                  << std::setw(18) << mem_str
                  << std::endl;

        delete[] A;
        delete[] B;
        delete[] C;
    }

    return 0;
}
