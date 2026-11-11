#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

// Matrix-Vector Multiplication: y[M] = A[M x N] * x[N]
// Uses float32.
void gemv_naive(const float* A, const float* x, float* y, int M, int N) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        const float* row = &A[i * N];
        for (int j = 0; j < N; j++) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

int main(int argc, char* argv[]) {
    // Default matrix dimensions (square N x N, capped at 8192 for BTP scope)
    std::vector<int> sizes = {1024, 2048, 4096, 8192};

    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; i++) {
            sizes.push_back(std::atoi(argv[i]));
        }
    }

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
        size_t vec_x_bytes = (size_t)N * sizeof(float);
        size_t vec_y_bytes = (size_t)M * sizeof(float);
        size_t total_bytes = mat_bytes + vec_x_bytes + vec_y_bytes;

        float* A = new float[(size_t)M * N];
        float* x = new float[N];
        float* y = new float[M];

        for (size_t i = 0; i < (size_t)M * N; i++) A[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < N; i++) x[i] = (float)(rand() % 100) / 100.0f;
        std::memset(y, 0, M * sizeof(float));

        // Warm up cache for small sizes
        if (N <= 2048) {
            gemv_naive(A, x, y, M, N);
        }

        // Run multiple iterations to get accurate timing if very fast
        int iterations = (N <= 2048) ? 10 : ((N <= 4096) ? 5 : 2);

        auto start = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iterations; it++) {
            gemv_naive(A, x, y, M, N);
        }
        auto end = std::chrono::high_resolution_clock::now();

        // Prevent optimization away
        volatile float sink = y[0] + y[M - 1];
        (void)sink;

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        double total_flops = 2.0 * M * N; // 1 mult + 1 add
        double gflops = (total_flops / 1e9) / (elapsed_ms / 1000.0);
        // Total data read/written per run = Matrix A (read once) + vector x (read) + vector y (written)
        double bw_gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);

        std::string mem_str;
        if (mat_bytes < 1024 * 1024) {
            mem_str = std::to_string(mat_bytes / 1024) + " KB";
        } else if (mat_bytes < 1024 * 1024 * 1024) {
            mem_str = std::to_string(mat_bytes / (1024 * 1024)) + " MB";
        } else {
            mem_str = std::to_string(mat_bytes / (1024 * 1024 * 1024)) + " GB";
        }

        std::cout << std::left
                  << std::setw(10) << N
                  << std::setw(18) << (long long)total_flops
                  << std::setw(18) << std::fixed << std::setprecision(3) << elapsed_ms
                  << std::setw(18) << std::fixed << std::setprecision(3) << gflops
                  << std::setw(18) << std::fixed << std::setprecision(2) << bw_gb_s
                  << std::setw(18) << mem_str
                  << std::endl;

        delete[] A;
        delete[] x;
        delete[] y;
    }

    return 0;
}
