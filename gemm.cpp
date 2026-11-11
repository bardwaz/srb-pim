#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

void blocked_gemm(int N, int B, const float* A, const float* B_mat, float* C) {
    for (int i = 0; i < N; i += B) {
        for (int j = 0; j < N; j += B) {
            for (int k = 0; k < N; k += B) {
                for (int ii = i; ii < min(i + B, N); ++ii) {
                    for (int kk = k; kk < min(k + B, N); ++kk) {
                        float a_val = A[ii * N + kk];
                        for (int jj = j; jj < min(j + B, N); ++jj) {
                            C[ii * N + jj] += a_val * B_mat[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);
    int B = 64; // Standard Cache Block size

    cout << "Allocating matrices of size " << N << "x" << N << "..." << endl;
    float* A = new float[N * N];
    float* B_mat = new float[N * N];
    float* C = new float[N * N]();

    cout << "Computing Blocked GEMM..." << endl;
    blocked_gemm(N, B, A, B_mat, C);

    // Prevent compiler from optimizing away the computation
    cout << "Computation complete. C[0] = " << C[0] << endl;

    delete[] A;
    delete[] B_mat;
    delete[] C;
    return 0;
}