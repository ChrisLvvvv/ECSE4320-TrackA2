#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

static double ai_dense(size_t m, size_t k, size_t n) {
    // FLOPs: 2*m*k*n
    // Bytes (rough lower bound, assumes perfect reuse is impossible; still ok for roofline context):
    // read A (m*k) + read B (k*n) + read+write C (2*m*n) doubles
    double flops = 2.0 * (double)m * (double)k * (double)n;
    double bytes = 8.0 * ( (double)m*k + (double)k*n + 2.0*(double)m*n );
    return flops / bytes;
}

static double ai_spmm_csr(size_t m, size_t k, size_t n, size_t nnz) {
    // FLOPs: 2*nnz*n
    // Bytes (conservative-ish):
    // CSR: values (8*nnz) + col_idx (4*nnz) + row_ptr (8*(m+1))
    // B read: each nnz touches a row of B length n -> assume B streamed: 8*nnz*n
    // C write: 8*m*n
    double flops = 2.0 * (double)nnz * (double)n;
    double bytes = 8.0*(double)nnz + 4.0*(double)nnz + 8.0*(double)(m+1)
                 + 8.0*(double)nnz*(double)n
                 + 8.0*(double)m*(double)n;
    return flops / bytes;
}

int main(int argc, char **argv) {
    // usage:
    //   roofline_points m k n density iters threads block seed
    if (argc < 9) {
        fprintf(stderr, "usage: %s m k n density iters threads block seed\n", argv[0]);
        return 1;
    }

    size_t m = (size_t)atol(argv[1]);
    size_t k = (size_t)atol(argv[2]);
    size_t n = (size_t)atol(argv[3]);
    double density = atof(argv[4]);
    int iters = atoi(argv[5]);
    int threads = atoi(argv[6]);
    int block = atoi(argv[7]);
    unsigned int seed = (unsigned int)atoi(argv[8]);

    // Run dense gemm
    // dense_gemm prints a kv line: kernel=dense_gemm,...,gflops=...,time_s=...
    // We'll just call it and let the outer script capture stdout.
    // Same for spmm; spmm prints nnz, gflops, time_s.

    // No-op. This program is only for AI formulas if you want them centralized,
    // but easiest is to compute AI in python. Keeping this file minimal is fine.
    (void)m; (void)k; (void)n; (void)density; (void)iters; (void)threads; (void)block; (void)seed;

    // Intentionally empty: AI is computed in python from logged m/k/n/nnz.
    return 0;
}
