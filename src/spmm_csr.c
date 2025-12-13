#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static void spmm_csr_rowmajor(const csr_matrix_t *A, const double *B, double *C,
                              size_t ncols_B)
{
    size_t m = A->rows;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < m; ++i) {
        double *Ci = &C[i * ncols_B];
        size_t start = A->row_ptr[i];
        size_t end   = A->row_ptr[i + 1];

        for (size_t idx = start; idx < end; ++idx) {
            int col = A->col_idx[idx];
            double val = A->values[idx];

            const double *Brow = &B[(size_t)col * ncols_B];
            #pragma omp simd
            for (size_t j = 0; j < ncols_B; ++j) {
                Ci[j] += val * Brow[j];
            }
        }
    }
}

int main(int argc, char **argv)
{
    /* usage:
       spmm_csr m k n density iters threads seed
    */
    if (argc < 8) {
        fprintf(stderr, "usage: %s m k n density iters threads seed\n", argv[0]);
        return 1;
    }

    size_t m = (size_t)atol(argv[1]);
    size_t k = (size_t)atol(argv[2]);
    size_t n = (size_t)atol(argv[3]);
    double density = atof(argv[4]);
    int iters = atoi(argv[5]);
    int threads = atoi(argv[6]);
    unsigned int seed = (unsigned int)atoi(argv[7]);

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#else
    (void)threads;
#endif

    csr_matrix_t *A = random_csr(m, k, density, seed + 1);
    double *B = alloc_dense(k, n);
    double *C = alloc_dense(m, n);

    if (!A || !B || !C) {
        fprintf(stderr, "alloc failed\n");
        free_csr(A);
        free_dense(B);
        free_dense(C);
        return 1;
    }

    random_dense(B, k * n, seed + 2);

    /* warmup */
    zero_dense(C, m * n);
    spmm_csr_rowmajor(A, B, C, n);

    double best = 1e100;
    for (int t = 0; t < iters; ++t) {
        zero_dense(C, m * n);
        double t0 = now_sec();
        spmm_csr_rowmajor(A, B, C, n);
        double t1 = now_sec();
        double dt = t1 - t0;
        if (dt < best) best = dt;
    }

    /* FLOPs ~ 2 * nnz * n */
    double flops = 2.0 * (double)A->nnz * (double)n;
    double gflops = (flops / best) / 1e9;

    double sec_per_nnz = best / ((A->nnz > 0) ? (double)A->nnz : 1.0);

    printf("kernel=spmm_csr,m=%zu,k=%zu,n=%zu,density=%.6f,nnz=%zu,iters=%d,threads=%d,time_s=%.6f,gflops=%.3f,sec_per_nnz=%.9e\n",
           m, k, n, density, A->nnz, iters, threads, best, gflops, sec_per_nnz);

    free_csr(A);
    free_dense(B);
    free_dense(C);
    return 0;
}