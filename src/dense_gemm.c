#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static void gemm_tiled_rowmajor(const double *A, const double *B, double *C,
                                size_t m, size_t k, size_t n, size_t bs)
{
    for (size_t ii = 0; ii < m; ii += bs) {
        size_t iimax = (ii + bs < m) ? (ii + bs) : m;
        for (size_t kk = 0; kk < k; kk += bs) {
            size_t kkmax = (kk + bs < k) ? (kk + bs) : k;
            for (size_t jj = 0; jj < n; jj += bs) {
                size_t jjmax = (jj + bs < n) ? (jj + bs) : n;

                for (size_t i = ii; i < iimax; ++i) {
                    for (size_t p = kk; p < kkmax; ++p) {
                        double a = A[i*k + p];
                        const double *Bp = &B[p*n + jj];
                        double *Ci = &C[i*n + jj];
                        #pragma omp simd
                        for (size_t j = jj; j < jjmax; ++j) {
                            Ci[j - jj] += a * Bp[j - jj];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    /* usage:
       dense_gemm m k n iters threads block seed
       block=0 means no tiling (uses block = n as a single tile)
    */
    if (argc < 8) {
        fprintf(stderr, "usage: %s m k n iters threads block seed\n", argv[0]);
        return 1;
    }

    size_t m = (size_t)atol(argv[1]);
    size_t k = (size_t)atol(argv[2]);
    size_t n = (size_t)atol(argv[3]);
    int iters = atoi(argv[4]);
    int threads = atoi(argv[5]);
    size_t block = (size_t)atol(argv[6]);
    unsigned int seed = (unsigned int)atoi(argv[7]);

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#else
    (void)threads;
#endif

    double *A = alloc_dense(m, k);
    double *B = alloc_dense(k, n);
    double *C = alloc_dense(m, n);
    if (!A || !B || !C) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    random_dense(A, m*k, seed + 1);
    random_dense(B, k*n, seed + 2);
    zero_dense(C, m*n);

    size_t bs = (block == 0) ? (size_t)128 : block;

    /* warmup */
    gemm_tiled_rowmajor(A, B, C, m, k, n, bs);
    zero_dense(C, m*n);

    double best = 1e100;
    for (int t = 0; t < iters; ++t) {
        zero_dense(C, m*n);
        double t0 = now_sec();

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        for (size_t ii = 0; ii < m; ii += bs) {
            size_t iimax = (ii + bs < m) ? (ii + bs) : m;
            for (size_t kk = 0; kk < k; kk += bs) {
                size_t kkmax = (kk + bs < k) ? (kk + bs) : k;
                for (size_t jj = 0; jj < n; jj += bs) {
                    size_t jjmax = (jj + bs < n) ? (jj + bs) : n;

                    for (size_t i = ii; i < iimax; ++i) {
                        for (size_t p = kk; p < kkmax; ++p) {
                            double a = A[i*k + p];
                            const double *Bp = &B[p*n + jj];
                            double *Ci = &C[i*n + jj];
                            #pragma omp simd
                            for (size_t j = jj; j < jjmax; ++j) {
                                Ci[j - jj] += a * Bp[j - jj];
                            }
                        }
                    }
                }
            }
        }
#else
        gemm_tiled_rowmajor(A, B, C, m, k, n, bs);
#endif

        double t1 = now_sec();
        double dt = t1 - t0;
        if (dt < best) best = dt;
    }

    double flops = 2.0 * (double)m * (double)k * (double)n;
    double gflops = (flops / best) / 1e9;

    /* CSV header-friendly single line */
    printf("kernel=dense_gemm,m=%zu,k=%zu,n=%zu,iters=%d,threads=%d,block=%zu,time_s=%.6f,gflops=%.3f\n",
           m, k, n, iters, threads, bs, best, gflops);

    free_dense(A);
    free_dense(B);
    free_dense(C);
    return 0;
}
