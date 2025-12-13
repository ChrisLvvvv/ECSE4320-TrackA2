#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

static void gemm_ref(const double *A, const double *B, double *C,
                     size_t m, size_t k, size_t n)
{
    for (size_t i = 0; i < m; ++i) {
        for (size_t p = 0; p < k; ++p) {
            double a = A[i*k + p];
            const double *Bp = &B[p*n];
            double *Ci = &C[i*n];
            for (size_t j = 0; j < n; ++j) {
                Ci[j] += a * Bp[j];
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s m k n seed\n", argv[0]);
        return 1;
    }

    size_t m = (size_t)atol(argv[1]);
    size_t k = (size_t)atol(argv[2]);
    size_t n = (size_t)atol(argv[3]);
    unsigned int seed = (unsigned int)atoi(argv[4]);

    double *A = alloc_dense(m, k);
    double *B = alloc_dense(k, n);
    double *C_ref = alloc_dense(m, n);
    double *C_blas = alloc_dense(m, n);

    if (!A || !B || !C_ref || !C_blas) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    random_dense(A, m*k, seed + 1);
    random_dense(B, k*n, seed + 2);
    zero_dense(C_ref, m*n);
    zero_dense(C_blas, m*n);

    gemm_ref(A, B, C_ref, m, k, n);

#if defined(HAVE_CBLAS)
    #include <cblas.h>
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)n, (int)k,
                1.0, A, (int)k,
                     B, (int)n,
                0.0, C_blas, (int)n);

    double rel = max_relative_error(C_blas, C_ref, m*n);
    printf("compare_blas: rel_err=%.3e (ours vs BLAS)\n", rel);
    int ok = (rel < 1e-6);
#else
    printf("compare_blas: OpenBLAS not enabled (compile with HAVE_CBLAS + link -lopenblas)\n");
    int ok = 1;
#endif

    free_dense(A);
    free_dense(B);
    free_dense(C_ref);
    free_dense(C_blas);
    return ok ? 0 : 2;
}
