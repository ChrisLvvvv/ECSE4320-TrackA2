#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

/* dense reference GEMM: C = A*B, all row-major */
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

static void spmm_csr(const csr_matrix_t *A, const double *B, double *C, size_t n)
{
    for (size_t i = 0; i < A->rows; ++i) {
        double *Ci = &C[i*n];
        for (size_t idx = A->row_ptr[i]; idx < A->row_ptr[i+1]; ++idx) {
            int col = A->col_idx[idx];
            double val = A->values[idx];
            const double *Brow = &B[(size_t)col * n];
            for (size_t j = 0; j < n; ++j) {
                Ci[j] += val * Brow[j];
            }
        }
    }
}

/* expand CSR to dense (slow, only for small correctness tests) */
static void csr_to_dense(const csr_matrix_t *A, double *Ad)
{
    zero_dense(Ad, A->rows * A->cols);
    for (size_t i = 0; i < A->rows; ++i) {
        for (size_t idx = A->row_ptr[i]; idx < A->row_ptr[i+1]; ++idx) {
            int col = A->col_idx[idx];
            Ad[i*A->cols + (size_t)col] += A->values[idx];
        }
    }
}

int main(void)
{
    size_t m = 64, k = 80, n = 48;
    double density = 0.05;
    unsigned int seed = 123;

    csr_matrix_t *As = random_csr(m, k, density, seed);
    double *Ad = alloc_dense(m, k);
    double *B  = alloc_dense(k, n);
    double *C_ref = alloc_dense(m, n);
    double *C_sp  = alloc_dense(m, n);

    if (!As || !Ad || !B || !C_ref || !C_sp) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    csr_to_dense(As, Ad);
    random_dense(B, k*n, seed + 1);

    zero_dense(C_ref, m*n);
    zero_dense(C_sp,  m*n);

    gemm_ref(Ad, B, C_ref, m, k, n);
    spmm_csr(As, B, C_sp, n);

    double rel = max_relative_error(C_ref, C_sp, m*n);
    printf("correctness: rel_err=%.3e (tol 1e-6)\n", rel);

    free_csr(As);
    free_dense(Ad);
    free_dense(B);
    free_dense(C_ref);
    free_dense(C_sp);

    return (rel < 1e-6) ? 0 : 2;
}
