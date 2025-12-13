#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

/* timing */
double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* dense matrix helpers */
double *alloc_dense(size_t rows, size_t cols)
{
    return (double *)aligned_alloc(64, rows * cols * sizeof(double));
}

void random_dense(double *A, size_t rows, size_t cols, unsigned int seed)
{
    srand(seed);
    for (size_t i = 0; i < rows * cols; ++i) {
        A[i] = (double)rand() / RAND_MAX;
    }
}

/* CSR helpers */
csr_matrix_t *alloc_csr(size_t rows, size_t cols, size_t nnz)
{
    csr_matrix_t *A = malloc(sizeof(csr_matrix_t));
    A->rows = rows;
    A->cols = cols;
    A->nnz  = nnz;
    A->row_ptr = malloc((rows + 1) * sizeof(size_t));
    A->col_idx = malloc(nnz * sizeof(size_t));
    A->values  = malloc(nnz * sizeof(double));
    return A;
}

void free_csr(csr_matrix_t *A)
{
    if (!A) return;
    free(A->row_ptr);
    free(A->col_idx);
    free(A->values);
    free(A);
}

csr_matrix_t *random_csr(size_t rows, size_t cols, double density, unsigned int seed)
{
    srand(seed);

    size_t max_nnz = (size_t)(rows * cols * density);
    if (max_nnz == 0) max_nnz = 1;

    csr_matrix_t *A = alloc_csr(rows, cols, max_nnz);

    size_t nnz = 0;
    A->row_ptr[0] = 0;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double r = (double)rand() / RAND_MAX;
            if (r < density) {
                if (nnz < max_nnz) {
                    A->col_idx[nnz] = j;
                    A->values[nnz]  = (double)rand() / RAND_MAX;
                    nnz++;
                }
            }
        }
        A->row_ptr[i + 1] = nnz;
    }

    A->nnz = nnz;
    return A;
}

/* validation */
double max_relative_error(const double *A, const double *B, size_t n)
{
    double max_err = 0.0;
    double max_ref = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double diff = fabs(A[i] - B[i]);
        if (diff > max_err) max_err = diff;
        if (fabs(B[i]) > max_ref) max_ref = fabs(B[i]);
    }

    if (max_ref == 0.0) return max_err;
    return max_err / max_ref;
}
