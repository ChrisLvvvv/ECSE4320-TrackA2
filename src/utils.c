#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* dense */
double *alloc_dense(size_t rows, size_t cols)
{
    size_t bytes = rows * cols * sizeof(double);
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return (double *)p;
}

void free_dense(double *p)
{
    free(p);
}

void zero_dense(double *A, size_t n)
{
    memset(A, 0, n * sizeof(double));
}

void random_dense(double *A, size_t n, unsigned int seed)
{
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        A[i] = (double)rand() / (double)RAND_MAX;
    }
}

/* CSR */
csr_matrix_t *alloc_csr(size_t rows, size_t cols, size_t nnz)
{
    csr_matrix_t *A = (csr_matrix_t *)malloc(sizeof(csr_matrix_t));
    if (!A) return NULL;
    A->rows = rows;
    A->cols = cols;
    A->nnz = nnz;

    A->row_ptr = (size_t *)malloc((rows + 1) * sizeof(size_t));
    A->col_idx = (int *)malloc(nnz * sizeof(int));
    A->values  = (double *)malloc(nnz * sizeof(double));

    if (!A->row_ptr || !A->col_idx || !A->values) {
        free_csr(A);
        return NULL;
    }
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

/* uniform random sparsity; allows duplicates within a row (fine for now) */
csr_matrix_t *random_csr(size_t rows, size_t cols, double density, unsigned int seed)
{
    if (density < 0.0) density = 0.0;
    if (density > 1.0) density = 1.0;

    srand(seed);

    size_t cap = (size_t)((double)rows * (double)cols * density);
    if (cap < 1) cap = 1;

    csr_matrix_t *A = alloc_csr(rows, cols, cap);
    if (!A) return NULL;

    size_t nnz = 0;
    A->row_ptr[0] = 0;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double r = (double)rand() / (double)RAND_MAX;
            if (r < density) {
                if (nnz < cap) {
                    A->col_idx[nnz] = (int)j;
                    A->values[nnz]  = (double)rand() / (double)RAND_MAX;
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
double max_relative_error(const double *ref, const double *x, size_t n)
{
    double max_err = 0.0;
    double max_ref = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double diff = fabs(ref[i] - x[i]);
        if (diff > max_err) max_err = diff;
        double ar = fabs(ref[i]);
        if (ar > max_ref) max_ref = ar;
    }

    if (max_ref == 0.0) return max_err;
    return max_err / max_ref;
}
