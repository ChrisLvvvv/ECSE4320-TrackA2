#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

/* timing */
double now_sec(void);

/* dense matrix */
double *alloc_dense(size_t rows, size_t cols);
void random_dense(double *A, size_t rows, size_t cols, unsigned int seed);

/* CSR matrix */
typedef struct {
    size_t rows;
    size_t cols;
    size_t nnz;
    size_t *row_ptr;
    size_t *col_idx;
    double *values;
} csr_matrix_t;

csr_matrix_t *alloc_csr(size_t rows, size_t cols, size_t nnz);
void free_csr(csr_matrix_t *A);
csr_matrix_t *random_csr(size_t rows, size_t cols, double density, unsigned int seed);

/* validation */
double max_relative_error(const double *A, const double *B, size_t n);

#endif
