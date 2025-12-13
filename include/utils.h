#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

double now_sec(void);

/* dense */
double *alloc_dense(size_t rows, size_t cols);
void free_dense(double *p);
void zero_dense(double *A, size_t n);
void random_dense(double *A, size_t n, unsigned int seed);

/* CSR */
typedef struct {
    size_t rows, cols;
    size_t nnz;
    size_t *row_ptr; /* len rows+1 */
    int    *col_idx; /* len nnz */
    double *values;  /* len nnz */
} csr_matrix_t;

csr_matrix_t *alloc_csr(size_t rows, size_t cols, size_t nnz);
void free_csr(csr_matrix_t *A);
csr_matrix_t *random_csr(size_t rows, size_t cols, double density, unsigned int seed);

/* validation */
double max_relative_error(const double *ref, const double *x, size_t n);

#endif
