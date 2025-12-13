#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>   // <-- for UINT_MAX

/*
  Dense -> CSR conversion benchmark.

  usage:
    convert_csr m k density iters seed

  Measures conversion time for generating a dense A (with zeros) then converting to CSR.
  Outputs best time and p95 time over iters.
*/

static double urand(unsigned int *state) {
    /* simple LCG */
    *state = (*state * 1664525u) + 1013904223u;
    return (double)(*state) / (double)UINT_MAX;
}

static double frand_signed(unsigned int *state) {
    return 2.0 * urand(state) - 1.0;
}

static void gen_dense_with_density(double *A, size_t m, size_t k, double density, unsigned int seed) {
    unsigned int st = seed;
    size_t mk = m * k;
    for (size_t idx = 0; idx < mk; ++idx) {
        double r = urand(&st);
        if (r < density) {
            A[idx] = frand_signed(&st);
        } else {
            A[idx] = 0.0;
        }
    }
}

static csr_matrix_t *dense_to_csr(const double *A, size_t m, size_t k) {
    csr_matrix_t *M = (csr_matrix_t *)calloc(1, sizeof(csr_matrix_t));
    if (!M) return NULL;
    M->rows = m;
    M->cols = k;

    M->row_ptr = (size_t *)calloc(m + 1, sizeof(size_t));
    if (!M->row_ptr) { free(M); return NULL; }

    /* first pass: count nnz per row */
    size_t nnz = 0;
    for (size_t i = 0; i < m; ++i) {
        size_t row_nnz = 0;
        const double *Ai = &A[i * k];
        for (size_t j = 0; j < k; ++j) {
            if (Ai[j] != 0.0) row_nnz++;
        }
        nnz += row_nnz;
        M->row_ptr[i + 1] = nnz;
    }
    M->nnz = nnz;

    M->col_idx = (int *)malloc(nnz * sizeof(int));
    M->values  = (double *)malloc(nnz * sizeof(double));
    if (!M->col_idx || !M->values) {
        free(M->row_ptr);
        free(M->col_idx);
        free(M->values);
        free(M);
        return NULL;
    }

    /* second pass: fill */
    size_t write_pos = 0;
    for (size_t i = 0; i < m; ++i) {
        const double *Ai = &A[i * k];
        for (size_t j = 0; j < k; ++j) {
            double v = Ai[j];
            if (v != 0.0) {
                M->col_idx[write_pos] = (int)j;
                M->values[write_pos]  = v;
                write_pos++;
            }
        }
    }

    return M;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s m k density iters seed\n", argv[0]);
        return 1;
    }

    size_t m = (size_t)atol(argv[1]);
    size_t k = (size_t)atol(argv[2]);
    double density = atof(argv[3]);
    int iters = atoi(argv[4]);
    unsigned int seed = (unsigned int)atoi(argv[5]);

    double *A = alloc_dense(m, k);
    if (!A) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    double *times = (double *)malloc((size_t)iters * sizeof(double));
    if (!times) {
        fprintf(stderr, "alloc failed\n");
        free_dense(A);
        return 1;
    }

    double best = 1e100;
    size_t last_nnz = 0;

    for (int t = 0; t < iters; ++t) {
        gen_dense_with_density(A, m, k, density, seed + 17u * (unsigned int)t);

        double t0 = now_sec();
        csr_matrix_t *C = dense_to_csr(A, m, k);
        double t1 = now_sec();

        if (!C) {
            fprintf(stderr, "conversion failed\n");
            free(times);
            free_dense(A);
            return 1;
        }

        double dt = t1 - t0;
        times[t] = dt;
        if (dt < best) best = dt;

        last_nnz = C->nnz;
        free_csr(C);
    }

    qsort(times, (size_t)iters, sizeof(double), cmp_double);
    int p95_idx = (int)ceil(0.95 * iters) - 1;
    if (p95_idx < 0) p95_idx = 0;
    if (p95_idx >= iters) p95_idx = iters - 1;
    double p95 = times[p95_idx];

    printf("kernel=convert_csr,m=%zu,k=%zu,density=%.6f,nnz=%zu,iters=%d,time_s=%.6f,p95_s=%.6f\n",
           m, k, density, last_nnz, iters, best, p95);

    free(times);
    free_dense(A);
    return 0;
}
