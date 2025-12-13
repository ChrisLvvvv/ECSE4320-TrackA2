#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

static void triad(double *a, const double *b, const double *c, double s, size_t n) {
    for (size_t i = 0; i < n; i++) a[i] = b[i] + s * c[i];
}

int main(int argc, char **argv) {
    // usage: stream_bw bytes iters
    // example: stream_bw 268435456 10   (256MB total array each)
    if (argc < 3) {
        fprintf(stderr, "usage: %s bytes iters\n", argv[0]);
        return 1;
    }

    size_t bytes = (size_t)atol(argv[1]);
    int iters = atoi(argv[2]);

    size_t n = bytes / sizeof(double);
    if (n < 1024) n = 1024;

    double *a = (double *)aligned_alloc(64, n * sizeof(double));
    double *b = (double *)aligned_alloc(64, n * sizeof(double));
    double *c = (double *)aligned_alloc(64, n * sizeof(double));
    if (!a || !b || !c) {
        fprintf(stderr, "alloc failed\n");
        free(a); free(b); free(c);
        return 1;
    }

    for (size_t i = 0; i < n; i++) {
        a[i] = 0.0;
        b[i] = 1.0;
        c[i] = 2.0;
    }

    // warmup
    triad(a, b, c, 3.0, n);

    double best = 1e100;
    for (int t = 0; t < iters; t++) {
        double t0 = now_sec();
        triad(a, b, c, 3.0, n);
        double t1 = now_sec();
        double dt = t1 - t0;
        if (dt < best) best = dt;
    }

    // triad bytes moved: read b + read c + write a = 24 bytes per element (double)
    double bytes_moved = (double)n * 24.0;
    double gbps = (bytes_moved / best) / 1e9;

    printf("kernel=stream_triad,bytes=%zu,n=%zu,iters=%d,time_s=%.6f,gbps=%.3f\n",
           bytes, n, iters, best, gbps);

    free(a); free(b); free(c);
    return 0;
}
