# ECSE 4320 – Project A2  
## Dense vs Sparse Matrix Multiplication  
SIMD, Multithreading, and Roofline Analysis

---

## Overview

This project studies the performance differences between dense matrix multiplication (Dense GEMM) and sparse matrix–dense matrix multiplication (SpMM) using the Compressed Sparse Row (CSR) format. Custom C kernels were implemented with OpenMP multithreading and compiler-driven SIMD vectorization to evaluate how arithmetic intensity, memory behavior, and parallelism interact across dense and sparse workloads.

A Roofline model is used to classify each kernel as compute-bound or memory-bandwidth-bound. The experiments identify the density break-even points and reuse conditions under which compressed sparse formats are beneficial, as well as the limits of SIMD and multithreading for memory-dominated kernels.

---

## Environment

- OS: Ubuntu 22.04 (WSL2)
- Architecture: x86-64
- CPU: 1 socket, 10 cores, 20 hardware threads
- Compiler: GCC
- Flags: `-O3 -march=native -fopenmp`
- Parallelism: OpenMP
- SIMD: compiler auto-vectorization
- Analysis: Python (pandas, matplotlib)

CPU frequency scaling and background services were left at system defaults. Each experiment was repeated multiple times with fixed random seeds, and best-of runtimes are reported.

---

## Implementations

### Dense GEMM

A dense matrix multiplication kernel (`A × B → C`) was implemented in C. The loop structure is cache-aware and parallelized across rows using OpenMP. SIMD vectorization is enabled through compiler optimization flags. No external BLAS library is used for performance measurements; BLAS is only referenced for correctness validation.

### CSR SpMM

Sparse matrix multiplication (`CSR A × Dense B → Dense C`) was implemented using the CSR format. The kernel iterates over nonzeros in each row and accumulates partial dot products into dense output rows. Parallelism is applied at the row level. The inner loops are structured to allow limited SIMD where possible, though indirect memory accesses restrict vector efficiency.

### Dense-to-CSR Conversion

A two-pass dense-to-CSR conversion routine was implemented to quantify sparse format setup cost. The conversion scans the dense matrix to count nonzeros per row and then populates CSR index and value arrays.

---

## Experimental Methodology

The following experimental axes were explored:

- Matrix sizes: square matrices with `m = k = n`, spanning cache-resident and memory-resident regimes
- Sparsity: uniform random density from 0.1% to 50%
- Thread count: 1 up to the number of physical cores
- Execution modes:
  - Scalar (single-thread)
  - Multithreaded without SIMD
  - Multithreaded with SIMD
- Metrics:
  - Runtime
  - GFLOP/s
  - Speedup relative to scalar baseline
  - Conversion cost vs execution cost

All raw results and generated figures are included in the repository for reproducibility.

---

## Results and Analysis

### 1. Density Break-even

For fixed matrix dimensions (`1024 × 1024`), runtime was measured while sweeping sparsity. Dense GEMM runtime remains nearly constant across densities due to its regular access pattern and fixed arithmetic workload. In contrast, CSR SpMM runtime scales approximately linearly with the number of nonzeros.

A break-even point is observed at approximately **0.1% density**. Below this threshold, CSR SpMM outperforms dense GEMM; above it, dense computation is faster. The extremely low break-even density reflects the large arithmetic intensity gap between dense and sparse kernels.

---

### 2. Thread Scaling

Both kernels benefit from multithreading, but their scaling behavior differs. Dense GEMM shows near-linear scaling up to the physical core count, indicating effective utilization of compute resources.

CSR SpMM scales less efficiently and saturates earlier. This behavior is attributed to irregular memory access patterns and increased pressure on the memory subsystem, which limit scalability before compute resources are fully utilized.

---

### 3. SIMD vs Multithreading

To isolate performance contributions, scalar, multithreaded-scalar, and multithreaded-SIMD configurations were compared.

For dense GEMM, SIMD is the dominant source of speedup, providing a substantial improvement over the scalar baseline by fully utilizing wide vector units. This confirms that dense GEMM operates in a compute-bound regime.

For CSR SpMM, SIMD provides only modest gains. Indirect memory accesses through column index arrays prevent efficient vector loads, making performance improvements primarily dependent on thread-level parallelism rather than data-level parallelism.

---

### 4. Working Set and Cache Transitions

Matrix size was swept to observe cache effects. Dense GEMM sustains high GFLOP/s while the working set fits in cache and degrades as data spills into LLC and DRAM. CSR SpMM exhibits an earlier and sharper performance drop.

Due to its low arithmetic intensity, sparse computation cannot amortize memory access costs and is highly sensitive to cache-to-memory transitions.

---

### 5. Dense-to-CSR Conversion Cost

Dense-to-CSR conversion time is largely independent of sparsity, as it requires scanning the entire dense matrix. In contrast, CSR SpMM runtime increases with the number of nonzeros.

At 0.1% density, the conversion cost is equivalent to approximately **15 CSR SpMM executions**. At 1–2% density, the conversion cost is comparable to a single SpMM. Above 5–10% density, conversion overhead becomes negligible relative to computation.

This indicates that sparse formats are most beneficial when the same sparse matrix is reused multiple times. For one-off sparse operations, conversion overhead can outweigh execution speedup.

---

### 6. Roofline Analysis

A Roofline model was constructed using measured memory bandwidth (**22.8 GB/s**) and estimated peak compute throughput (**118 GFLOP/s**).

Dense GEMM operates at high arithmetic intensity and approaches the compute roof, confirming compute-bound behavior. CSR SpMM operates at low arithmetic intensity (below approximately 0.25 FLOP/byte) and aligns with the bandwidth roof, confirming memory-bound behavior.

The achieved performance points align with Roofline predictions and explain the observed limits of SIMD and multithreading for sparse computation.

---

## Discussion

Dense GEMM and CSR SpMM occupy fundamentally different performance regimes. Dense GEMM is compute-bound and benefits strongly from SIMD and multithreading, justifying aggressive tiling and vectorization. CSR SpMM is memory-bound; its performance is dominated by data movement, cache behavior, and metadata overhead.

As a result, sparse optimization should focus on improving data locality and reuse rather than increasing peak FLOP throughput. Dense optimization, in contrast, benefits directly from wider SIMD units and additional cores.

---

## Limitations

- Experiments were conducted on a single hardware platform.
- No explicit NUMA control or hardware performance counters were used.
- SIMD relied on compiler auto-vectorization rather than hand-written intrinsics.

---

## Reproducibility

All source code, scripts, raw data, and figures are included in the repository. The main results can be reproduced using the following Makefile targets:
make test # correctness
make density # break-even analysis
make threads # thread scaling
make simd # SIMD vs threads
make cache # working-set transitions
make convert_cost # dense-to-CSR conversion cost
make roofline # roofline model


---

## Summary

This project demonstrates how arithmetic intensity and memory behavior determine the effectiveness of SIMD, multithreading, and sparse representations. Dense GEMM remains a compute-dominated workload, while CSR SpMM is fundamentally limited by memory bandwidth. Understanding these distinctions is critical when selecting data formats and optimization strategies in scientific computing and machine learning workloads.
