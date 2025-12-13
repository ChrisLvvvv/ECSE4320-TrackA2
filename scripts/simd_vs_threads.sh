#!/usr/bin/env bash
set -euo pipefail

m="${1:-1024}"
k="${2:-1024}"
n="${3:-1024}"
density="${4:-0.05}"
iters="${5:-5}"
block="${6:-128}"
seed="${7:-1}"
threads="${8:-10}"

mkdir -p results/raw
out="results/raw/simd_vs_threads_m${m}_k${k}_n${n}_d${density}.csv"
: > "$out"

export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Ensure binaries exist
make variants >/dev/null

echo "# OMP_DYNAMIC=${OMP_DYNAMIC}, OMP_PROC_BIND=${OMP_PROC_BIND}, OMP_PLACES=${OMP_PLACES}" >> "$out"
echo "# threads_target=${threads}" >> "$out"

# --- Dense GEMM ---
# scalar baseline (no vectorize), 1 thread
./build/dense_gemm_novec "$m" "$k" "$n" "$iters" 1 "$block" "$seed" | sed 's/^/variant=novec,/' >> "$out"

# threads only (no vectorize), T threads
./build/dense_gemm_novec "$m" "$k" "$n" "$iters" "$threads" "$block" "$seed" | sed 's/^/variant=novec,/' >> "$out"

# SIMD + threads (vectorized), T threads
./build/dense_gemm_vec "$m" "$k" "$n" "$iters" "$threads" "$block" "$seed" | sed 's/^/variant=vec,/' >> "$out"

# --- CSR SpMM ---
./build/spmm_csr_novec "$m" "$k" "$n" "$density" "$iters" 1 "$seed" | sed 's/^/variant=novec,/' >> "$out"
./build/spmm_csr_novec "$m" "$k" "$n" "$density" "$iters" "$threads" "$seed" | sed 's/^/variant=novec,/' >> "$out"
./build/spmm_csr_vec "$m" "$k" "$n" "$density" "$iters" "$threads" "$seed" | sed 's/^/variant=vec,/' >> "$out"

echo "wrote $out"
