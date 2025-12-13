#!/usr/bin/env bash
set -euo pipefail

# Working-set / cache transition sweep
# Output: results/raw/cache_transition.csv

sizes=(256 512 768 1024 1536 2048 3072 4096)

density="${DENSITY:-0.05}"
iters="${ITERS:-5}"
threads="${THREADS:-10}"
block="${BLOCK:-128}"
seed="${SEED:-1}"

mkdir -p results/raw
out="results/raw/cache_transition.csv"
: > "$out"

# Make runs less noisy (OpenMP pinning)
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "# sizes=${sizes[*]}" >> "$out"
echo "# density=${density}, iters=${iters}, threads=${threads}, block=${block}, seed=${seed}" >> "$out"
echo "# OMP_DYNAMIC=${OMP_DYNAMIC}, OMP_PROC_BIND=${OMP_PROC_BIND}, OMP_PLACES=${OMP_PLACES}" >> "$out"

for s in "${sizes[@]}"; do
  # Dense GEMM (vec build)
  ./build/dense_gemm "$s" "$s" "$s" "$iters" "$threads" "$block" "$seed" >> "$out"

  # CSR SpMM (vec build), A is sparse (s x s), B is dense (s x s), C is dense (s x s)
  ./build/spmm_csr  "$s" "$s" "$s" "$density" "$iters" "$threads" "$seed" >> "$out"
done

echo "wrote ${out}"
