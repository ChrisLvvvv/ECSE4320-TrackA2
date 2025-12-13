#!/usr/bin/env bash
set -euo pipefail

m="${1:-2048}"
k="${2:-2048}"
n="${3:-2048}"
density="${4:-0.05}"
iters="${5:-5}"
block="${6:-128}"
seed="${7:-1}"

out="results/raw/threads_m${m}_k${k}_n${n}_d${density}.csv"
mkdir -p results/raw
: > "$out"

threads_list=(1 2 4 8)

for t in "${threads_list[@]}"; do
  ./build/dense_gemm "$m" "$k" "$n" "$iters" "$t" "$block" "$seed" >> "$out"
  ./build/spmm_csr "$m" "$k" "$n" "$density" "$iters" "$t" "$seed" >> "$out"
done

echo "wrote $out"
