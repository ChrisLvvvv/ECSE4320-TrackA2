#!/usr/bin/env bash
set -euo pipefail

m="${1:-2048}"
k="${2:-2048}"
n="${3:-2048}"
iters="${4:-5}"
threads="${5:-1}"
block="${6:-128}"
seed="${7:-1}"

out="results/raw/density_m${m}_k${k}_n${n}_t${threads}.csv"
mkdir -p results/raw
: > "$out"

densities=(0.001 0.005 0.01 0.02 0.05 0.10 0.20 0.50)

./build/dense_gemm "$m" "$k" "$n" "$iters" "$threads" "$block" "$seed" >> "$out"

for d in "${densities[@]}"; do
  ./build/spmm_csr "$m" "$k" "$n" "$d" "$iters" "$threads" "$seed" >> "$out"
done

echo "wrote $out"
