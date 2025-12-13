#!/usr/bin/env bash
set -euo pipefail

m="${1:-1024}"
k="${2:-1024}"
n="${3:-1024}"
threads="${4:-10}"
iters="${5:-5}"
block="${6:-128}"
seed="${7:-1}"

densities=(0.001 0.002 0.005 0.01 0.02 0.05 0.10 0.20 0.50)

mkdir -p results/raw
out="results/raw/convert_m${m}_k${k}_n${n}_t${threads}.csv"
: > "$out"

export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "# m=${m},k=${k},n=${n},threads=${threads},iters=${iters},seed=${seed}" >> "$out"
echo "# densities=${densities[*]}" >> "$out"
echo "# OMP_DYNAMIC=${OMP_DYNAMIC}, OMP_PROC_BIND=${OMP_PROC_BIND}, OMP_PLACES=${OMP_PLACES}" >> "$out"

for d in "${densities[@]}"; do
  ./build/convert_csr "$m" "$k" "$d" "$iters" "$seed" >> "$out"
  ./build/spmm_csr   "$m" "$k" "$n" "$d" "$iters" "$threads" "$seed" >> "$out"
done

echo "wrote $out"
