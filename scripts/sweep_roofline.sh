#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/raw

out="results/raw/roofline_points.csv"
: > "$out"

export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# measure sustained bandwidth
./build/stream_bw 268435456 10 >> "$out"

# representative kernels/points
# dense: sizes spanning cache->dram
for s in 512 1024 2048 4096; do
  ./build/dense_gemm "$s" "$s" "$s" 5 10 128 1 >> "$out"
done

# spmm: fixed size, sweep density (reuses your density grid style)
for d in 0.001 0.01 0.05 0.20; do
  ./build/spmm_csr 1024 1024 1024 "$d" 5 10 1 >> "$out"
done

echo "wrote $out"
