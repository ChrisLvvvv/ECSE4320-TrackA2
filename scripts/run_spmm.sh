#!/usr/bin/env bash
set -euo pipefail

m="${1:-1024}"
k="${2:-1024}"
n="${3:-1024}"
density="${4:-0.05}"
iters="${5:-5}"
threads="${6:-1}"
seed="${7:-1}"

./build/spmm_csr "$m" "$k" "$n" "$density" "$iters" "$threads" "$seed"
