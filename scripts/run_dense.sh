#!/usr/bin/env bash
set -euo pipefail

m="${1:-1024}"
k="${2:-1024}"
n="${3:-1024}"
iters="${4:-5}"
threads="${5:-1}"
block="${6:-128}"
seed="${7:-1}"

./build/dense_gemm "$m" "$k" "$n" "$iters" "$threads" "$block" "$seed"
