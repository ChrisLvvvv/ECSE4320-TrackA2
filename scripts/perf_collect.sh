#!/usr/bin/env bash
set -euo pipefail

# usage:
#   ./scripts/perf_collect.sh <tag> -- <cmd...>
# example:
#   ./scripts/perf_collect.sh dense_2048x2048 -- ./build/dense_gemm 2048 2048 2048 5 8 128 1

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <tag> -- <cmd...>"
  exit 1
fi

tag="$1"
shift

if [[ "${1:-}" != "--" ]]; then
  echo "usage: $0 <tag> -- <cmd...>"
  exit 1
fi
shift

mkdir -p results/raw

ts="$(date +%Y%m%d_%H%M%S)"
out="results/raw/perf_${tag}_${ts}.txt"

echo "# tag=${tag}" | tee "$out"
echo "# cmd=$*" | tee -a "$out"
echo "# uname=$(uname -a)" | tee -a "$out"
echo "# gcc=$(gcc --version | head -n 1)" | tee -a "$out"
echo "" | tee -a "$out"

# counters: keep it common + stable
events=(
  cycles
  instructions
  cache-references
  cache-misses
  branches
  branch-misses
  L1-dcache-loads
  L1-dcache-load-misses
  LLC-loads
  LLC-load-misses
)

# perf in WSL2 can be finicky. If perf fails, we still run the command and save stdout/stderr.
set +e
perf stat -e "$(IFS=,; echo "${events[*]}")" -- "$@" 2>>"$out" | tee -a "$out"
rc=$?
set -e

if [[ $rc -ne 0 ]]; then
  echo "" | tee -a "$out"
  echo "# perf failed (rc=$rc). Running command without perf..." | tee -a "$out"
  "$@" 2>&1 | tee -a "$out"
fi

echo "wrote $out"
