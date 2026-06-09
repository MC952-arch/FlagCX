#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PERF_DIR="$PROJECT_ROOT/test/perf"
PERF_BIN="$PERF_DIR/host_api/build/bin"

export MPI_HOME="${MPI_HOME:-/usr/local/mpi}"
export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PROJECT_ROOT/build/lib:${LD_LIBRARY_PATH:-}"

NP="${NP:-2}"

# Detect available GPUs and skip gracefully if fewer than NP are present.
if command -v nvidia-smi &>/dev/null; then
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
  if [ "$GPU_COUNT" -lt "$NP" ]; then
    echo "WARNING: Only $GPU_COUNT GPU(s) available but NP=$NP requested. Skipping RMA tests."
    exit 0
  fi
fi

echo "=== Running RMA one-sided perf tests (np=$NP) ==="

echo ""
echo "--- test_put ---"
mpirun -np "$NP" --allow-run-as-root \
    -x FLAGCX_USE_HETERO_COMM=1 \
    -x FLAGCX_MEM_ENABLE=1 \
    -x FLAGCX_VMM_ENABLE=1 \
    "$PERF_BIN/perf_put" -b 8 -e 1M -f 2 -R 2

echo ""
echo "--- test_get ---"
mpirun -np "$NP" --allow-run-as-root \
    -x FLAGCX_USE_HETERO_COMM=1 \
    -x FLAGCX_MEM_ENABLE=1 \
    -x FLAGCX_VMM_ENABLE=1 \
    "$PERF_BIN/perf_get" -b 8 -e 1M -f 2 -R 2

echo ""
echo "--- test_one_side_register ---"
mpirun -np "$NP" --allow-run-as-root \
    -x FLAGCX_USE_HETERO_COMM=1 \
    -x FLAGCX_MEM_ENABLE=1 \
    -x FLAGCX_VMM_ENABLE=1 \
    "$PERF_BIN/perf_one_side_register" -b 8 -e 1M -f 2 -R 2

echo ""
echo "All RMA tests passed."
