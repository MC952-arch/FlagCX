#!/usr/bin/env bash

set -euo pipefail

mpi_timeout="${FLAGCX_CI_MPI_TIMEOUT:-20m}"
mpi_kill_after="${FLAGCX_CI_MPI_KILL_AFTER:-30s}"
mpi_launcher="${FLAGCX_CI_MPIRUN:-mpirun}"

if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required to run CI MPI tests" >&2
  exit 1
fi
if ! command -v "$mpi_launcher" >/dev/null 2>&1; then
  echo "MPI launcher not found: $mpi_launcher" >&2
  exit 1
fi

# Each call owns a fresh timeout process, so one stalled MPI test cannot consume
# the entire job timeout or affect the budget of a later MPI invocation.
echo "Starting MPI invocation (timeout=$mpi_timeout, kill-after=$mpi_kill_after)"
if timeout --signal=TERM --kill-after="$mpi_kill_after" \
  "$mpi_timeout" "$mpi_launcher" "$@"; then
  exit 0
else
  status=$?
  if [[ "$status" == 124 || "$status" == 137 ]]; then
    echo "MPI invocation exceeded its $mpi_timeout timeout" >&2
  fi
  exit "$status"
fi
