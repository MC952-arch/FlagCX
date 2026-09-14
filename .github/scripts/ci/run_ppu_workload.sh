#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <torch-api|perf>" >&2
  exit 2
fi

workload=$1
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
mpi_runner="$project_root/.github/scripts/ci/run_mpi_with_timeout.sh"
ppu_env="$project_root/.github/scripts/set_env/ppu.sh"

# shellcheck source=/dev/null
source "$ppu_env"
flagcx_ci_prepare "$workload"
flagcx_ci_validate_rdma "$workload"

export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$project_root/build/lib:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
export FLAGCX_IB_DISABLE=0
export FLAGCX_DEBUG=${FLAGCX_DEBUG:-INFO}
export FLAGCX_DEBUG_SUBSYS=${FLAGCX_DEBUG_SUBSYS:-INIT,NET,P2P,PROXY}

build_flagcx() {
  make -C "$project_root" --jobs="$(nproc)" USE_PPU=1 USE_ACCL_BAREX=1
}

run_perf() {
  local mode=$1
  local operation=$2
  shift 2
  local binary="$project_root/test/perf/host_api/build/bin/perf_$operation"
  local -a clean_mode_env=(
    env
    -u FLAGCX_USE_HOST_COMM
    -u FLAGCX_USE_HETERO_COMM
    -u FLAGCX_CLUSTER_SPLIT_LIST
    -u FLAGCX_MEM_ENABLE
    -u FLAGCX_VMM_ENABLE
    -u FLAGCX_P2P_TRANSPORT
    -u FLAGCX_P2P_DISABLE
  )
  local -a mode_env=()

  if [[ "$mode" == heterogeneous ]]; then
    mode_env=(
      -x FLAGCX_USE_HETERO_COMM=1
      -x FLAGCX_CLUSTER_SPLIT_LIST=2
      -x FLAGCX_MEM_ENABLE=1
      -x FLAGCX_VMM_ENABLE=0
      -x FLAGCX_P2P_TRANSPORT=accl
      -x FLAGCX_IB_DISABLE=0
    )
  fi

  FLAGCX_CI_MPI_LABEL="PPU $mode perf: $operation" \
    "${clean_mode_env[@]}" "$mpi_runner" -np 8 --allow-run-as-root \
    -x LD_LIBRARY_PATH -x LD_PRELOAD -x FLAGCX_DEBUG \
    -x FLAGCX_DEBUG_SUBSYS "${mode_env[@]}" "$binary" "$@"
}

run_perf_suite() {
  local mode=$1
  local begin=$2
  local end=$3
  local log_file=${4:-}
  local -a common_args=(-b "$begin" -e "$end" -f 2 -p 1)
  local operation

  for operation in alltoall alltoallv sendrecv allreduce allgather \
                   reducescatter; do
    if [[ -n "$log_file" ]]; then
      run_perf "$mode" "$operation" "${common_args[@]}" 2>&1 | tee -a "$log_file"
    else
      run_perf "$mode" "$operation" "${common_args[@]}"
    fi
  done
  for operation in broadcast gather scatter reduce; do
    if [[ -n "$log_file" ]]; then
      run_perf "$mode" "$operation" "${common_args[@]}" -r 0 2>&1 | tee -a "$log_file"
    else
      run_perf "$mode" "$operation" "${common_args[@]}" -r 0
    fi
  done
}

case "$workload" in
  torch-api)
    command -v python3
    python3 -c 'import torch; print("torch", torch.__version__, "devices", torch.cuda.device_count()); assert torch.cuda.device_count() >= 8'
    build_flagcx
    (
      cd "$project_root/plugin/torch"
      export TORCH_DEVICE_BACKEND_AUTOLOAD=0
      export FLAGCX_ADAPTOR=ppu
      export USE_PPU=1
      python3 setup.py build_ext --inplace
    )

    torch_log=${RUNNER_TEMP:-/tmp}/flagcx-ppu-torch-api.log
    export PYTHON_BIN=python3
    export FLAGCX_ADAPTOR=ppu
    export FLAGCX_USE_HETERO_COMM=1
    export FLAGCX_CLUSTER_SPLIT_LIST=2
    export FLAGCX_MEM_ENABLE=1
    export FLAGCX_VMM_ENABLE=0
    export FLAGCX_P2P_TRANSPORT=accl
    bash "$project_root/test/script/torch_api_test.sh" 2>&1 | tee "$torch_log"
    if ! grep -Eq 'NET/(ACCL_P2P|BAREX)' "$torch_log"; then
      echo "PPU heterogeneous Torch API tests did not report ACCL/BAREX transport" >&2
      exit 1
    fi
    ;;
  perf)
    build_flagcx
    make -C "$project_root/test/perf" --jobs="$(nproc)" \
      USE_PPU=1 USE_ACCL_BAREX=1

    # Homogeneous PCCL parity with the CUDA/Hygon/MetaX platform jobs.
    run_perf_suite homogeneous 128M 1G

    barex_log=${RUNNER_TEMP:-/tmp}/flagcx-ppu-barex-perf.log
    : >"$barex_log"
    run_perf_suite heterogeneous 128M 1G "$barex_log"
    if ! grep -Eq 'NET/(ACCL_P2P|BAREX)' "$barex_log"; then
      echo "PPU heterogeneous perf tests did not report ACCL/BAREX transport" >&2
      exit 1
    fi
    ;;
  *)
    echo "Unknown PPU workload: $workload" >&2
    exit 2
    ;;
esac
