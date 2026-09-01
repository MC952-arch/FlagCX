#!/usr/bin/env bash

# MetaX-specific unit-test environment setup.

FLAGCX_CI_MPI_BASE_HOME=${MPI_HOME:-/usr/local/mpi}

# Use the real OpenMPI launcher if the image provides a wrapper.
if [[ -x "$FLAGCX_CI_MPI_BASE_HOME/bin/mpirun.real" ]]; then
  FLAGCX_CI_MPI_HOME=$(mktemp -d)
  mkdir -p "$FLAGCX_CI_MPI_HOME/bin"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/bin/mpirun.real" \
    "$FLAGCX_CI_MPI_HOME/bin/mpirun"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/include" "$FLAGCX_CI_MPI_HOME/include"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/lib" "$FLAGCX_CI_MPI_HOME/lib"
  export MPI_HOME=$FLAGCX_CI_MPI_HOME
else
  export MPI_HOME=$FLAGCX_CI_MPI_BASE_HOME
fi

export PATH="/opt/maca/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/mxdriver/lib:/opt/maca/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_METAX=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_METAX=1)
FLAGCX_CI_INTRA_NP=8
FLAGCX_CI_RUNNER_NP=8
export NP=8

flagcx_ci_configure_suite() {
  local suite=$1

  case "$suite" in
    adaptor|p2p)
      export FLAGCX_DEBUG=TRACE
      export FLAGCX_DEBUG_SUBSYS=ALL
      ;;
    rma)
      FLAGCX_CI_TEST_MAKE_ARGS+=(
        "HETERO_ENV=-x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0 -x FLAGCX_USE_TUNER=1 -x TUNNING_WITH_SINGLE_COMM=1 -x FLAGCX_USE_HOST_COMM=1 -x FLAGCX_P2P_DISABLE=1"
      )
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing MetaX environment for unit-test suite: $suite"
  command -v mpirun
  command -v mxcc
}

flagcx_ci_build_suite_override() {
  local suite=$1
  local suite_dir=$2
  shift 2
  local -a args=("$@")

  if [[ "$suite" == "symmem" ]]; then
    FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=1
    cmake -S "$PROJECT_ROOT/third-party/googletest" \
      -B "$PROJECT_ROOT/third-party/googletest/build"
    cmake --build "$PROJECT_ROOT/third-party/googletest/build" --parallel "$(nproc)"
    make -C "$suite_dir" --jobs="$(nproc)" "${args[@]}"
    return
  fi

  FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=0
}

flagcx_ci_run_suite_override() {
  local suite=$1
  local suite_dir=$2
  shift 2
  local -a args=("$@")

  if [[ "$suite" == "runner" ]]; then
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=1
    make -C "$suite_dir" run-unit "${args[@]}"
    echo "Skipping MetaX runner MPI tests: mcclAllGather segfaults in the current MCCL backend."
    return
  fi

  if [[ "$suite" == "rma" ]]; then
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=1
    make -C "$suite_dir" run-unit "${args[@]}"
    echo "Skipping MetaX RMA MPI tests: one-sided RMA is not supported by the current MetaX backend."
    return
  fi

  if [[ "$suite" == "symmem" ]]; then
    FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=1
    "$suite_dir/build/bin/symmem_unit_tests"
    echo "Skipping MetaX symmem MPI tests: symmetric windows are not supported by the current MetaX backend."
    return
  fi

  FLAGCX_CI_RUN_SUITE_OVERRIDE_HANDLED=0
}
