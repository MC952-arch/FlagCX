import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class PpuCiRegressionTest(unittest.TestCase):
    def test_ppu_runner_executes_barex_heterogeneous_variants(self):
        source = (
            REPO_ROOT / ".github/scripts/set_env/ppu.sh"
        ).read_text()

        self.assertNotIn("Skipping PPU runner heterogeneous", source)
        self.assertIn("runner BAREX heterogeneous SendRecv smoke", source)
        self.assertIn("runner BAREX heterogeneous", source)
        self.assertIn("runner BAREX forced NET", source)
        self.assertGreaterEqual(source.count("FLAGCX_P2P_TRANSPORT=accl"), 3)
        self.assertNotIn("NCCL_P2P_DISABLE", source)
        self.assertNotIn("NCCL_SHM_DISABLE", source)
        self.assertEqual(source.count("FLAGCX_USE_HETERO_COMM=1"), 1)
        full_heterogeneous = source[
            source.index('FLAGCX_CI_MPI_LABEL="runner BAREX heterogeneous"'):
        ]
        self.assertNotIn("FLAGCX_USE_HETERO_COMM=1", full_heterogeneous)
        self.assertEqual(
            full_heterogeneous.count(
                "env -u FLAGCX_USE_HOST_COMM -u FLAGCX_USE_HETERO_COMM"
            ),
            2,
        )
        default_runner = source[source.index('FLAGCX_CI_MPI_LABEL="runner default"'):]
        default_runner = default_runner[:default_runner.index('FLAGCX_CI_MPI_LABEL="runner BAREX')]
        for variable in (
            "FLAGCX_USE_HETERO_COMM",
            "FLAGCX_CLUSTER_SPLIT_LIST",
            "FLAGCX_MEM_ENABLE",
            "FLAGCX_VMM_ENABLE",
            "FLAGCX_P2P_TRANSPORT",
            "FLAGCX_P2P_DISABLE",
        ):
            self.assertIn(f"-u {variable}", default_runner)

    def test_ppu_perf_uses_supported_collectives_for_each_runner(self):
        ppu_source = (
            REPO_ROOT / ".github/scripts/ci/run_ppu_workload.sh"
        ).read_text()
        source = (
            REPO_ROOT / ".github/scripts/ci/run_host_perf_suite.sh"
        ).read_text()
        homogeneous_operations = (
            "alltoall",
            "alltoallv",
            "sendrecv",
            "allreduce",
            "allgather",
            "reducescatter",
            "broadcast",
            "gather",
            "scatter",
            "reduce",
        )
        heterogeneous_operations = (
            "alltoall",
            "alltoallv",
            "sendrecv",
            "allgather",
            "broadcast",
            "gather",
            "scatter",
        )

        homogeneous_suite = source[source.index("homogeneous)\n"):]
        homogeneous_suite = homogeneous_suite[:homogeneous_suite.index(";;")]
        heterogeneous_suite = source[source.index("heterogeneous)\n"):]
        heterogeneous_suite = heterogeneous_suite[:heterogeneous_suite.index(";;")]
        for operation in homogeneous_operations:
            self.assertIn(operation, homogeneous_suite)
        for operation in heterogeneous_operations:
            self.assertIn(operation, heterogeneous_suite)
        for operation in ("allreduce", "reducescatter", "reduce"):
            self.assertNotIn(operation, heterogeneous_suite)
        self.assertIn('"$perf_runner" homogeneous "$perf_bin"', ppu_source)
        self.assertIn('"$perf_runner" heterogeneous "$perf_bin"', ppu_source)
        self.assertIn("FLAGCX_USE_HETERO_COMM=1", source)
        self.assertIn("FLAGCX_P2P_TRANSPORT=accl", ppu_source)

        self.assertIn("FLAGCX_MEM_ENABLE=1", heterogeneous_suite)
        self.assertNotIn("FLAGCX_CLUSTER_SPLIT_LIST", heterogeneous_suite)

        clean_mode = source[source.index("clean_mode_env=("):]
        clean_mode = clean_mode[:clean_mode.index("mpi_args=(")]
        for variable in (
            "FLAGCX_USE_HETERO_COMM",
            "FLAGCX_CLUSTER_SPLIT_LIST",
            "FLAGCX_MEM_ENABLE",
            "FLAGCX_VMM_ENABLE",
            "FLAGCX_P2P_TRANSPORT",
            "FLAGCX_P2P_DISABLE",
        ):
            self.assertIn(f"-u {variable}", clean_mode)

    def test_ppu_torch_heterogeneous_mode_uses_hybrid_runner(self):
        source = (
            REPO_ROOT / ".github/scripts/ci/run_ppu_workload.sh"
        ).read_text()
        torch_case = source[source.index("  torch-api)\n"):]
        torch_case = torch_case[:torch_case.index("    ;;")]

        self.assertIn("unset FLAGCX_USE_HETERO_COMM", torch_case)
        self.assertNotIn("export FLAGCX_USE_HETERO_COMM=1", torch_case)
        self.assertIn("export FLAGCX_CLUSTER_SPLIT_LIST=2", torch_case)
        self.assertIn("export FLAGCX_MEM_ENABLE=1", torch_case)
        self.assertIn("export FLAGCX_P2P_TRANSPORT=accl", torch_case)

    def test_ppu_jobs_are_present_in_public_workflows(self):
        perf_workflow = (
            REPO_ROOT / ".github/workflows/test.yml"
        ).read_text()
        torch_workflow = (
            REPO_ROOT / ".github/workflows/torch-api-test.yml"
        ).read_text()

        self.assertIn("perf-test-ppu:", perf_workflow)
        self.assertIn("name: perf-test (ppu)", perf_workflow)
        self.assertIn("run_hardware_container.sh\"", perf_workflow)
        self.assertIn("ppu perf", perf_workflow)
        self.assertIn("torch-api-test-ppu:", torch_workflow)
        self.assertIn("name: torch-api-test (ppu)", torch_workflow)
        self.assertIn("run_hardware_container.sh\"", torch_workflow)
        self.assertIn("ppu torch-api", torch_workflow)

        # BAREX needs host networking. GitHub Actions job containers reject
        # --network, so the PPU jobs must launch Docker explicitly.
        self.assertNotIn("container:", perf_workflow[perf_workflow.index("perf-test-ppu:"):])
        self.assertNotIn("container:", torch_workflow[torch_workflow.index("torch-api-test-ppu:"):])

        container_runner = (
            REPO_ROOT / ".github/scripts/ci/run_hardware_container.sh"
        ).read_text()
        ppu_config = (REPO_ROOT / ".github/configs/ppu.yml").read_text()
        self.assertIn("docker run --rm", container_runner)
        self.assertNotIn("ruby ", container_runner)
        for variable in (
            "FLAGCX_CI_IMAGE",
            "FLAGCX_CI_DOCKER_ARGS",
            "FLAGCX_CI_SET_ENV",
        ):
            self.assertIn(variable, container_runner)
            self.assertIn(variable, perf_workflow)
            self.assertIn(variable, torch_workflow)
        self.assertIn("--network=host", ppu_config)

    def test_reference_platform_coverage_is_explicit(self):
        perf_workflow = (
            REPO_ROOT / ".github/workflows/test.yml"
        ).read_text()
        torch_workflow = (
            REPO_ROOT / ".github/workflows/torch-api-test.yml"
        ).read_text()

        hygon_perf = perf_workflow[perf_workflow.index("  perf-test-hygon:"):]
        hygon_perf = hygon_perf[:hygon_perf.index("  perf-test-cuda:")]
        metax_perf = perf_workflow[perf_workflow.index("  perf-test-metax:"):]
        metax_perf = metax_perf[:metax_perf.index("  perf-test-ppu:")]
        hygon_handler = (
            REPO_ROOT / ".github/scripts/ci/run_hygon_workload.sh"
        ).read_text()
        self.assertIn("hygon perf", hygon_perf)
        self.assertIn('"$perf_runner" homogeneous "$perf_bin"', hygon_handler)
        self.assertIn('"$perf_runner" heterogeneous "$perf_bin"', hygon_handler)

        self.assertIn("Perf tests (MetaX uniRunner 8-chip)", metax_perf)
        self.assertIn("run_host_perf_suite.sh heterogeneous", metax_perf)

        hygon_torch = torch_workflow[torch_workflow.index("  torch-api-test-hygon:"):]
        hygon_torch = hygon_torch[:hygon_torch.index("  torch-api-test-cuda:")]
        metax_torch = torch_workflow[torch_workflow.index("  torch-api-test-metax:"):]
        metax_torch = metax_torch[:metax_torch.index("  torch-api-test-ppu:")]
        self.assertIn("hygon torch-api", hygon_torch)
        self.assertIn("unset FLAGCX_SKIP_HETERO", hygon_handler)
        self.assertNotIn("FLAGCX_USE_HETERO_COMM", metax_torch)

    def test_common_launcher_and_static_rdma_preflight_cover_all_platforms(self):
        unit_workflow = (
            REPO_ROOT / ".github/workflows/unit_tests_common.yml"
        ).read_text()
        self.assertIn("run_hardware_container.sh\"", unit_workflow)
        self.assertIn('unittest "${{ matrix.suite }}"', unit_workflow)

        for platform in ("cuda", "metax", "hygon", "ppu"):
            source = (
                REPO_ROOT / f".github/scripts/set_env/{platform}.sh"
            ).read_text()
            self.assertIn("rdma_static_preflight.sh", source)
            self.assertIn("flagcx_ci_validate_rdma_static", source)

    def test_perf_collectives_fail_fast_on_flagcx_errors(self):
        perf_dir = REPO_ROOT / "test/perf/host_api"
        operations = (
            "alltoall",
            "alltoallv",
            "sendrecv",
            "allreduce",
            "allgather",
            "reducescatter",
            "broadcast",
            "gather",
            "scatter",
            "reduce",
        )

        for operation in operations:
            source = (perf_dir / f"test_{operation}.cpp").read_text()
            self.assertIn("PERF_CHECK(flagcx", source, operation)


if __name__ == "__main__":
    unittest.main()
