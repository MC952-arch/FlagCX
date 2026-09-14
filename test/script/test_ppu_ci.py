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

    def test_ppu_workload_covers_all_ten_collectives_in_both_modes(self):
        source = (
            REPO_ROOT / ".github/scripts/ci/run_ppu_workload.sh"
        ).read_text()
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
            self.assertIn(operation, source)
        self.assertIn("run_perf_suite homogeneous 128M 1G", source)
        self.assertIn("run_perf_suite heterogeneous 128M 1G", source)
        self.assertNotIn("run_perf heterogeneous sendrecv", source)
        self.assertIn("FLAGCX_USE_HETERO_COMM=1", source)
        self.assertIn("FLAGCX_CLUSTER_SPLIT_LIST=2", source)
        self.assertIn("FLAGCX_P2P_TRANSPORT=accl", source)
        self.assertIn("NET/(ACCL_P2P|BAREX)", source)

        clean_mode = source[source.index("local -a clean_mode_env=("):]
        clean_mode = clean_mode[:clean_mode.index("local -a mode_env=()")]
        for variable in (
            "FLAGCX_USE_HETERO_COMM",
            "FLAGCX_CLUSTER_SPLIT_LIST",
            "FLAGCX_MEM_ENABLE",
            "FLAGCX_VMM_ENABLE",
            "FLAGCX_P2P_TRANSPORT",
            "FLAGCX_P2P_DISABLE",
        ):
            self.assertIn(f"-u {variable}", clean_mode)

    def test_ppu_jobs_are_present_in_public_workflows(self):
        perf_workflow = (
            REPO_ROOT / ".github/workflows/test.yml"
        ).read_text()
        torch_workflow = (
            REPO_ROOT / ".github/workflows/torch-api-test.yml"
        ).read_text()

        self.assertIn("perf-test-ppu:", perf_workflow)
        self.assertIn("name: perf-test (ppu)", perf_workflow)
        self.assertIn("run_ppu_container.sh\" perf", perf_workflow)
        self.assertIn("torch-api-test-ppu:", torch_workflow)
        self.assertIn("name: torch-api-test (ppu)", torch_workflow)
        self.assertIn("run_ppu_container.sh\" torch-api", torch_workflow)

        # BAREX needs host networking. GitHub Actions job containers reject
        # --network, so the PPU jobs must launch Docker explicitly.
        self.assertNotIn("container:", perf_workflow[perf_workflow.index("perf-test-ppu:"):])
        self.assertNotIn("container:", torch_workflow[torch_workflow.index("torch-api-test-ppu:"):])

        container_runner = (
            REPO_ROOT / ".github/scripts/ci/run_ppu_container.sh"
        ).read_text()
        self.assertIn("--network=host", container_runner)
        self.assertIn("docker run --rm", container_runner)

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
