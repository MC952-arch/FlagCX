"""Check that hardware CI rejects missing or skipped GPU adaptor cases."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / ".github/scripts/ci/check_adaptor_gpu_tests.py"
SPEC = importlib.util.spec_from_file_location("check_adaptor_gpu_tests", CHECKER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AdaptorGpuCiTest(unittest.TestCase):
    def check(self, names, skipped=()):
        cases = "".join(
            f'<testcase name="{name}" status="run" result="completed">'
            f'{"<skipped/>" if name in skipped else ""}</testcase>'
            for name in names
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.xml"
            path.write_text(
                f'<testsuites><testsuite name="NetAdaptorLoopback">'
                f"{cases}</testsuite></testsuites>"
            )
            return MODULE.check_results(path)

    def test_all_cases_pass(self):
        self.assertEqual(self.check(MODULE.REQUIRED_CASES), [])

    def test_skipped_case_fails(self):
        failures = self.check(MODULE.REQUIRED_CASES, skipped=("GpuIget",))
        self.assertEqual(failures, ["NetAdaptorLoopback.GpuIget: skipped or not run"])

    def test_missing_case_fails(self):
        failures = self.check(MODULE.REQUIRED_CASES[:-1])
        self.assertEqual(failures, ["NetAdaptorLoopback.GpuBounds: expected once, found 0"])


if __name__ == "__main__":
    unittest.main()
