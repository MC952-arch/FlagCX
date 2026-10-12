#!/usr/bin/env python3
"""Require the shared GPU one-sided adaptor cases to run in hardware CI."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REQUIRED_CASES = (
    "GpuIput",
    "GpuIget",
    "GpuBatchPutAndGet",
    "GpuBounds",
)


def check_results(path: Path) -> list[str]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        return [f"Cannot read adaptor GoogleTest XML at {path}: {error}"]

    cases = {}
    for suite in root.iter("testsuite"):
        if suite.get("name") == "NetAdaptorLoopback":
            for case in suite.findall("testcase"):
                cases.setdefault(case.get("name"), []).append(case)

    errors = []
    for name in REQUIRED_CASES:
        matches = cases.get(name, [])
        if len(matches) != 1:
            errors.append(f"NetAdaptorLoopback.{name}: expected once, found {len(matches)}")
            continue
        case = matches[0]
        if (
            case.get("status") != "run"
            or case.get("result") != "completed"
            or case.find("skipped") is not None
        ):
            errors.append(f"NetAdaptorLoopback.{name}: skipped or not run")
        elif case.find("failure") is not None or case.find("error") is not None:
            errors.append(f"NetAdaptorLoopback.{name}: failed")
    return errors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: check_adaptor_gpu_tests.py <gtest-xml>")
    failures = check_results(Path(sys.argv[1]))
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        sys.exit(1)
    print("All four GPU one-sided adaptor cases ran and passed")
