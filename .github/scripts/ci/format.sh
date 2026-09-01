# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# Utility file to run pre-commit hooks locally.
# Usage: bash .github/scripts/ci/format.sh

set -e

: "${TE_PATH:=.}"

cd "$TE_PATH"

pip3 install pre-commit clang-format==14.0.6
clang-format --version
python3 -m pre_commit run --all-files
