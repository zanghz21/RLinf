#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run a command with project-local CUDA 13 compatibility libraries.
set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: bash run_with_cuda_compat.sh COMMAND [ARG ...]" >&2
    exit 2
fi

workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
compat_dir="${COSMOS3_CUDA_COMPAT_DIR:-${workspace_dir}/.cuda-compat/13.0/usr/local/cuda-13.0/compat}"
driver_versions="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u)"
if [[ ! "$driver_versions" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Expected one NVIDIA driver version, got: ${driver_versions}" >&2
    exit 1
fi
driver_major="${driver_versions%%.*}"

if (( driver_major == 570 )); then
    if [[ ! -r "${compat_dir}/libcuda.so.1" || ! -r "${compat_dir}/libnvidia-ptxjitcompiler.so.1" ]]; then
        echo "CUDA 13 compatibility libraries are missing from ${compat_dir}" >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="${compat_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "CUDA 13 compatibility enabled for driver ${driver_versions}: ${compat_dir}" >&2
elif (( driver_major >= 580 )); then
    echo "Using system CUDA driver ${driver_versions}; compatibility override is unnecessary." >&2
else
    echo "This launcher supports driver 570 or >=580; found ${driver_versions}." >&2
    exit 1
fi

exec "$@"
