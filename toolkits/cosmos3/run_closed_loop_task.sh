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

set -euo pipefail

COSMOS_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COSMOS_REPO_ROOT="$(cd -- "${COSMOS_SCRIPT_DIR}/../.." && pwd)"
cd -- "${COSMOS_REPO_ROOT}"

exec "${COSMOS_PYTHON:-python3}" -m toolkits.cosmos3.run_closed_loop \
    --config "${COSMOS_SCRIPT_DIR}/config/closed_loop_nano.yaml" "$@"
