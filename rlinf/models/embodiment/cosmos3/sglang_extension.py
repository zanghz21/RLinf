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

"""Register Cosmos3 policy sampling through SGLang's external package API."""

from __future__ import annotations

import json

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.registry import register_pipeline
from sglang.multimodal_gen.runtime.pipelines.cosmos3_pipeline import Cosmos3Pipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs

from rlinf.utils.logging import get_logger


class RLinfCosmos3SamplingParams(Cosmos3SamplingParams):
    """Honor policy guidance supplied by the action endpoint, whose default is 1."""

    def _guidance_is_explicit(self) -> bool:
        if self.action_mode == "policy":
            return True
        return super()._guidance_is_explicit()

    def _adjust(self, server_args: ServerArgs) -> None:
        super()._adjust(server_args)
        get_logger().info(
            "RLinf Cosmos3 sampling: %s",
            json.dumps(
                {
                    "action_mode": self.action_mode,
                    "request_id": self.request_id,
                    "guidance_scale": self.guidance_scale,
                    "num_inference_steps": self.num_inference_steps,
                    "num_frames": self.num_frames,
                    "width": self.width,
                    "height": self.height,
                },
                sort_keys=True,
            ),
        )


class RLinfCosmos3Pipeline(Cosmos3Pipeline):
    """Reuse the native Cosmos3 pipeline with RLinf policy sampling settings."""

    pipeline_name = "RLinfCosmos3Pipeline"


register_pipeline(
    RLinfCosmos3Pipeline,
    sampling_param_cls=RLinfCosmos3SamplingParams,
    pipeline_config_cls=Cosmos3Config,
)
