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

"""Check native Cosmos LoRA initialization, optimizer scope and checkpoint reload."""

import pytest
import torch
from torch import nn

from rlinf.models.embodiment.cosmos3.wovr_policy import _configure_wovr_lora

native_lora = pytest.importorskip("cosmos_framework.utils.generator.lora")
PROJECTIONS = ("q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen")
CONFIG = {"lora_rank": 2, "lora_alpha": 4}


class TinyAttention(nn.ModuleDict):
    def forward(self, x):
        return sum(self[name](x) for name in PROJECTIONS)


class TinyBlock(nn.Module):
    _no_split_modules = ["TinyBlock", "TinyAttention"]

    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention(
            {name: nn.Linear(4, 4) for name in (*PROJECTIONS, "q_proj")}
        )
        self.mlp = nn.Linear(4, 4)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = TinyBlock()
        self.action2llm = nn.Linear(4, 4)

    def forward(self, x):
        return self.language_model(x)


def test_generation_lora_preserves_base_and_restores_trained_adapters():
    net = TinyGenerator().double()
    x = torch.arange(1, 5, dtype=torch.float64)
    baseline = net(x).detach()
    base_weights = {name: p.detach().clone() for name, p in net.named_parameters()}
    _configure_wovr_lora(net, CONFIG)
    torch.testing.assert_close(net(x), baseline, rtol=0, atol=0)

    expected = {
        f"language_model.self_attn.{proj}.{adapter}.weight"
        for proj in PROJECTIONS
        for adapter in ("lora_A", "lora_B")
    }
    trainable = {name: p for name, p in net.named_parameters() if p.requires_grad}
    assert trainable.keys() == expected
    for proj in PROJECTIONS:
        layer = net.language_model.self_attn[proj]
        assert isinstance(layer, native_lora.LoraInjectedLinear)
        assert layer.lora_A.weight.dtype == layer.weight.dtype
        assert layer.lora_B.weight.dtype == layer.weight.dtype
        assert layer.lora_A.weight.shape == (2, 4)
        assert layer.lora_B.weight.shape == (4, 2)
        assert layer.lora_A.weight.count_nonzero() > 0
        assert layer.lora_B.weight.count_nonzero() == 0

    optimizer = torch.optim.SGD(trainable.values(), lr=0.01)
    net(x).sum().backward()
    for proj in PROJECTIONS:
        assert net.language_model.self_attn[proj].lora_B.weight.grad.count_nonzero() > 0
    optimizer.step()
    assert not torch.equal(net(x), baseline)
    for name, value in base_weights.items():
        param = net.get_parameter(name)
        assert param.grad is None
        torch.testing.assert_close(param, value, rtol=0, atol=0)

    restored = TinyGenerator().double()
    _configure_wovr_lora(restored, CONFIG)
    restored.load_state_dict(net.state_dict(), strict=True)
    _configure_wovr_lora(restored, CONFIG)
    for name, param in restored.named_parameters():
        torch.testing.assert_close(param, net.get_parameter(name), rtol=0, atol=0)
    torch.testing.assert_close(restored(x), net(x), rtol=0, atol=0)
    for incompatible in ({"lora_rank": 3}, {"lora_alpha": 2}):
        with pytest.raises(ValueError):
            _configure_wovr_lora(restored, {**CONFIG, **incompatible})


def test_generation_lora_requires_each_projection():
    net = TinyGenerator()
    del net.language_model.self_attn["v_proj_moe_gen"]
    with pytest.raises(ValueError, match="v_proj_moe_gen"):
        _configure_wovr_lora(net, CONFIG)


@pytest.mark.parametrize("setting", [{"lora_rank": 0}, {"lora_alpha": -1}])
def test_generation_lora_rejects_invalid_scale(setting):
    with pytest.raises(ValueError):
        _configure_wovr_lora(TinyGenerator(), {**CONFIG, **setting})
