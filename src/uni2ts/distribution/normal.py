#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Any, Callable, Optional

import torch
from jaxtyping import Float
from torch.distributions import Normal
from torch.nn import functional as F

from ._base import DistributionOutput

PyTree = Any

class NormalOutput(DistributionOutput):
    distr_cls = Normal
    args_dim = dict(loc=1, scale=1)

    @property
    def domain_map(self) -> PyTree:
        return dict(
            loc=self._loc,
            scale=self._scale,
        )

    @staticmethod
    def _loc(loc: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        return loc.squeeze(-1)

    @staticmethod
    def _scale(scale: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        epsilon = torch.finfo(scale.dtype).eps
        return F.softplus(scale).clamp_min(epsilon).squeeze(-1)


class NormalFixedScaleOutput(DistributionOutput):
    distr_cls = Normal
    args_dim = dict(loc=1)

    def __init__(self, scale: float = 1e-3):
        self.scale = scale

    @property
    def domain_map(
        self,
    ) -> PyTree:
        return dict(loc=self._loc)

    @staticmethod
    def _loc(loc: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        return loc.squeeze(-1)

    def _distribution(
        self,
        distr_params: PyTree,
        validate_args: Optional[bool] = None,
    ) -> Normal:
        loc = distr_params["loc"]
        distr_params["scale"] = torch.as_tensor(
            self.scale, dtype=loc.dtype, device=loc.device
        )

        for param, value in distr_params.items(): # FIXME: my code
            value = torch.nan_to_num(value, nan=0.0, posinf=1e9, neginf=-1e9) # clamp if needed
            """if param == "df" or param == "scale":
                value = torch.clamp(value, min=1e-6)"""
            distr_params[param] = value
        
        return self.distr_cls(**distr_params, validate_args=validate_args)
