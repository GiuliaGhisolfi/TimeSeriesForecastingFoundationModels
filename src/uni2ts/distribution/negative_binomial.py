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

from typing import Any, Callable, Optional, Union

import torch
from jaxtyping import Float
from torch.distributions import Distribution, Gamma, constraints
from torch.distributions.utils import (broadcast_all, lazy_property,
                                       logits_to_probs)
from torch.nn import functional as F

from ._base import DistributionOutput

PyTree = Any
class NegativeBinomial(Distribution):
    arg_constraints = {
        "total_count": constraints.positive,
        "logits": constraints.real,
    }
    support = constraints.nonnegative
    has_rsample = False

    def __init__(
        self,
        total_count: Union[float, torch.Tensor],
        logits: Union[float, torch.Tensor],
        validate_args: Optional[bool] = None,
    ):
        # FIXME: my code
        total_count = torch.nan_to_num(total_count, nan=0.0, posinf=1e9, neginf=-1e9)
        total_count = torch.clamp(total_count, min=1e-6)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)
        #################
        (
            self.total_count,
            self.logits,
        ) = broadcast_all(total_count, logits)
        self.total_count = self.total_count.type_as(self.logits)
        batch_shape = self.logits.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape: torch.Size, _instance=None) -> "NegativeBinomial":
        new = self._get_checked_instance(NegativeBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new.logits = self.logits.expand(batch_shape)
        super(NegativeBinomial, new).__init__(
            batch_shape=batch_shape,
            validate_args=False,
        )
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits, is_binary=True)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            sample = torch.poisson(
                Gamma(
                    concentration=self.total_count,
                    rate=torch.exp(-self.logits),
                    validate_args=False,
                ).sample(sample_shape),
            )
        return sample

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)
        log_unnormalized_prob = (
            self.total_count * F.logsigmoid(-self.logits)
            + F.logsigmoid(self.logits) * value
        )
        log_normalization = self._lbeta(1 + value, self.total_count) + torch.log(
            self.total_count + value
        )
        return log_unnormalized_prob - log_normalization

    def _lbeta(self, x, y):
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)

    @property
    def mean(self) -> torch.Tensor:
        return self.total_count * torch.exp(self.logits)

    @property
    def variance(self) -> torch.Tensor:
        return self.mean / torch.sigmoid(-self.logits)


class NegativeBinomialOutput(DistributionOutput):
    distr_cls = NegativeBinomial
    args_dim = dict(total_count=1, logits=1)

    @property
    def domain_map(
        self,
    ) -> PyTree:
        return dict(total_count=self._total_count, logits=self._logits)

    @staticmethod
    def _total_count(
        total_count: Float[torch.Tensor, "*batch 1"]
    ) -> Float[torch.Tensor, "*batch"]:
        return F.softplus(total_count).squeeze(-1)

    @staticmethod
    def _logits(
        logits: Float[torch.Tensor, "*batch 1"]
    ) -> Float[torch.Tensor, "*batch"]:
        return logits.squeeze(-1)
