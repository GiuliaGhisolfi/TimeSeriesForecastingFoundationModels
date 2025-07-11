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

import abc
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import torch
from einops import rearrange
from jaxtyping import Float
from torch import nn
from torch.distributions import (AffineTransform, Distribution,
                                 TransformedDistribution)
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from uni2ts.common.core import abstract_class_property
from uni2ts.module.ts_embed import MultiOutSizeLinear

PyTree = Any


# TODO: Replace with tree_map when multiple trees supported
def tree_map_multi(
    func: Callable, tree: PyTree, *other: PyTree
) -> PyTree:
    """Tree map with function requiring multiple inputs, where other inputs are from a PyTree too."""
    leaves, treespec = tree_flatten(tree)
    other_leaves = [tree_flatten(o)[0] for o in other]
    return_leaves = [func(*leaf) for leaf in zip(leaves, *other_leaves)]
    return tree_unflatten(return_leaves, treespec)


def convert_to_module(tree: PyTree) -> PyTree:
    """Convert a simple container PyTree into an nn.Module PyTree"""
    if isinstance(tree, dict):
        return nn.ModuleDict(
            {key: convert_to_module(child) for key, child in tree.items()}
        )
    if isinstance(tree, (list, tuple)):
        return nn.ModuleList([convert_to_module(child) for child in tree])
    return tree


def convert_to_container(tree: PyTree) -> PyTree:
    """Convert an nn.Module PyTree into a simple container PyTree"""
    if isinstance(tree, nn.ModuleDict):
        return {key: convert_to_container(child) for key, child in tree.items()}
    if isinstance(tree, nn.ModuleList):
        return [convert_to_container(child) for child in tree]
    return tree


class DistrParamProj(nn.Module):
    """
    Projection layer from representations to distribution parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: Union[int, tuple[int, ...], list[int]],
        args_dim: PyTree,
        domain_map: PyTree,
        proj_layer: Callable[..., nn.Module] = MultiOutSizeLinear,
        **kwargs: Any,
    ):
        """
        :param in_features: size of representation
        :param out_features: size multiplier of distribution parameters
        :param args_dim: dimensionality of distribution parameters
        :param domain_map: mapping for distribution parameters
        :param proj_layer: projection layer
        :param kwargs: additional kwargs for proj_layer
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.args_dim = args_dim
        self.domain_map = domain_map

        if isinstance(out_features, int):

            def proj(dim):
                proj_layer(in_features, dim * out_features, **kwargs)

        elif isinstance(out_features, Sequence):

            def proj(dim):
                return proj_layer(
                    in_features,
                    tuple(dim * of for of in out_features),
                    dim=dim,
                    **kwargs,
                )

        else:
            raise ValueError(
                f"out_features must be int or sequence of ints, got invalid type: {type(out_features)}"
            )

        self.proj = convert_to_module(tree_map(proj, args_dim))
        self.out_size = (
            out_features if isinstance(out_features, int) else max(out_features)
        )

    def forward(self, *args) -> PyTree:
        params_unbounded = tree_map(
            lambda proj: rearrange(
                proj(*args),
                "... (dim out_size) -> ... out_size dim",
                out_size=self.out_size,
            ),
            convert_to_container(self.proj),
        )
        params = tree_map_multi(
            lambda func, inp: func(inp), self.domain_map, params_unbounded
        )
        return params


class AffineTransformed(TransformedDistribution):
    def __init__(
        self,
        base_dist: Distribution,
        loc: Optional[Union[torch.Tensor, float]] = None,
        scale: Optional[Union[torch.Tensor, float]] = None,
        validate_args: Optional[bool] = None,
    ):
        self.loc = loc if loc is not None else 0.0
        self.scale = scale if scale is not None else 1.0
        super().__init__(
            base_dist,
            [AffineTransform(loc=self.loc, scale=self.scale)],
            validate_args=validate_args,
        )

    @property
    def mean(self) -> torch.Tensor:
        return self.base_dist.mean * self.scale + self.loc

    @property
    def variance(self) -> torch.Tensor:
        return self.base_dist.variance * self.scale**2


@abstract_class_property("distr_cls")
class DistributionOutput:
    """
    Base class for distribution outputs.
    Defines the type of output distribution and provides several helper methods for predictive distributions.
    """

    distr_cls: type[Distribution] = NotImplemented

    def distribution(
        self,
        distr_params: PyTree,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> Distribution:
        distr = self._distribution(distr_params, validate_args=validate_args)
        if loc is not None or scale is not None:
            distr = AffineTransformed(distr, loc=loc, scale=scale)
        return distr

    def _distribution(
        self,
        distr_params: PyTree,
        validate_args: Optional[bool] = None,
    ) -> Distribution:
        for param, value in distr_params.items(): # FIXME: my code
            value = torch.nan_to_num(value, nan=0.0, posinf=1e9, neginf=-1e9) # clamp if needed
            if param == "df" or param == "scale":
                value = torch.clamp(value, min=1e-6)
            distr_params[param] = value

        return self.distr_cls(**distr_params, validate_args=validate_args)

    @property
    @abc.abstractmethod
    def args_dim(self) -> PyTree:
        """
        Returns the dimensionality of the distribution parameters in the form of a pytree.
        For simple distributions, this will be a simple dictionary:
        e.g. for a univariate normal distribution, the args_dim should return {"loc": 1, "scale": 1}.
        For more complex distributions, this could be an arbitrarily complex pytree.

        :return: pytree of integers representing the dimensionality of the distribution parameters
        """
        ...

    @property
    @abc.abstractmethod
    def domain_map(self) -> PyTree:
        """
        Returns a pytree of callables that maps the unconstrained distribution parameters
        to the range required by their distributions.

        :return: callables in the same PyTree format as args_dim
        """
        ...

    def get_param_proj(
        self,
        in_features: int,
        out_features: Union[int, tuple[int, ...], list[int]],
        proj_layer: Callable[..., nn.Module] = MultiOutSizeLinear,
        **kwargs: Any,
    ) -> nn.Module:
        """
        Get a projection layer mapping representations to distribution parameters.

        :param in_features: input feature dimension
        :param out_features: size multiplier of distribution parameters
        :param proj_layer: projection layer
        :param kwargs: additional kwargs for proj_layer
        :return: distribution parameter projection layer
        """
        return DistrParamProj(
            in_features=in_features,
            out_features=out_features,
            args_dim=self.args_dim,
            domain_map=self.domain_map,
            proj_layer=proj_layer,
            **kwargs,
        )
