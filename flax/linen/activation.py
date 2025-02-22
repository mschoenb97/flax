# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Activation functions."""

# pylint: disable=unused-import
# re-export activation functions from jax.nn
from typing import Any, Optional

from flax.linen.module import compact
from flax.linen.module import Module
from flax.linen.linear import Dense

from jax.nn import celu
from jax.nn import elu
from jax.nn import gelu
from jax.nn import glu
from jax.nn import hard_sigmoid
from jax.nn import hard_silu
from jax.nn import hard_swish
from jax.nn import hard_tanh
from jax.nn import leaky_relu
from jax.nn import log_sigmoid
from jax.nn import log_softmax
from jax.nn import logsumexp
from jax.nn import one_hot
from jax.nn import relu
from jax.nn import relu6
from jax.nn import selu
from jax.nn import sigmoid
from jax.nn import silu
from jax.nn import soft_sign
from jax.nn import softmax
from jax.nn import softplus
from jax.nn import standardize
from jax.nn import swish
import jax.numpy as jnp
from jax.numpy import tanh

# Normalize is a deprecated alias of standardize
normalize = standardize

# pylint: enable=unused-import


Array = Any
Dtype = Any


class PReLU(Module):
  """Parametric Rectified Linear Unit (PReLU) activation function.

  Note that PReLU is a Flax layer and not a simple activation function, so
  it needs to be initialized before being called.

  Example usage::
    >>> import flax.linen as nn

    >>> class MLP(nn.Module):
    ...   @nn.compact
    ...   def __call__(self, x):
    ...     x = nn.Dense(2)(x)
    ...     x = nn.PReLU()(x) # initialized
    ...     return x

  Attributes:
    param_dtype: the dtype passed to parameter initializers (default: float32).
    negative_slope_init: the value to initialize the negative slope
      (default 0.01).
  """

  param_dtype: Dtype = jnp.float32
  negative_slope_init: float = 0.01

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies an activation to the inputs.

    Args:
      inputs: the nd-array to apply the activation function to.

    Returns:
      The transformed input.
    """
    negative_slope = self.param(
        'negative_slope',
        lambda k: jnp.asarray(self.negative_slope_init, self.param_dtype),
    )
    return jnp.where(
        inputs >= 0, inputs, jnp.asarray(negative_slope, inputs.dtype) * inputs
    )

class GeGLU(Module):
    """Gated Linear Unit with GELU (GeGLU) activation function.

    GeGLU is a Flax layer that combines a linear transformation with a GELU
    activation function in a gating mechanism. It is often used in Transformer models
    to provide non-linear capabilities while preserving a strong linear component.

    Example usage::
        >>> import flax.linen as nn

        >>> class TransformerBlock(nn.Module):
        ...   @nn.compact
        ...   def __call__(self, x):
        ...     x = nn.Dense(2)(x)
        ...     x = nn.GeGLU()(x) # initialized
        ...     return x

    Attributes:
        features: the number of output features (default: None).
    """
    output_dim: int = -1

    @compact
    def __call__(self, inputs: Array) -> Array:
        """Applies the GeGLU activation to the inputs.

        Args:
            inputs: the nd-array to apply the GeGLU activation function to.

        Returns:
            The transformed input.
        """
        if self.output_dim == -1:
          output_dim = inputs.shape[-1]
        else:
            output_dim = self.output_dim

        x = Dense(output_dim * 2)(inputs)
        x, gate = x[..., : output_dim], x[..., output_dim :]
        return x * gelu(gate)