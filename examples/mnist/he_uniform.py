"""Copied from https://github.com/google/jax/blob/main/jax/_src/nn/initializers.py"""


from collections.abc import Sequence
import math
from typing import Any, Literal, Protocol, Union

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax import random
from jax._src import core
from jax._src import dtypes
from jax._src.util import set_module

export = set_module('jax.nn.initializers')

KeyArray = jax.Array
Array = Any
# TODO: Import or define these to match
# https://github.com/numpy/numpy/blob/main/numpy/typing/_dtype_like.py.
DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex
RealNumeric = Any  # Scalar jnp array or float


# def _compute_fans(shape: core.NamedShape,
#                   in_axis: Union[int, Sequence[int]] = -2,
#                   out_axis: Union[int, Sequence[int]] = -1,
#                   batch_axis: Union[int, Sequence[int]] = ()
#                   ) -> tuple[Array, Array]:
#   """
#   Compute effective input and output sizes for a linear or convolutional layer.

#   Axes not in in_axis, out_axis, or batch_axis are assumed to constitute the
#   "receptive field" of a convolution (kernel spatial dimensions).
#   """
#   if shape.rank <= 1:
#     raise ValueError(f"Can't compute input and output sizes of a {shape.rank}"
#                      "-dimensional weights tensor. Must be at least 2D.")

#   if isinstance(in_axis, int):
#     in_size = shape[in_axis]
#   else:
#     in_size = math.prod([shape[i] for i in in_axis])
#   if isinstance(out_axis, int):
#     out_size = shape[out_axis]
#   else:
#     out_size = math.prod([shape[i] for i in out_axis])
#   if isinstance(batch_axis, int):
#     batch_size = shape[batch_axis]
#   else:
#     batch_size = math.prod([shape[i] for i in batch_axis])
#   receptive_field_size = shape.total / in_size / out_size / batch_size
#   fan_in = in_size * receptive_field_size
#   fan_out = out_size * receptive_field_size
#   return fan_in, fan_out



def _compute_fans(shape):
    """Copied from keras github

    Computes the number of input and output units for a weight shape.

    Args:
      shape: Integer shape tuple or TF tensor shape.

    Returns:
      A tuple of integer scalars (fan_in, fan_out).
    """
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)



def get_he_uniform_max_val(shape):

  # return 1.0

  fan_in, _ = _compute_fans(shape)

  return jnp.sqrt(6.0 / (fan_in))