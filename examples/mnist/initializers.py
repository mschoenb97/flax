from jax import numpy as jnp
from jax import lax

import jax
KeyArray = jax.Array
from jax._src import core
from typing import Any
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex

from jax.nn.initializers import he_uniform

from he_uniform import get_he_uniform_max_val

class ste_initializer:

  def __init__(self):

    self.init_func = he_uniform()

  def remap(self, x):

    return x

  def __call__(self, key: KeyArray,
                shape: core.Shape,
                dtype: DTypeLikeInexact = jnp.float_):
    
    return self.init_func(key, shape, dtype)

class dsq_multi_bit_initializer:

  def __init__(self, bits, k):
    self.bits = bits
    self._k = k
    self.max_val = None
    self.init_func = he_uniform()

  @property
  def k(self):
    return self._k / self.max_val

  @property
  def delta(self):

    return (2.0 * self.max_val) / (2.0 ** self.bits - 1)

  @property
  def interval_integral(self):

    s = 1.0 / jnp.tanh(0.5 * self.k * self.delta)
    endpoint = self.k * self.delta / 2.0

    integral = (endpoint + jnp.sinh(endpoint) * jnp.cosh(endpoint)) / (s)
    integral = integral * 2.0 / (self.delta * (self.k ** 2))

    return integral

  @property
  def lr_adjustment(self):

    return self.delta / self.interval_integral

  def remap(self, x):

    i = jnp.floor((x + self.max_val) / self.delta)

    zero_point = jnp.floor((2.0 ** self.bits - 1) / 2.0)
    centered_i = i - zero_point
    base_point = self.interval_integral * centered_i

    m_i = -self.max_val + (i + 0.5) * self.delta
    s = 1.0 / jnp.tanh(0.5 * self.k * self.delta)
    endpoint = self.k * (x - m_i)
    increment = (endpoint + jnp.sinh(endpoint) * jnp.cosh(endpoint)) / s
    increment = increment / (self.delta * (self.k ** 2))

    integral = base_point + increment

    remap_val = self.lr_adjustment * integral

    return remap_val

  def __call__(self, key: KeyArray,
                shape: core.Shape,
                dtype: DTypeLikeInexact = jnp.float_):

    self.max_val = get_he_uniform_max_val(shape)

    return self.remap(self.init_func(key, shape, dtype))