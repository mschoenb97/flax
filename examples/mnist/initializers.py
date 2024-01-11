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
    self.init_func = he_uniform()

  def k(self, max_val):
    return self._k / max_val

  def delta(self, max_val):

    return (2.0 * max_val) / (2.0 ** self.bits - 1)

  def interval_integral(self, max_val):

    s = 1.0 / jnp.tanh(0.5 * self.k(max_val) * self.delta(max_val))
    endpoint = self.k(max_val) * self.delta(max_val) / 2.0

    integral = (endpoint + jnp.sinh(endpoint) * jnp.cosh(endpoint)) / (s)
    integral = integral * 2.0 / (self.delta(max_val) * (self.k(max_val) ** 2))

    return integral

  def lr_adjustment(self, max_val):

    return self.delta(max_val) / self.interval_integral(max_val)

  def remap(self, x):

    max_val = get_he_uniform_max_val(x.shape)

    i = jnp.floor((x + max_val) / self.delta(max_val))

    zero_point = jnp.floor((2.0 ** self.bits - 1) / 2.0)
    centered_i = i - zero_point
    base_point = self.interval_integral(max_val) * centered_i

    m_i = -max_val + (i + 0.5) * self.delta(max_val)
    s = 1.0 / jnp.tanh(0.5 * self.k(max_val) * self.delta(max_val))
    endpoint = self.k(max_val) * (x - m_i)
    increment = (endpoint + jnp.sinh(endpoint) * jnp.cosh(endpoint)) / s
    increment = increment / (self.delta(max_val) * (self.k(max_val) ** 2))

    integral = base_point + increment

    remap_val = self.lr_adjustment(max_val) * integral

    return remap_val

  def __call__(self, key: KeyArray,
                shape: core.Shape,
                dtype: DTypeLikeInexact = jnp.float_):

    return self.remap(self.init_func(key, shape, dtype))
  
def get_initializer_from_config(config):
    """
    Factory function to create an initializer instance based on the provided configuration.

    Args:
        config: A configuration object or dict containing the initializer type and its parameters.

    Returns:
        An instance of either ste_initializer or dsq_multi_bit_initializer.
    """
    initializer_type = config.get('initializer_type')
    
    if initializer_type == 'ste':
        return ste_initializer()
    elif initializer_type == 'dsq':
        bits = config.get('bits')
        k = config.get('k')
        if bits is None or k is None:
            raise ValueError("Missing required parameters 'bits' and 'k' for dsq initializer.")
        return dsq_multi_bit_initializer(bits, k)
    else:
        raise ValueError(f"Unknown initializer type: {initializer_type}")
