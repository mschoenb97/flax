from jax import numpy as jnp
from jax import lax
import jax
from he_uniform import get_he_uniform_max_val
from flax.struct import dataclass

@dataclass
class pwl_multi_bit_quantizer:
  """Pwl quantizer to match the behavior of the dsq quantizer"""


  bits: int
  k: float
  adjust_learning_rate: bool

  def k_norm(self, max_val):
    return self.k / max_val

  def delta(self, max_val):
    return (2.0 * max_val) / (2.0 ** self.bits - 1)


  def interval_integral(self, max_val):

    s = 1 / jnp.tanh(0.5 * self.k_norm(max_val) * self.delta(max_val))
    endpoint = self.k_norm(max_val) * self.delta(max_val) / 2.0
    integral = (endpoint + jnp.sinh(endpoint) * jnp.cosh(endpoint)) / (s)
    integral = integral * 2.0 / (self.delta(max_val) * (self.k_norm(max_val) ** 2))

    return integral

  def lr_adjustment(self, max_val):

    if self.adjust_learning_rate:
      return self.delta(max_val) / self.interval_integral(max_val)
    else:
      return 1.0

  def __call__(self, x):

    max_val = get_he_uniform_max_val(x.shape)

    quantized_estimate = self.lr_adjustment(max_val) * jnp.clip(x, -max_val, max_val)

    clipped = jnp.clip(x, -max_val, max_val)
    quantized = self.delta(max_val) * jnp.round((clipped + max_val) / self.delta(max_val)) - max_val

    return quantized_estimate + lax.stop_gradient(quantized - quantized_estimate)

@dataclass
class dsq_multi_bit_quantizer:
  """k is a parameter that determines how "sharp" the approximation is.
  The larger the value, the closer this is to the staircase function.
  k=2.5 is a reasonable value
  See https://arxiv.org/pdf/1908.05033.pdf.

  max_val determines the max absolute value of the representable range.
  I'd think that 3 would be a good value for gaussian distributed weights with
  mean 0 and standard deviation 1.
  """

  bits: int
  k: float

  def k_norm(self, max_val):
    return self.k / max_val

  def __call__(self, x):

    max_val = get_he_uniform_max_val(x.shape)

    delta = (2.0 * max_val) / (2.0 ** self.bits - 1)

    i = jnp.floor((x + max_val) / delta)
    m_i = -max_val + (i + 0.5) * delta
    s = 1.0 / jnp.tanh(0.5 * self.k_norm(max_val) * delta)
    phi = s * (jnp.tanh(self.k_norm(max_val) * (x - m_i)))

    phi = phi * delta / 2.0

    quantized_estimate = jnp.where(x < - max_val, - max_val, phi)
    quantized_estimate = jnp.where(x > max_val, max_val, quantized_estimate)


    clipped = jnp.clip(x, -max_val, max_val)
    quantized = delta * jnp.round((clipped + max_val) / delta) - max_val

    return quantized_estimate + lax.stop_gradient(quantized - quantized_estimate)
