from jax import numpy as jnp
from jax import lax
from he_uniform import get_he_uniform_max_val

class pwl_multi_bit_quantizer:
  """Pwl quantizer to match the behavior of the dsq quantizer"""

  def __init__(self, bits, k, adjust_learning_rate):

    self.bits = bits
    self._k = k
    self.max_val = None
    self.adjust_learning_rate = adjust_learning_rate


  @property
  def k(self):
    return self._k / self.max_val

  @property
  def delta(self):
    return (2.0 * self.max_val) / (2.0 ** self.bits - 1)


  @property
  def interval_integral(self):

    s = 1 / jnp.tanh(0.5 * self.k * self.delta)
    endpoint = self.k * self.delta / 2.0
    integral = (endpoint + jnp.sinh(endpoint) * jnp.cosh(endpoint)) / (s)
    integral = integral * 2.0 / (self.delta * (self.k ** 2))

    return integral

  @property
  def lr_adjustment(self):

    if self.adjust_learning_rate:
      return self.delta / self.interval_integral
    else:
      return 1.0

  def set_shape(self, shape):
    """set the shape of the input data"""

    max_val = get_he_uniform_max_val(shape)
    self.max_val = max_val

  def __call__(self, x):

    quantized_estimate = self.lr_adjustment * jnp.clip(x, -self.max_val, self.max_val)

    clipped = jnp.clip(x, -self.max_val, self.max_val)
    quantized = self.delta * jnp.round((clipped + self.max_val) / self.delta) - self.max_val

    return quantized_estimate + lax.stop_gradient(quantized - quantized_estimate)

class dsq_multi_bit_quantizer:

  def __init__(self, bits, k):
    """k is a parameter that determines how "sharp" the approximation is.
    The larger the value, the closer this is to the staircase function.
    k=2.5 is a reasonable value
    See https://arxiv.org/pdf/1908.05033.pdf.

    max_val determines the max absolute value of the representable range.
    I'd think that 3 would be a good value for gaussian distributed weights with
    mean 0 and standard deviation 1.
    """

    self.bits = bits
    self._k = k
    self.max_val = None


  @property
  def k(self):
    return self._k / self.max_val

  def set_shape(self, shape):
    """set the shape of the input data"""

    max_val = get_he_uniform_max_val(shape)
    self.max_val = max_val

  def __call__(self, x):

    delta = (2.0 * self.max_val) / (2.0 ** self.bits - 1)

    i = jnp.floor((x + self.max_val) / delta)
    m_i = -self.max_val + (i + 0.5) * delta
    s = 1.0 / jnp.tanh(0.5 * self.k * delta)
    phi = s * (jnp.tanh(self.k * (x - m_i)))

    phi = phi * delta / 2.0

    quantized_estimate = jnp.where(x < - self.max_val, - self.max_val, phi)
    quantized_estimate = jnp.where(x > self.max_val, self.max_val, quantized_estimate)


    clipped = jnp.clip(x, -self.max_val, self.max_val)
    quantized = delta * jnp.round((clipped + self.max_val) / delta) - self.max_val

    return quantized_estimate + lax.stop_gradient(quantized - quantized_estimate)

