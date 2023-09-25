



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

  # Using max val from HeUniform:
  # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeUniform
  return np.sqrt(6.0 / (fan_in))


class ste_binary_quantizer:

  def set_shape(self, _):
    pass

  @property
  def lr_adjustment(self):
    return 1.0

  @tf.custom_gradient
  def __call__(self, x):
    output = tf.where(x >= 0.0, tf.ones_like(x), -tf.ones_like(x))

    def grad(dy):
      return dy

    return output, grad


class tanh_binary_quantizer:

  def __init__(self, k):
    self._k = k  # how sharp the tanh function is

  @property
  def k(self):
    return self._k / self.max_val

  def set_shape(self, shape):
    """set the shape of the input data"""

    max_val = get_he_uniform_max_val(shape)
    self.max_val = max_val

  @tf.custom_gradient
  def __call__(self, x):
    output = tf.where(x >= 0.0, tf.ones_like(x), -tf.ones_like(x))

    def grad(dy):
      return dy * self.k * (1 - tf.tanh(self.k * x) ** 2)

    return output, grad


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

    s = 1 / tf.tanh(0.5 * self.k * self.delta)
    endpoint = self.k * self.delta / 2.0
    integral = (endpoint + tf.sinh(endpoint) * tf.cosh(endpoint)) / (s)
    integral = integral * 2.0 / (self.delta * (self.k ** 2))

    return integral

  @property
  def lr_adjustment(self):

    if self.adjust_learning_rate:
      return tf.keras.backend.cast_to_floatx(self.delta / self.interval_integral)
    else:
      return 1.0

  def set_shape(self, shape):
    """set the shape of the input data"""

    max_val = get_he_uniform_max_val(shape)
    self.max_val = max_val

  @tf.custom_gradient
  def __call__(self, x):

    def grad(dy):

      # absorbing lr adustment into gradient, since adjustment differs for different gradient estimators
      return self.lr_adjustment * dy * tf.where(tf.abs(x) < self.max_val, 1.0, 0.0)

    clipped = tf.clip_by_value(x, -self.max_val, self.max_val)
    quantized = self.delta * \
        tf.round((clipped + self.max_val) / self.delta) - self.max_val

    return quantized, grad


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

  @tf.custom_gradient
  def __call__(self, x):

    delta = (2.0 * self.max_val) / (2.0 ** self.bits - 1)

    def grad(dy):
      i = tf.math.floor((x + self.max_val) / delta)
      m_i = -self.max_val + (i + 0.5) * delta
      s = 1.0 / tf.tanh(0.5 * self.k * delta)
      s = tf.keras.backend.cast_to_floatx(s)
      phi = s * self.k * (1 - tf.tanh(self.k * (x - m_i)) ** 2.0)

      phi = tf.where(tf.logical_and(
          x <= self.max_val, x >= -self.max_val), phi, 0)

      phi = phi * delta / 2.0

      return dy * phi

    clipped = tf.clip_by_value(x, -self.max_val, self.max_val)
    quantized = delta * \
        tf.round((clipped + self.max_val) / delta) - self.max_val

    return quantized, grad
