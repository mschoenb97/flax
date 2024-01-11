import math
from collections.abc import Sequence
from dataclasses import field
from functools import partial, wraps
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax import core, struct
from flax.struct import dataclass
from flax.training import train_state
from jax import lax, random
from jax._src import core as jax_core, dtypes
from jax._src.util import set_module
from jax.tree_util import tree_leaves, tree_map_with_path
from jax.nn.initializers import he_uniform
import optax

# Aliases and custom type definitions
KeyArray = jax.Array
Array = Any
DTypeLikeFloat = Any  # TODO: Import or define these to match Numpy's dtype_like definitions
DTypeLikeComplex = Any
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex
RealNumeric = Any  # Scalar jnp array or float

# He uniform logic

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


# Initializers

class ste_initializer:

  def __init__(self):

    self.init_func = he_uniform()

  def remap(self, x):

    return x

  def __call__(self, key: KeyArray,
                shape: jax_core.Shape,
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
                shape: jax_core.Shape,
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
    

# Optimizers


def get_optimizer_from_config(config):
    """
    Creates an optax optimizer based on the given configuration.

    Args:
        config (dict): Configuration parameters for the optimizer.

    Returns:
        An optax optimizer.
    """
    optimizer_type = config.get('optimizer_type')
    momentum = config.get('momentum', 0.9)  # Default momentum
    initial_lr = config.get('initial_learning_rate', 0.001)  # Default learning rate
    lr_warmup_target = config.get('warmup_target', initial_lr)
    warmup_steps = config.get('warmup_steps', 0)
    decay_steps = config.get('decay_steps', 1000)

    # Learning rate schedule
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=0.0, end_value=lr_warmup_target, transition_steps=warmup_steps),
            optax.cosine_decay_schedule(init_value=lr_warmup_target, decay_steps=decay_steps)
        ],
        boundaries=[warmup_steps]
    )

    # Selecting the optimizer
    if optimizer_type == 'sgd':
        return optax.chain(optax.trace(decay=momentum, nesterov=False), optax.scale_by_schedule(lr_schedule))
    elif optimizer_type == 'adam':
        return optax.chain(optax.adam(learning_rate=lr_schedule, b2=0.95))
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


# Quantizers

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
  

def get_quantizer_from_config(config):
  """
  Factory function to create a quantizer instance based on the provided configuration.

  Args:
      config: A configuration object or dict containing the quantizer type and its parameters.

  Returns:
      An instance of either pwl_multi_bit_quantizer or dsq_multi_bit_quantizer.
  """
  quantizer_type = config.get('quantizer_type')
  
  # Common parameters
  bits = config.get('bits')
  k = config.get('k')

  if quantizer_type == 'pwl':
    adjust_learning_rate = config.get('warp_initilize', False)
    return pwl_multi_bit_quantizer(bits=bits, k=k, adjust_learning_rate=adjust_learning_rate)
  elif quantizer_type == 'dsq':
    return dsq_multi_bit_quantizer(bits=bits, k=k)
  else:
    raise ValueError(f"Unknown quantizer type: {quantizer_type}")
  


def conv_path_only(jit_compile=True, default_return=None):
  def apply_conv_path_only(func):
    if jit_compile:
      func = jax.jit(func)
    @wraps(func)
    def wrapper(path, *args, **kwargs):
      if 'Conv' in path[0].key:
        return func(*args, **kwargs)
      else:
        return default_return
    return wrapper
  return apply_conv_path_only

Array = Any

def _get_change_point_data(points_changed_tensor, qweights, step):
  assert points_changed_tensor.shape == qweights.shape
  coords = jnp.argwhere(points_changed_tensor)
  coords_float = jnp.array(coords, dtype=jnp.float32)
  quantized_values = qweights[jnp.where(points_changed_tensor)]
  batch_tensor = jnp.full((coords.shape[0],), step)
  batch_tensor_float = jnp.array(batch_tensor, dtype=jnp.float32)

  concat_input = [
      coords_float,
      jnp.expand_dims(quantized_values, axis=-1),
      jnp.expand_dims(batch_tensor_float, axis=-1),
  ]

  result_tensor = jnp.concatenate(concat_input, axis=1)
  return result_tensor

get_change_point_data = conv_path_only(jit_compile=False)(_get_change_point_data)

@conv_path_only()
def init_change_points(weights, *, quantizer):
  """Set up initial change point data stractures. use `tree_map_with_path`"""
  qweights = quantizer(weights)
  points_changed = jnp.ones_like(weights)
  return _get_change_point_data(points_changed, qweights, 0)

@conv_path_only()
def conv_only(x):

  return x

@conv_path_only()
def init_points_changed(weights):
  """Set up initial change point data stractures. use `tree_map_with_path`"""
  points_changed = jnp.ones_like(weights)
  return points_changed

@conv_path_only()
def get_points_changed_tensor(new_q_tensor, old_q_tensor):

  return jnp.logical_not(jnp.isclose(new_q_tensor, old_q_tensor))

def _get_quantized(weights, *, quantizer):
  return quantizer(weights)

_get_quantized = jax.jit(
  _get_quantized, 
  static_argnames=('quantizer',))

get_quantized = conv_path_only(jit_compile=False)(_get_quantized)

@conv_path_only(default_return=0.0)
def get_total_distance_leaf(prev_params, curr_params):

  max_val = get_he_uniform_max_val(prev_params.shape)

  return jnp.sum(jnp.abs(prev_params-curr_params) / max_val)

@conv_path_only()
def conv_append(array0, array1):

  return jnp.append(array0, array1)


class CustomTrainState(struct.PyTreeNode):
  """Forked from flax/training/train_state.py"""

  # Original train_state fields
  step: int
  apply_fn: Callable = struct.field(pytree_node=False)
  params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  prev_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState = struct.field(pytree_node=True)
  # Quantization fields
  quantizer: struct.dataclass
  epochs_interval: int
  last_quantized: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  change_points: Optional[core.FrozenDict[str, Any]] = None
  history: dict = field(default_factory=dict)
  # quantized_vals_list: List[core.FrozenDict[str, Any]] = field(default_factory=list)
  # points_changed_list: List[core.FrozenDict[str, Any]] = field(default_factory=list)
  stored_weights: dict = field(default_factory=dict)
  stored_distances: dict = field(default_factory=dict)
  distance_traveled: float = 0
  epoch: int = 0
  final_logits: Optional[Array] = None
  correct_output_values: Optional[Array] = None
  initial_weights: Optional[Array] = None


  def apply_epoch_updates(self):

    new_epoch = self.epoch + 1
    if (new_epoch) % self.epochs_interval == 0:
      self.stored_weights[new_epoch] = tree_map_with_path(conv_only, self.params)
      # Store the current distance_traveled
      self.stored_distances[new_epoch] = self.distance_traveled

    return self.replace(epoch=self.epoch + 1)
  
  def update_history(self, train_loss, test_loss, train_accuracy, test_accuracy):

    if len(self.history) == 0:
      self.history['loss'] = []
      self.history['accuracy'] = []
      self.history['val_loss'] = []
      self.history['val_accuracy'] = []

    self.history['loss'].append(train_loss)
    self.history['accuracy'].append(train_accuracy)
    self.history['val_loss'].append(test_loss)
    self.history['val_accuracy'].append(test_accuracy)

  def add_final_logits(self, test_ds):

    logits = self.apply_fn({'params': self.params}, test_ds['image'])

    return self.replace(
      final_logits=logits,
      correct_output_values=test_ds['label'],
    )

  def update_change_points(self):
    """Call this after gradient updates have been applied"""

    partial_get_quantized = partial(get_quantized, quantizer=self.quantizer.__call__)
    partial_get_change_points = partial(get_change_point_data, step=self.step)

    new_quantized = tree_map_with_path(partial_get_quantized, self.params)
    points_changed = tree_map_with_path(get_points_changed_tensor, new_quantized, self.last_quantized)
    change_points = tree_map_with_path(
      partial_get_change_points, points_changed, new_quantized)
    
    if self.step == 1:
      new_change_points = change_points
    else:
      new_change_points = tree_map_with_path(conv_append, self.change_points, change_points)

    return self.replace(
      last_quantized=new_quantized,
      change_points=new_change_points,
    )

  def get_distance_traveled(self):

    leaf_distances = tree_map_with_path(get_total_distance_leaf, self.prev_params, self.params)
    return jnp.sum(jnp.array(tree_leaves(leaf_distances)))

  @jax.jit
  def update_distance(self):
    
    distance_traveled_update = self.get_distance_traveled()

    return self.replace(
      distance_traveled=self.distance_traveled + distance_traveled_update)

  @jax.jit
  def apply_gradients(self, *, grads, **kwargs):
    """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

    Note that internally this function calls `.tx.update()` followed by a call
    to `optax.apply_updates()` to update `params` and `opt_state`.

    Args:
      grads: Gradients that have the same pytree structure as `.params`.
      **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

    Returns:
      An updated instance of `self` with `step` incremented by one, `params`
      and `opt_state` updated by applying `grads`, and additional attributes
      replaced as specified by `kwargs`.
    """
    updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
    old_params = self.params
    new_params = optax.apply_updates(self.params, updates)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        prev_params=old_params,
        opt_state=new_opt_state,
        **kwargs,
    )
  
  def apply_batch_updates(self, *, grads, **kwargs):

    if self.step == 0:
      self = self.replace(initial_weights=tree_map_with_path(conv_only, self.params))
    self_with_grads = self.apply_gradients(grads=grads, **kwargs)
    self_with_distance = self_with_grads.update_distance()
    return self_with_distance.update_change_points()

  @classmethod
  def create(cls, *, apply_fn, params, tx, quantizer, epochs_interval, **kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    opt_state = tx.init(params)

    partial_get_quantized = partial(get_quantized, quantizer=quantizer)
    
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        prev_params=params,
        tx=tx,
        opt_state=opt_state,
        quantizer=quantizer,
        last_quantized=tree_map_with_path(partial_get_quantized, params),
        epochs_interval=epochs_interval,
        **kwargs,
    )

