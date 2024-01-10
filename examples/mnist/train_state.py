"""Custom train state with functionality of original keras callbacks"""

from typing import Any, Callable, List

from flax.training import train_state
from jax.tree_util import tree_map_with_path, tree_leaves
from jax import numpy as jnp
from flax import struct, core
import jax
from functools import partial, wraps
from dataclasses import field
import optax

from he_uniform import get_he_uniform_max_val



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

def _get_change_point_data(points_changed_tensor, qweights, *, total_batches):
  assert points_changed_tensor.shape == qweights.shape
  coords = jnp.argwhere(points_changed_tensor)
  coords_float = jnp.array(coords, dtype=jnp.float32)
  quantized_values = qweights[jnp.where(points_changed_tensor)]
  batch_tensor = jnp.full((coords.shape[0],), total_batches)
  batch_tensor_float = jnp.array(batch_tensor, dtype=jnp.float32)

  concat_input = [
      coords_float,
      jnp.expand_dims(quantized_values, axis=-1),
      jnp.expand_dims(batch_tensor_float, axis=-1),
  ]

  result_tensor = jnp.concatenate(concat_input, axis=1)
  return result_tensor

@conv_path_only()
def init_change_points(weights, *, quantizer):
  """Set up initial change point data stractures. use `tree_map_with_path`"""
  qweights = quantizer(weights)
  points_changed = jnp.ones_like(weights)
  return _get_change_point_data(points_changed, qweights, total_batches=0)


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

@conv_path_only()
def append(x0, x1):
  return jnp.concatenate((x0, x1), axis=0)

@conv_path_only(default_return=0.0)
def get_total_distance_leaf(prev_params, curr_params):

  max_val = get_he_uniform_max_val(prev_params.shape)

  return jnp.sum(jnp.abs(prev_params-curr_params) / max_val)





















# class StoreWeightsCallback(Callback):
#   """Tracks weights and weight distances. Ignores the last two layers, as they are unquantized"""

#   def __init__(self, epochs_interval):
#     super().__init__()
#     self.epochs_interval = epochs_interval
#     self.stored_weights = {}
#     self.stored_distances = {}  # To store the distance traveled at each epoch
#     self.distance_traveled = 0.0  # Initialize distance_traveled attribute as a scalar
#     self.prev_weights = None  # To store the previous weights for distance calculation

#   def on_train_begin(self, logs=None):
#     # Initialize distance_traveled as 0.0 at the beginning of training
#     self.distance_traveled = 0.0

#   def on_epoch_end(self, epoch, logs=None):
#     if (epoch + 1) % self.epochs_interval == 0:
#       self.stored_weights[epoch + 1] = self.model.get_weights()[:-2]
#       # Store the current distance_traveled
#       self.stored_distances[epoch + 1] = self.distance_traveled

#   def on_batch_end(self, batch, logs=None):
#     # Get the current weights
#     current_weights = self.model.get_weights()
#     # If prev_weights is None, initialize it with the current weights
#     if self.prev_weights is None:
#       self.prev_weights = current_weights
#     # Calculate the sum of absolute differences for all weights and update distance_traveled
#     for curr_w, prev_w in zip(current_weights[:-2], self.prev_weights[:-2]):
#       max_val = get_he_uniform_max_val(curr_w.shape)
#       self.distance_traveled += tf.reduce_sum(
#           tf.math.abs(curr_w - prev_w) / max_val)
#     # Update prev_weights with current_weights for the next batch
#     self.prev_weights = current_weights
























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
  # quantizer: Callable[[Array], Array]
  epochs_interval: int
  points_changed: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  last_quantized: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  quantized_vals_list: List[core.FrozenDict[str, Any]] = field(default_factory=list)
  points_changed_list: List[core.FrozenDict[str, Any]] = field(default_factory=list)
  stored_weights: dict = field(default_factory=dict)
  stored_distances: dict = field(default_factory=dict)

  distance_traveled: float = 0
  total_batches: int = 0
  epoch: int = 0


  def apply_epoch_updates(self):

    new_epoch = self.epoch + 1
    if (new_epoch) % self.epochs_interval == 0:
      self.stored_weights[new_epoch] = self.params
      # Store the current distance_traveled
      self.stored_distances[new_epoch] = self.distance_traveled

    return self.replace(epoch=self.epoch + 1)

  def update_change_points(self, quantizer):
    """Call this after gradient updates have been applied"""

    partial_get_quantized = partial(get_quantized, quantizer=quantizer)

    new_quantized = tree_map_with_path(partial_get_quantized, self.params)
    points_changed = tree_map_with_path(get_points_changed_tensor, new_quantized, self.last_quantized)

    self.quantized_vals_list.append(new_quantized)
    self.points_changed_list.append(points_changed)

    return self.replace(
      last_quantized=new_quantized,
      total_batches=self.total_batches + 1,
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
  
  def apply_batch_updates(self, *, grads, quantizer, **kwargs):

    self_with_grads = self.apply_gradients(grads=grads, **kwargs)
    self_with_distance = self_with_grads.update_distance()
    return self_with_distance.update_change_points(quantizer)

  @classmethod
  def create(cls, *, apply_fn, params, tx, quantizer, epochs_interval, **kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    opt_state = tx.init(params)

    partial_init_points_changed = partial(init_points_changed)
    partial_get_quantized = partial(get_quantized, quantizer=quantizer)
    
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        prev_params=params,
        tx=tx,
        opt_state=opt_state,
        points_changed=tree_map_with_path(partial_init_points_changed, params),
        last_quantized=tree_map_with_path(partial_get_quantized, params),
        epochs_interval=epochs_interval,
        **kwargs,
    )




