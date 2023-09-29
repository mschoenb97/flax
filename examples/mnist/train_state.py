"""Custom train state with functionality of original keras callbacks"""

from typing import Any, Callable

from flax.training import train_state
from jax.tree_util import tree_map_with_path
from jax import numpy as jnp
from flax import struct, core
import jax
from functools import partial, wraps


def conv_path_only(func):
  @wraps(func)
  def wrapper(path, *args, **kwargs):
    if 'Conv' in path[0].key:
      return func(*args, **kwargs)
    else:
      return None
  return wrapper

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


get_change_point_data = conv_path_only(_get_change_point_data)

@conv_path_only
def init_change_points(weights, *, quantizer):
  """Set up initial change point data stractures. use `tree_map_with_path`"""
  qweights = quantizer(weights)
  points_changed = jnp.ones_like(weights)
  return _get_change_point_data(points_changed, qweights, total_batches=0)

init_change_points = jax.jit(
  init_change_points, 
  static_argnums=(0,), 
  static_argnames=('quantizer',))

@conv_path_only
def get_points_changed_tensor(new_q_tensor, old_q_tensor):

  return jnp.logical_not(jnp.isclose(new_q_tensor, old_q_tensor))

@conv_path_only
def get_quantized(weights, *, quantizer):
  return quantizer(weights)

get_quantized = jax.jit(
  get_quantized, 
  static_argnums=(0,), 
  static_argnames=('quantizer',))

@conv_path_only
def append(x0, x1):
  return jnp.concatenate((x0, x1), axis=0)

class CustomTrainState(train_state.TrainState):

  quantizer: Callable[[Array], Array]
  change_points: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
  last_quantized: core.FrozenDict[str, Any] = struct.field(pytree_node=True)

  total_batches: int = 0

  def update_change_points(self):
    """Call this after gradient updates have been applied"""

    partial_get_quantized = partial(get_quantized, quantizer=self.quantizer)
    partial_get_change_point_data = partial(get_change_point_data, total_batches=self.total_batches)

    new_quantized = tree_map_with_path(partial_get_quantized, self.params)
    points_changed = tree_map_with_path(get_points_changed_tensor, new_quantized, self.last_quantized)

    new_change_points = tree_map_with_path(partial_get_change_point_data, points_changed, new_quantized)
    updated_change_points = tree_map_with_path(append, self.change_points, new_change_points)

    return self.replace(
      last_quantized=new_quantized,
      total_batches=self.total_batches + 1,
      change_points=updated_change_points,
    )

  @classmethod
  def create(cls, *, apply_fn, params, tx, quantizer, **kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    opt_state = tx.init(params)

    partial_init_change_points = partial(init_change_points, quantizer=quantizer)
    partial_get_quantized = partial(get_quantized, quantizer=quantizer)
    
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
        quantizer=quantizer,
        change_points=tree_map_with_path(partial_init_change_points, params),
        last_quantized=tree_map_with_path(partial_get_quantized, params),
        **kwargs,
    )




