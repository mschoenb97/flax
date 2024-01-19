import ml_collections
from copy import deepcopy
from tqdm import tqdm
import os
import sys
import pickle
import random
from tensorflow import keras
from copy import deepcopy
import tensorflow.compat.v2 as tf
import math
import pandas as pd
from matplotlib import pyplot as plt
from pprint import pprint
import json
from scipy.stats import pearsonr
import numpy as np
import tensorflow as tf
import argparse
import importlib.util
from jax import tree_util
from jaxlib.xla_extension import ArrayImpl
from pprint import pprint

import jax.numpy as jnp

from train import train_and_evaluate
import matts_imports

# NUM_SAMPLES = 2 * 1280 #update this
NUM_SAMPLES = 60000

def train_model(config):

  pprint(config)

  res = {}
  
  state = train_and_evaluate(config, '/tmp/mnist')

  cleaned_change_points = {}

  for i, leaf in enumerate(tree_util.tree_leaves(state.change_points)):
    if leaf is not None:
      cleaned_change_points[i] = leaf

  res = {
      'history': state.history,
      'weights_init': state.initial_weights,
      'weights_final': tree_util.tree_map_with_path(matts_imports.conv_only, state.params),
      'change_points': cleaned_change_points,
      'total_batches': state.step,
      'stored_weights': state.stored_weights,
      'distance_traveled': state.distance_traveled,
      'stored_distances': state.stored_distances,
      'correct_output_values': state.correct_output_values,  # Add the correct output values
      'model_predictions': state.final_logits  # Add the model's predictions
  }

  return res



def train_corresponding_models(*, epochs, optimizer, binary,
                               k, bits, steps_per_epoch, initial_learning_rate, 
                               warmup_target, warmup_steps,
                               decay_steps, warp_initialize, get_change_point_stats,
                               epochs_interval):
  """Get histories for both the Standard initializer with the tanh gradient and
  the warped initializer with the STE gradient"""

  config = ml_collections.ConfigDict()
  config.binary = binary
  config.bits = bits
  config.k = k
  config.warp_initialize = warp_initialize
  config.optimizer_type = optimizer
  config.initial_learning_rate = initial_learning_rate
  config.warmup_target = warmup_target
  config.warmup_steps = warmup_steps
  config.decay_steps = decay_steps
  config.num_epochs = epochs
  config.batch_size = NUM_SAMPLES // steps_per_epoch
  config.get_change_point_stats = get_change_point_stats
  config.epochs_interval = epochs_interval

  quantizer_warp_model_config = deepcopy(config)
  initializer_warp_model_config = deepcopy(config)

  quantizer_warp_model_config.quantizer_type = 'dsq'
  initializer_warp_model_config.quantizer_type = 'pwl'

  quantizer_warp_model_config.initializer_type = 'ste'
  initializer_warp_model_config.initializer_type = 'dsq' if warp_initialize else 'ste'

  # get quantizer warp model data
  quantizer_warp_data = train_model(quantizer_warp_model_config)
  # get initializer warp model data
  initializer_warp_data = train_model(initializer_warp_model_config)

  return quantizer_warp_data, initializer_warp_data


"""# Change point calculations"""


def remove_elements_dataframe(df):
  stack = []
  oscillation_count = []
  final_df_list = []

  for index, row in df.iterrows():
    stack.append(row)
    # Initialize oscillation_count for the new element
    oscillation_count.append(0)

    # Check if the last four elements on the stack form a palindrome based on 'qvalue'
    while len(stack) >= 4 and stack[-1]['qvalue'] == stack[-3]['qvalue'] and stack[-2]['qvalue'] == stack[-4]['qvalue']:
      # Increment the oscillation_count for the first element in the palindrome
      oscillation_count[-4] += 1

      # Remove the middle two elements from the stack
      last_elt = stack.pop()
      _ = stack.pop()
      _ = stack.pop()

      _ = oscillation_count.pop()
      _ = oscillation_count.pop()
      _ = oscillation_count.pop()

      # Reset the oscillation_count for the new last element
      stack.append(last_elt)
      oscillation_count.append(0)

  # Add the remaining elements and their oscillation_count to the final list
  for row, count in zip(stack, oscillation_count):
    final_row = row.copy()
    final_row['oscillation_count'] = count
    final_df_list.append(final_row)

  # Create the final DataFrame
  final_df = pd.DataFrame(final_df_list)
  return final_df


def organize_change_point_data(change_points):

  organized_change_points = {}

  for i, change_point_data in (change_points.items()):
    organized_change_points[i] = {}

    aggregated = change_point_data
    shape = aggregated.shape
    length = shape[-1]

    coordinate_cols = [f'c{i}' for i in range(length - 2)]
    columns = coordinate_cols + ['qvalue', 'step_count']

    aggregated = pd.DataFrame(aggregated, columns=columns)

    organized_change_points[i] = aggregated

  return organized_change_points

def get_min_delta(max_val_tree, quantizer):

  min_delta = float('inf')

  for max_val in tree_util.tree_leaves(max_val_tree):
    delta = quantizer.delta(max_val)
    min_delta = min(delta, min_delta)

  return min_delta

def compare_change_point_data(quantizer_warp_cps, initializer_warp_cps, 
                              max_val_tree, quantizer):

  quantizer_warp_cps = organize_change_point_data(quantizer_warp_cps)
  initializer_warp_cps = organize_change_point_data(initializer_warp_cps)

  assert quantizer_warp_cps.keys() == initializer_warp_cps.keys()


  min_delta = get_min_delta(max_val_tree, quantizer)

  total_weights = 0
  correct_weight_sequences = 0
  total_q_oscillation = 0
  total_i_oscillation = 0
  total_oscillation_error = 0
  total_step_count_error = 0
  total_compressed_change_points = 0

  exact_correct_weight_sequences = 0
  total_exact_step_count_error = 0
  total_exact_change_points = 0

  for (i, qcps), (_, icps) in tqdm(list(zip(quantizer_warp_cps.items(), initializer_warp_cps.items()))):

    assert qcps.shape[-1] == icps.shape[-1]
    length = qcps.shape[-1]
    coordinate_cols = [f'c{i}' for i in range(length - 2)]

    pd.testing.assert_frame_equal(
        qcps[coordinate_cols].drop_duplicates(),
        icps[coordinate_cols].drop_duplicates()
    )

    for (qcols, qsequence), (icols, isequence) in (zip(qcps.groupby(coordinate_cols), icps.groupby(coordinate_cols))):

      assert qcols == icols
      total_weights += 1

      compressed_qsequence = remove_elements_dataframe(
          qsequence).reset_index(drop=True)
      compressed_isequence = remove_elements_dataframe(
          isequence).reset_index(drop=True)

      correct_sequence = (
          (compressed_qsequence.shape == compressed_isequence.shape)
          and np.all(np.abs(compressed_qsequence['qvalue'] - compressed_isequence['qvalue']) < min_delta)
      )

      exact_correct_sequence = (
          (qsequence.shape == isequence.shape)
          and np.all(np.abs(qsequence['qvalue'] - isequence['qvalue']) < min_delta)
      )

      if correct_sequence:
        correct_weight_sequences += 1
        total_q_oscillation += compressed_qsequence['oscillation_count'].sum()
        total_i_oscillation += compressed_isequence['oscillation_count'].sum()
        total_step_count_error += np.abs(
            compressed_qsequence['step_count'] - compressed_isequence['step_count']).sum()
        total_oscillation_error += np.abs(
            compressed_qsequence['oscillation_count'] - compressed_isequence['oscillation_count']).sum()
        assert len(compressed_qsequence) > 0
        total_compressed_change_points += (len(compressed_qsequence) - 1)

      if exact_correct_sequence:
        exact_correct_weight_sequences += 1
        total_exact_step_count_error += np.abs(
            qsequence['step_count'] - isequence['step_count']).sum()
        assert len(qsequence) > 0
        total_exact_change_points += (len(qsequence) - 1)

  result_dict = {
      'total_weights': total_weights,
      'correct_weight_sequences': correct_weight_sequences,
      'total_q_oscillation': total_q_oscillation,
      'total_i_oscillation': total_i_oscillation,
      'total_oscillation_error': total_oscillation_error,
      'total_step_count_error': total_step_count_error,
      'total_compressed_change_points': total_compressed_change_points,
      'exact_correct_weight_sequences': exact_correct_weight_sequences,
      'total_exact_step_count_error': total_exact_step_count_error,
      'total_exact_change_points': total_exact_change_points,
    }

  return result_dict


def sanitize_filename(s):
  return "".join(c for c in s if c.isalnum() or c in (' ', '.', '_')).rstrip()


def generate_filename(func, *args, **kwargs):

  elts = []

  # Include the function name
  elts.append(func.__name__)

  # Include positional arguments
  for arg in args:
    elts.append(str(arg))

  # Include keyword arguments
  for k, v in kwargs.items():
    elts.append(str(k))
    elts.append(str(v))

  st = sanitize_filename(''.join(elts))
  filename_chars = list(st)
  seed = sum(ord(ch) for ch in filename_chars)
  random.seed(seed)
  random.shuffle(filename_chars)
  filename = ''.join(filename_chars[:50])

  # return os.path.join('data', sanitize_filename(m.hexdigest() + ".pkl"))
  return sanitize_filename(filename + ".pkl")


def save_or_load_output(func, *args, actual_args=None,
                        actual_kwargs=None, path=None, **kwargs):
  filename = generate_filename(func, *args, **kwargs)

  if actual_args is not None:
    args = actual_args
  if actual_kwargs is not None:
    kwargs = actual_kwargs

  if path is not None:
    filename = os.path.join(path, filename)

  print(filename)

  # Check if the file already exists
  if os.path.exists(filename):
    # Load and return the precomputed output
    with open(filename, 'rb') as f:
      return pickle.load(f)
  else:
    # Compute the output
    output = func(*args, **kwargs)

    # Save the output to a file
    with open(filename, 'wb') as f:
      pickle.dump(output, f)

    return output


def get_inference_results(quantizer_warp_data, initializer_warp_data):
  res = {}

  tf.debugging.assert_equal(
      quantizer_warp_data['correct_output_values'], initializer_warp_data['correct_output_values'])

  qpredictions = np.argmax(quantizer_warp_data['model_predictions'], axis=1)
  ipredictions = np.argmax(initializer_warp_data['model_predictions'], axis=1)

  res['quantizer_model_accuracy'] = (
      qpredictions == initializer_warp_data['correct_output_values']).mean()
  res['initializer_model_accuracy'] = (
      ipredictions == initializer_warp_data['correct_output_values']).mean()

  inference_agreement_total = (qpredictions == ipredictions).sum()
  res['inference_agreement_proportion'] = inference_agreement_total / \
      len(qpredictions)

  incorrect_inference_agreement_total = ((qpredictions == ipredictions)
                                         & (ipredictions != quantizer_warp_data['correct_output_values'])
                                         & (qpredictions != quantizer_warp_data['correct_output_values'])
                                         ).sum()
  res['incorrect_inference_agreement_proportion'] = incorrect_inference_agreement_total / \
      ((ipredictions != quantizer_warp_data['correct_output_values']) & (
          qpredictions != quantizer_warp_data['correct_output_values'])).sum()

  res['logit_mse'] = np.sum((quantizer_warp_data['model_predictions'] -
                            initializer_warp_data['model_predictions']) ** 2) / quantizer_warp_data['model_predictions'].size

  res['avg_logit_difference'] = np.sum(np.abs(quantizer_warp_data['model_predictions'] -
                                       initializer_warp_data['model_predictions'])) / quantizer_warp_data['model_predictions'].size

  return res


def plot_data(history1, history2, metric, label1, label2, path, 
  linewidth=0.5, offset=0, name=""):
  # Create subplots
  _, axs = plt.subplots(1, 2, figsize=(12, 4))

  prettify = {
    'loss': 'Loss',
    'val_loss': 'Validation Loss',
    'val_accuracy': 'Validation Accuracy',
  }

  # Plot raw data
  axs[0].plot(history1[metric][offset:], label=label1,
              linewidth=linewidth, color='purple', alpha=0.7)
  axs[0].plot(history2[metric][offset:], label=label2,
              linewidth=linewidth, color='red', linestyle='--', alpha=0.7)
  axs[0].set_title(f'{prettify[metric]} (Raw Data)')
  axs[0].set_ylabel(prettify[metric])
  axs[0].set_xlabel('Epoch')
  axs[0].legend(loc='upper left')

  # Plot difference between curves
  difference = [a - b for a,
                b in zip(history1[metric][offset:], history2[metric][offset:])]
  axs[1].plot(difference, label=f'{label1} - {label2}',
              linewidth=linewidth, color='blue', alpha=0.7)

  # Add line at zero
  axs[1].axhline(0, color='grey', linestyle='--', linewidth=linewidth)

  axs[1].set_title(f'{prettify[metric]} (Difference)')
  axs[1].set_ylabel('Difference')
  axs[1].set_xlabel('Epoch')
  axs[1].legend(loc='upper left')

  # Save the plot
  plt.savefig(os.path.join(path, f"{name}_{metric}.png"))

  # Show correlation
  correlation, _ = pearsonr(
      history1[metric][offset:], history2[metric][offset:])
  return {f"{metric}_correlation": correlation}


def get_history_data(quantizer_warp_history, initializer_warp_history, name, *, path):
  offset = 0
  linewidth = 0.5
  res = {}
  res.update(plot_data(quantizer_warp_history, initializer_warp_history, 'loss',
                       '$\hat Q$ Model Train Loss', 'STE Model Train Loss', 
                       path, linewidth, offset, name))

  res.update(plot_data(quantizer_warp_history, initializer_warp_history, 'val_loss',
                       '$\hat Q$ Model Test Loss', 'STE Model Test Loss', 
                       path, linewidth, offset, name))

  res.update(plot_data(quantizer_warp_history, initializer_warp_history, 'val_accuracy',
                       '$\hat Q$ Model Test Accuracy', 'STE Model Test Accuracy', 
                       path, linewidth, offset, name))

  return res


def get_change_point_results(change_point_res, quantizer_warp_data):

  res = {}

  res['correct_sequences_proportion'] = change_point_res['correct_weight_sequences'] / \
      change_point_res['total_weights']
  res['total_q_oscillation'] = change_point_res["total_q_oscillation"]
  res['total_i_oscillation'] = change_point_res["total_i_oscillation"]
  res['average_oscillation_error'] = change_point_res['total_oscillation_error'] / \
      change_point_res['total_compressed_change_points']
  res['average_step_count_error'] = change_point_res['total_step_count_error'] / \
      change_point_res['total_compressed_change_points']
  res['total_steps'] = quantizer_warp_data['total_batches']

  res['exact_correct_sequences_proportion'] = change_point_res['exact_correct_weight_sequences'] / \
      change_point_res['total_weights']
  res['average_exact_step_count_error'] = change_point_res['total_exact_step_count_error'] / \
      change_point_res['total_exact_change_points']

  return res


def sum_tree_leaves(tree):
    leaves = tree_util.tree_leaves(tree)
    return sum(leaf for leaf in leaves if leaf is not None)

def compute_distance_metric(
    qstored_weights, istored_weights, istored_distances, initializer, quantizer):

    assert qstored_weights.keys() == istored_weights.keys() == istored_distances.keys()

    # Process the results to get distances and quantized_agreements
    distances = {}
    quantized_agreements = {}
    for key in istored_distances.keys():

      distances[key], quantized_agreements[key] = compute_distance_metric_for_tree(
        qstored_weights[key], istored_weights[key], istored_distances[key],
        initializer, quantizer,
      )

    return distances, quantized_agreements


def compute_distance_metric_for_tree(
    qstored_weights_tree, istored_weights_tree, istored_distance, initializer, quantizer):


    @matts_imports.conv_path_only(jit_compile=False)
    def compute_total_sum(qstored_weight, istored_weight):
      max_val = matts_imports.get_he_uniform_max_val(qstored_weight.shape)
      return jnp.sum(jnp.abs(initializer.remap(qstored_weight) - istored_weight)) / max_val

    @matts_imports.conv_path_only(jit_compile=False)
    def compute_total_weights(qstored_weight):
      return qstored_weight.size
    
    @matts_imports.conv_path_only(jit_compile=False)
    def compute_total_qweight_agreements(qstored_weight, istored_weight):

      max_val = matts_imports.get_he_uniform_max_val(qstored_weight.shape)

      qq = quantizer(qstored_weight)
      iq = quantizer(istored_weight)
      close = jnp.abs(qq - iq) < (quantizer.delta(max_val) / 2)

      return jnp.sum(close, dtype=jnp.float32)

    total_sum = tree_util.tree_map_with_path(
      compute_total_sum, qstored_weights_tree, istored_weights_tree)
    total_weights = tree_util.tree_map_with_path(
      compute_total_weights, qstored_weights_tree)
    total_qweight_agreements = tree_util.tree_map_with_path(
      compute_total_qweight_agreements, qstored_weights_tree, istored_weights_tree)

    distance = (sum_tree_leaves(total_sum) / istored_distance).item()
    quantized_agreements = (sum_tree_leaves(total_qweight_agreements) / 
                                  sum_tree_leaves(total_weights)).item()

    return distance, quantized_agreements


def get_initializer(identifier_kwargs):

  config = deepcopy(identifier_kwargs)

  config['initializer_type'] = 'dsq' if config['warp_initialize'] else 'ste'

  return matts_imports.get_initializer_from_config(config)

def _get_quantizer(identifier_kwargs):

  config = deepcopy(identifier_kwargs)
  config['quantizer_type'] = 'dsq'
  return matts_imports.get_quantizer_from_config(config)


def get_distance_metric(quantizer_warp_data, initializer_warp_data, identifier_kwargs):

  warp_initializer = get_initializer(identifier_kwargs)
  quantizer = _get_quantizer(identifier_kwargs)
  distances, quantized_agreements = compute_distance_metric(
      quantizer_warp_data['stored_weights'],
      initializer_warp_data['stored_weights'],
      initializer_warp_data['stored_distances'],
      warp_initializer,
      quantizer
  )

  return {'distance_metric': distances, 'quantized_agreements': quantized_agreements}


def get_flattened_latent_weights(weights, initializer=None):

  weight_ls = []
  for i, weight in enumerate(tree_util.tree_leaves(weights)):
    if weight is not None:
      if initializer is not None:
        weight = initializer.remap(weight)
      weight = weight / matts_imports.get_he_uniform_max_val(weight.shape)
      weight_ls.append(np.array(weight).flatten())

  return np.concatenate(weight_ls)


def plot_weight_alignment_and_movement(
    quantizer_warp_data, initializer_warp_data, identifier_kwargs, *, path, name=""):
  res = {}
  warp_initializer = get_initializer(identifier_kwargs)
  s = 1

  plt.clf()
  # Create the Weight Alignment plot
  plt.scatter(
      get_flattened_latent_weights(quantizer_warp_data['weights_final']),
      get_flattened_latent_weights(initializer_warp_data['weights_final']),
      label='Raw weights', s=s
  )
  plt.scatter(
      get_flattened_latent_weights(
          quantizer_warp_data['weights_final'], initializer=warp_initializer),
      get_flattened_latent_weights(initializer_warp_data['weights_final']),
      label='$I_{\\hat Q}$(weights)', s=s
  )
  plt.xlabel('$\\hat Q$ model weights')
  plt.ylabel('STE model weights')
  plt.title("Weight Alignment")
  plt.legend()

  # Save the Weight Alignment plot
  plt.savefig(os.path.join(path, f"{name}_Weight_Alignment.png"))
  plt.clf()

  # Create the Weight Movement plot
  plt.scatter(
      get_flattened_latent_weights(initializer_warp_data['weights_init']),
      get_flattened_latent_weights(initializer_warp_data['weights_final']), s=s
  )
  plt.xlabel('Initial STE Model weights')
  plt.ylabel('Final  STE Model weights')
  plt.title("Weight Movement")

  # Save the Weight Movement plot
  plt.savefig(os.path.join(path, f"{name}_Weight_Movement.png"))
  plt.clf()

  # Assuming you want to return some results, you can populate the 'res' dictionary here
  # For example, you might want to calculate and return some statistics about the weights
  # res['some_statistic'] = calculate_some_statistic(quantizer_warp_data, initializer_warp_data)

  return res


def convert_all_float32_to_float(d):
  if isinstance(d, dict):
    return {k: convert_all_float32_to_float(v) for k, v in d.items()}
  elif isinstance(d, list):
    return [convert_all_float32_to_float(v) for v in d]
  elif isinstance(d, np.float32):
    return float(d)
  else:
    return d



def run_analysis_for_one(quantizer_warp_data, initializer_warp_data,
                         identifier_kwargs, name, path, cache_data=True):

  res = {}

  warp_initializer = get_initializer(identifier_kwargs)
  quantizer = _get_quantizer(identifier_kwargs)
  init_distance, init_agreement = compute_distance_metric_for_tree(
    quantizer_warp_data['weights_init'],
    initializer_warp_data['weights_init'],
    1,
    warp_initializer,
    quantizer, 
  )

  assert init_distance == 0, init_distance
  assert init_agreement == 1.0, init_agreement

  @matts_imports.conv_path_only(jit_compile=False)
  def conv_max_val(weights):

    return matts_imports.get_he_uniform_max_val(weights.shape)
  
  max_val_tree = tree_util.tree_map_with_path(
    conv_max_val, quantizer_warp_data['weights_init'])
  quantizer = _get_quantizer(identifier_kwargs)

  if cache_data:
    actual_args = [
        quantizer_warp_data['change_points'],
        initializer_warp_data['change_points'],
        max_val_tree,
        quantizer,
    ]
    actual_kwargs = {}
    change_point_res = save_or_load_output(
        compare_change_point_data, **identifier_kwargs, actual_args=actual_args,
        actual_kwargs=actual_kwargs, path=path
    )
  else:
    change_point_res = compare_change_point_data(
        quantizer_warp_data, initializer_warp_data, max_val_tree, quantizer)

    change_point_results = get_change_point_results(
        change_point_res, quantizer_warp_data)
    res.update(change_point_results)
    
  inference_results = get_inference_results(
      quantizer_warp_data, initializer_warp_data)
  res.update(inference_results)

  history_data = get_history_data(
      quantizer_warp_data['history'], initializer_warp_data['history'], name=name, path=path)
  res.update(history_data)

  distance_metric = get_distance_metric(
      quantizer_warp_data, initializer_warp_data, identifier_kwargs)
  res.update(distance_metric)

  plot_weight_alignment_and_movement(
      quantizer_warp_data, initializer_warp_data, identifier_kwargs, name=name, path=path)

  return res


def get_default_kwargs(config):

  default_kwargs = {
      'epochs': config['sgd_epochs'],
      'optimizer': 'sgd',
      'binary': False,
      'warp_initialize': True,
      'k': config['bit_to_k_map'][config['default_bits']],
      'bits': config['default_bits'],
      'steps_per_epoch': config['steps_per_epoch'],
      # 'weight_decay': config['weight_decay'],
      'initial_learning_rate': 0.0,
      'warmup_target': config['sgd_lr'],
      'warmup_steps': config['steps_per_epoch'] * config['warmup_proportion'] * config['sgd_epochs'],
      'decay_steps': config['steps_per_epoch'] * (1 - config['warmup_proportion']) * config['sgd_epochs']
  }

  return default_kwargs


def get_train_kwargs(config, optimizer_type, jitter=False, scaledown=False):
  if optimizer_type == 'sgd':
    lr = config['sgd_lr']
    epochs = config['sgd_epochs']
    optimizer = 'sgd'
    warp_initialize = True
  elif optimizer_type == 'adam':
    lr = config['adam_lr']
    epochs = config['adam_epochs']
    optimizer = 'adam'
    warp_initialize = False
  else:
    raise ValueError("Invalid optimizer_type. Must be either 'sgd' or 'adam'.")

  # weight_decay = config['weight_decay']

  if jitter:
    lr = lr * (1.0 + config['lr_jitter_scale'])
    warp_initialize = False
  if scaledown:
    lr = lr * config['lr_scaledown']
    epochs *= config['epoch_scale_up_for_lr_scale_down']
    # weight_decay *= config['lr_scaledown']

  updates = {
      'epochs': epochs,
      'initial_learning_rate': 0.0,
      'warmup_target': lr,
      'warmup_steps': config['steps_per_epoch'] * config['warmup_proportion'] * epochs,
      'decay_steps': config['steps_per_epoch'] * (1 - config['warmup_proportion']) * epochs,
      'optimizer': optimizer,
      'warp_initialize': warp_initialize,
      'get_change_point_stats': config['get_change_point_stats'],
      'epochs_interval': config['epochs_interval'],
      # 'weight_decay': weight_decay,
  }

  default_kwargs = get_default_kwargs(config)

  kwargs = default_kwargs.copy()
  kwargs.update(updates)

  return kwargs

def get_quantizer_kwargs(config, bits):

  updates = {
      'k': config['bit_to_k_map'][bits],
      'bits': bits,
  }

  default_kwargs = get_default_kwargs(config)

  kwargs = default_kwargs.copy()
  kwargs.update(updates)

  return kwargs


def run_models_from_kwargs(kwargs, config):

  if config['cache_data']:
    quantizer_warp_data, initializer_warp_data = save_or_load_output(
        train_corresponding_models, **kwargs, path=config['path']
    )
  else:
    quantizer_warp_data, initializer_warp_data = train_corresponding_models(**kwargs)

  return quantizer_warp_data, initializer_warp_data


def jax_to_numpy(obj):
    """
    Recursively convert jaxlib.xla_extension.ArrayImpl objects to numpy arrays in a JSON-like object.
    """
    if isinstance(obj, ArrayImpl) and obj.shape == ():
        # Direct conversion of JAX array to NumPy array
        return float(obj)
    elif isinstance(obj, dict):
        # Recursively apply to dictionary values
        return {k: jax_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively apply to list elements
        return [jax_to_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        # Recursively apply to tuple elements
        return tuple(jax_to_numpy(v) for v in obj)
    else:
        # Return the object as is if it's not a type that needs conversion
        return obj


def run_full_analysis(config):

  if not os.path.exists(config['path']):
    # make directory
    os.makedirs(config['path'])

  res = {}
  # optimizer_types = ['sgd', 'adam']
  # optimizer_types = ['sgd']
  optimizer_types = ['adam']

  # run models for both optimizers
  for opt_type in optimizer_types:
    kwargs = get_train_kwargs(config, opt_type)
    quantizer_warp_data, initializer_warp_data = run_models_from_kwargs(
        kwargs, config)
    results = run_analysis_for_one(
        quantizer_warp_data,
        initializer_warp_data,
        kwargs,
        opt_type,
        cache_data=config['cache_data'],
        path=config['path'])
    res[opt_type] = results

  #   jitter_kwargs = get_train_kwargs(config, opt_type, jitter=True)
  #   quantizer_warp_data_jitter, _ = run_models_from_kwargs(jitter_kwargs, config)
  #   jitter_results = run_analysis_for_one(
  #       quantizer_warp_data,
  #       quantizer_warp_data_jitter,
  #       jitter_kwargs,
  #       f"{opt_type}_jitter",
  #       cache_data=config['cache_data'],
  #       path=config['path'])
  #   res[f"{opt_type}_jitter"] = jitter_results

  #   scaledown_kwargs = get_train_kwargs(config, opt_type, scaledown=True)
  #   quantizer_warp_data_scaledown, initializer_warp_data_scaledown = run_models_from_kwargs(
  #       scaledown_kwargs, config)
  #   scaledown_results = run_analysis_for_one(
  #       quantizer_warp_data_scaledown,
  #       initializer_warp_data_scaledown,
  #       scaledown_kwargs,
  #       f"{opt_type}_scaledown",
  #       cache_data=config['cache_data'],
  #       path=config['path'])
  #   res[f"{opt_type}_scaledown"] = scaledown_results

  # # run without warp_initialize for sgd
  # no_warp_kwargs = get_train_kwargs(config, 'sgd')
  # no_warp_kwargs['warp_initialize'] = False
  # quantizer_warp_data, initializer_warp_data = run_models_from_kwargs(
  #       no_warp_kwargs, config)
  # no_warp_results = run_analysis_for_one(
  #     quantizer_warp_data,
  #     initializer_warp_data,
  #     no_warp_kwargs,
  #     'sgd_no_warp',
  #     cache_data=config['cache_data'],
  #     path=config['path'])
  # res['sgd_no_warp'] = no_warp_results

  # # run models for other bits
  # for bits in config['other_bits']:
  #   bits_kwargs = get_quantizer_kwargs(config, bits)
  #   quantizer_warp_data, initializer_warp_data = run_models_from_kwargs(
  #       bits_kwargs, config)
  #   bits_results = run_analysis_for_one(
  #       quantizer_warp_data,
  #       initializer_warp_data,
  #       bits_kwargs,
  #       f"{bits}_bits",
  #       cache_data=config['cache_data'],
  #       path=config['path'])

  #   res[f"{bits}_bits"] = bits_results

  res = convert_all_float32_to_float(res)
  res = jax_to_numpy(res)

  with open(os.path.join(config['path'], 'results.json'), 'w') as f:
    json.dump(res, f)

  return res

def read_config_from_directory(directory, config_filename='config.py'):
  config_path = os.path.join(directory, config_filename)
  
  if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found in directory {directory}. Please make sure '{config_filename}' exists.")
  
  # Add the directory to sys.path
  sys.path.append(directory)
  
  # Dynamically import the config module
  spec = importlib.util.spec_from_file_location("config", config_path)
  config_module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(config_module)
  
  # Remove the directory from sys.path
  sys.path.remove(directory)
  
  config = config_module.config
  
  # Add the path to the config dictionary
  config['path'] = directory
  
  return config

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run analysis with config from a directory.')
  parser.add_argument('directory', type=str, help='Directory where config.py is located.')
  
  args = parser.parse_args()
  
  try:
    config = read_config_from_directory(args.directory)
    res = run_full_analysis(config)
    pprint(res)
  except FileNotFoundError as e:
    print(e)
    print("Please provide a directory containing 'config.py'.")

