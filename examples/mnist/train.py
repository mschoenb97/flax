# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MNIST example.

Library file which executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

# See issue #620.
# pytype: disable=wrong-keyword-args

from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import ml_collections
import numpy as np
import optax
import tensorflow_datasets as tfds
from typing import Optional, Callable, Any, Tuple

from tqdm import tqdm

import matts_imports
 
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?




class CNN(nn.Module):
  """A simple CNN model."""

  quantizer: Optional[Callable[[Array], Array]] = None
  kernel_init: Optional[Callable[[PRNGKey, Shape, Dtype], Array]] = None


  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3), 
                quantizer=self.quantizer, kernel_init=self.kernel_init)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3), 
                quantizer=self.quantizer, kernel_init=self.kernel_init)(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


@jax.jit
def apply_model(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(params):
    logits = state.apply_fn({'params': params}, images)
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy


def update_model(state, grads):
  return state.apply_batch_updates(grads=grads)

def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_accuracy = []

  for perm in tqdm(perms):
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]
    grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  state = state.apply_epoch_updates()
  return state, train_loss, train_accuracy


def get_datasets(test):
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.0
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.0

  if test:
    for key in train_ds.keys():
      train_ds[key] = train_ds[key][:len(train_ds[key]) // 20]
      test_ds[key] = test_ds[key][:len(test_ds[key]) // 20]
  return train_ds, test_ds


def create_train_state(rng, config):
  """Creates initial `TrainState`."""

  quantizer = matts_imports.get_quantizer_from_config(config)
  initializer = matts_imports.get_initializer_from_config(config)
  tx = matts_imports.get_optimizer_from_config(config)
  cnn = CNN(quantizer=quantizer, kernel_init=initializer)
  params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
  return matts_imports.CustomTrainState.create(apply_fn=cnn.apply, params=params, tx=tx, 
                                 quantizer=quantizer, epochs_interval=10)


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> matts_imports.CustomTrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The train state (which includes the `.params`).
  """

  train_ds, test_ds = get_datasets(config.test)
  rng = jax.random.key(0)

  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(dict(config))

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, config)

  for epoch in tqdm(range(1, config.num_epochs + 1)):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(
        state, train_ds, config.batch_size, input_rng
    )
    _, test_loss, test_accuracy = apply_model(
        state, test_ds['image'], test_ds['label']
    )

    logging.info(
        'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f,'
        ' test_accuracy: %.2f'
        % (
            epoch,
            train_loss,
            train_accuracy * 100,
            test_loss,
            test_accuracy * 100,
        )
    )

    summary_writer.scalar('train_loss', train_loss, epoch)
    summary_writer.scalar('train_accuracy', train_accuracy, epoch)
    summary_writer.scalar('test_loss', test_loss, epoch)
    summary_writer.scalar('test_accuracy', test_accuracy, epoch)
    state.update_history(
      train_loss, test_loss, train_accuracy, test_accuracy)

  state = state.add_final_logits(test_ds)

  summary_writer.flush()
  return state
