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

"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.momentum = 0.9
  config.batch_size = 1280
  config.num_epochs = 10
  config.test = True
  config.quantizer_type = 'pwl'
  config.initializer_type = 'dsq'
  config.bits = 2
  config.k = 5.5
  config.optimizer_type = 'sgd'
  config.momentum = 0.9
  config.initial_learning_rate = 0.0
  config.warmup_target = 0.1
  config.warmup_steps = 0.0
  config.decay_steps = 1000
  return config


def metrics():
  return []
