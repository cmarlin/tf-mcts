# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Tests for `qtransforms.py`."""
from absl.testing import absltest
import tensorflow as tf
import numpy as np
import os
from tf_mcts._src import qtransforms


class QtransformsTest(absltest.TestCase):

  def setUp(self):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

  def test_mix_value(self):
    """Tests the output of _compute_mixed_value()."""
    raw_value = tf.convert_to_tensor(-0.8)
    prior_logits = tf.convert_to_tensor(
      [-float("inf"), -1.0, 2.0, -float("inf")]
      )
    probs = tf.nn.softmax(prior_logits)
    visit_counts = tf.convert_to_tensor([0, 4.0, 4.0, 0])
    qvalues = 10.0 / 54 * tf.convert_to_tensor([20.0, 3.0, -1.0, 10.0])
    mix_value = qtransforms._compute_mixed_value(
        raw_value, qvalues, visit_counts, probs)

    num_simulations = tf.reduce_sum(visit_counts)
    expected_mix_value = 1.0 / (num_simulations + 1) * (
        raw_value + num_simulations *
        (probs[1] * qvalues[1] + probs[2] * qvalues[2]))
    np.testing.assert_allclose(expected_mix_value, mix_value)

  def test_mix_value_with_zero_visits(self):
    """Tests that zero visit counts do not divide by zero."""
    raw_value = tf.convert_to_tensor(-0.8)
    prior_logits = tf.convert_to_tensor(
      [-float("inf"), -1.0, 2.0, -float("inf")]
      )
    probs = tf.nn.softmax(prior_logits)
    visit_counts = tf.convert_to_tensor([0, 0, 0, 0])
    qvalues = tf.zeros_like(probs)
    mix_value = qtransforms._compute_mixed_value(
        raw_value, qvalues, visit_counts, probs)
    tf.debugging.assert_all_finite(mix_value, message="jax.debug_nans")

    np.testing.assert_allclose(raw_value, mix_value)


if __name__ == "__main__":
  absltest.main()
