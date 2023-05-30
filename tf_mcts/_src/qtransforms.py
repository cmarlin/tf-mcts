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
"""Monotonic transformations for the Q-values."""

import tensorflow as tf

from tf_mcts._src import tree as tree_lib


def qtransform_by_min_max(
    tree: tree_lib.Tree,
    node_index: tf.Tensor,
    *,
    min_value: tf.Tensor,
    max_value: tf.Tensor,
) -> tf.Tensor:
  """Returns Q-values normalized by the given `min_value` and `max_value`.

  Args:
    tree: [B, ...] MCTS tree state.
    node_index: scalar index of the parent node.
    min_value: given minimum value. Usually the `min_value` is minimum possible
      untransformed Q-value.
    max_value: given maximum value. Usually the `max_value` is maximum possible
      untransformed Q-value.

  Returns:
    Q-values normalized by `(qvalues - min_value) / (max_value - min_value)`.
    The unvisited actions will have zero Q-value. Shape `[num_actions]`.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  if tf.executing_eagerly():
    tf.debugging.assert_equal(tf.shape(node_index), (batch_size))
  qvalues = tree.qvalues(node_index)
  visit_counts = tf.gather(
    tree.children_visits, node_index, axis=1, batch_dims=1)
  value_score = tf.where(visit_counts > 0, qvalues, min_value)
  value_score = (value_score - min_value) / ((max_value - min_value))
  return value_score


def qtransform_by_parent_and_siblings(
    tree: tree_lib.Tree,
    node_index: tf.Tensor,
    *,
    epsilon: float = 1e-8,
) -> tf.Tensor:
  """Returns qvalues normalized by min, max over V(node) and qvalues.

  Args:
    tree: MCTS tree state. (JAX was _unbatched_)
    node_index: scalar index of the parent node.
    epsilon: the minimum denominator for the normalization.

  Returns:
    Q-values normalized to be from the [0, 1] interval. The unvisited actions
    will have zero Q-value. Shape `[num_actions]`.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  if tf.executing_eagerly():
    tf.debugging.assert_equal(tf.shape(node_index), (batch_size))
  qvalues = tree.qvalues(node_index)
  visit_counts = tf.gather(
    tree.children_visits, node_index, axis=1, batch_dims=1)
  tf.debugging.assert_rank(qvalues, 2)
  tf.debugging.assert_rank(visit_counts, 2)
  tf.debugging.assert_rank(node_index, 1)
  node_value = tf.gather(tree.node_values, node_index, axis=1, batch_dims=1)
  safe_qvalues = tf.where(visit_counts > 0, qvalues, node_value[..., None])
  if tf.executing_eagerly():
    tf.debugging.assert_equal(tf.shape(safe_qvalues), tf.shape(qvalues))
  min_value = tf.math.minimum(
    node_value,
    tf.reduce_min(safe_qvalues, axis=-1)
    )[..., None]
  max_value = tf.math.maximum(
    node_value,
    tf.reduce_max(safe_qvalues, axis=-1)
    )[..., None]

  completed_by_min = tf.where(visit_counts > 0, qvalues, min_value)
  normalized = (completed_by_min - min_value) / (
      tf.math.maximum(max_value - min_value, epsilon))
  if tf.executing_eagerly():
    tf.debugging.assert_equal(tf.shape(normalized), tf.shape(qvalues))
  return normalized


def qtransform_completed_by_mix_value(
    tree: tree_lib.Tree,
    node_index: tf.Tensor,
    *,
    value_scale: float = 0.1,
    maxvisit_init: float = 50.0,
    rescale_values: bool = True,
    use_mixed_value: bool = True,
    epsilon: float = 1e-8,
) -> tf.Tensor:
  """Returns completed qvalues.

  The missing Q-values of the unvisited actions are replaced by the
  mixed value, defined in Appendix D of
  "Policy improvement by planning with Gumbel":
  https://openreview.net/forum?id=bERaNdoegnO

  The Q-values are transformed by a linear transformation:
    `(maxvisit_init + max(visit_counts)) * value_scale * qvalues`.

  Args:
    tree: [B, ...] MCTS tree state.
    node_index: [B] scalar index of the parent node.
    value_scale: scale for the Q-values.
    maxvisit_init: offset to the `max(visit_counts)` in the scaling factor.
    rescale_values: if True, scale the qvalues by `1 / (max_q - min_q)`.
    use_mixed_value: if True, complete the Q-values with mixed value,
      otherwise complete the Q-values with the raw value.
    epsilon: the minimum denominator when using `rescale_values`.

  Returns:
    Completed Q-values. Shape `[num_actions]`.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(node_index), (batch_size))
  qvalues = tree.qvalues(node_index)
  visit_counts = tf.gather(
    tree.children_visits, node_index, axis=1, batch_dims=1)

  # Computing the mixed value and producing completed_qvalues.
  raw_value = tf.gather(tree.raw_values, node_index, axis=1, batch_dims=1)
  prior_probs = tf.nn.softmax(
      tf.gather(tree.children_prior_logits, node_index, axis=1, batch_dims=1))
  if use_mixed_value:
    value = _compute_mixed_value(
        raw_value,
        qvalues=qvalues,
        visit_counts=visit_counts,
        prior_probs=prior_probs)
  else:
    value = raw_value
  completed_qvalues = _complete_qvalues(
      qvalues, visit_counts=visit_counts, value=value)

  # Scaling the Q-values.
  if rescale_values:
    completed_qvalues = _rescale_qvalues(completed_qvalues, epsilon)
  maxvisit = tf.cast(
    tf.reduce_max(visit_counts, axis=-1),
    tf.float32
    )[..., None]
  visit_scale = maxvisit_init + maxvisit
  return visit_scale * value_scale * completed_qvalues


def _rescale_qvalues(qvalues, epsilon):
  """Rescales the given completed Q-values to be from the [0, 1] interval."""
  min_value = tf.reduce_min(qvalues, axis=-1, keepdims=True)
  max_value = tf.reduce_max(qvalues, axis=-1, keepdims=True)
  return (qvalues - min_value) / tf.math.maximum(max_value - min_value, epsilon)


def _complete_qvalues(qvalues, *, visit_counts, value):
  """Returns completed Q-values, with the `value` for unvisited actions."""
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(qvalues), tf.shape(visit_counts))
    tf.assert_equal(tf.shape(value), (tf.shape(qvalues)[0]))

  # The missing qvalues are replaced by the value.
  completed_qvalues = tf.where(
      visit_counts > 0,
      qvalues,
      value[..., None])
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(completed_qvalues), tf.shape(qvalues))
  return completed_qvalues


def _compute_mixed_value(raw_value, qvalues, visit_counts, prior_probs):
  """Interpolates the raw_value and weighted qvalues.

  Args:
    raw_value: an approximate value of the state. Shape `[]`.
    qvalues: Q-values for all actions. Shape `[num_actions]`. The unvisited
      actions have undefined Q-value.
    visit_counts: the visit counts for all actions. Shape `[num_actions]`.
    prior_probs: the action probabilities, produced by the policy network for
      each action. Shape `[num_actions]`.

  Returns:
    An estimator of the state value. Shape `[]`.
  """
  sum_visit_counts = tf.reduce_sum(visit_counts, axis=-1)
  # Ensuring non-nan weighted_q, even if the visited actions have zero
  # prior probability.
  prior_probs = tf.math.maximum(
    tf.experimental.numpy.finfo(prior_probs.dtype).tiny,
    prior_probs
    )
  # Summing the probabilities of the visited actions.
  sum_probs = tf.reduce_sum(tf.where(visit_counts > 0, prior_probs, 0.0),
                      axis=-1)
  weighted_q = tf.reduce_sum(
    tf.where(
      visit_counts > 0,
      prior_probs * qvalues /
        tf.where(
          visit_counts > 0,
          tf.expand_dims(sum_probs, -1),
          1.0),
      0.0
    ), axis=-1)
  return (raw_value + tf.cast(sum_visit_counts, tf.float32) * weighted_q) / \
    tf.cast(sum_visit_counts + 1, tf.float32)
