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
"""A collection of action selection functions."""
from typing import NamedTuple, Optional, TypeVar, Union

import numpy as np
import tensorflow as tf

from mctx._src import base
from mctx._src import qtransforms
from mctx._src import seq_halving
from mctx._src import tree as tree_lib


def switching_action_selection_wrapper(
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn
) -> base.InteriorActionSelectionFn:
  """Wraps root and interior action selection fns in a conditional statement."""

  def switching_action_selection_fn(
      rng_key: base.PRNGKey,
      tree: tree_lib.Tree,
      node_index: base.NodeIndices,
      depth: base.Depth) -> tf.Tensor:
    return tf.where(
      depth == 0.0,
      root_action_selection_fn(rng_key, tree, node_index),
      interior_action_selection_fn(rng_key, tree, node_index, depth),
    )

  return switching_action_selection_fn


def muzero_action_selection(
    rng_key: base.PRNGKey,
    tree: tree_lib.Tree,
    node_index: tf.Tensor,
    depth: Union[int, tf.Tensor],
    *,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    qtransform: base.QTransform = qtransforms.qtransform_by_parent_and_siblings,
) -> tf.Tensor:
  """Returns the action selected for a node index.

  See Appendix B in https://arxiv.org/pdf/1911.08265.pdf for more details.

  Args:
    rng_key: random number generator state.
    tree: [B, ...] MCTS tree state.  (JAX was _unbatched_)
    node_index: scalar index of the node from which to select an action.
    depth: the scalar depth of the current node. The root has depth zero.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    qtransform: a monotonic transformation to convert the Q-values to [0, 1].

  Returns:
    action: the action selected from the given node.
  """
  visit_counts = tf.cast(
    tf.gather(tree.children_visits, node_index, axis=1, batch_dims=1),
    tf.float32
    )
  node_visit = tf.cast(
    tf.gather(tree.node_visits, node_index, axis=1, batch_dims=1),
    tf.float32
    )
  pb_c = pb_c_init + tf.math.log((node_visit + pb_c_base + 1.) / pb_c_base)
  prior_logits = \
    tf.gather(tree.children_prior_logits, node_index, axis=1, batch_dims=1)
  prior_probs = tf.nn.softmax(prior_logits)
  policy_score = (tf.math.sqrt(node_visit) * pb_c)[..., None] * prior_probs / \
    (visit_counts + 1)
  batch_size = tree_lib.infer_batch_size(tree)
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(node_index), (batch_size))
    tf.assert_equal(tf.shape(node_visit), (batch_size))
    tf.assert_equal(tf.shape(prior_probs), tf.shape(visit_counts))
    tf.assert_equal(tf.shape(prior_probs), tf.shape(policy_score))
  value_score = qtransform(tree, node_index)

  # Add tiny bit of randomness for tie break
  node_noise_score = 1e-7 * \
    tf.random.stateless_uniform((batch_size, tree.num_actions), seed=rng_key)
  to_argmax = value_score + policy_score + node_noise_score

  # Masking the invalid actions at the root.
  return masked_argmax(
    to_argmax,
    tf.logical_and(
      tree.root_invalid_actions,
      tf.convert_to_tensor(depth == 0)[..., None]
      )
    )


class GumbelMuZeroExtraData(NamedTuple):
  """Extra data for Gumbel MuZero search."""
  root_gumbel: tf.Tensor


GumbelMuZeroExtraDataType = TypeVar(  # pylint: disable=invalid-name
    "GumbelMuZeroExtraDataType", bound=GumbelMuZeroExtraData)


def gumbel_muzero_root_action_selection(
    rng_key: "base.PRNGKey",
    tree: tree_lib.Tree,  # tree_lib.Tree[GumbelMuZeroExtraDataType],
    node_index: tf.Tensor,
    *,
    num_simulations: tf.Tensor,
    max_num_considered_actions: tf.Tensor,
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value,
) -> tf.Tensor:
  """Returns the action selected by Sequential Halving with Gumbel.

  Initially, we sample `max_num_considered_actions` actions without replacement.
  From these, the actions with the highest `gumbel + logits + qvalues` are
  visited first.

  Args:
    rng_key: random number generator state.
    tree: [B, ...] MCTS tree state.
    node_index: [B] scalar index of the node from which to take an action.
    num_simulations: the simulation budget.
    max_num_considered_actions: the number of actions sampled without
      replacement.
    qtransform: a monotonic transformation for the Q-values.

  Returns:
    action: the action selected from the given node.
  """
  del rng_key
  batch_size = tree_lib.infer_batch_size(tree)
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(node_index), (batch_size))
  visit_counts = tf.gather(
    tree.children_visits, node_index, axis=1, batch_dims=1)
  prior_logits = tf.gather(
    tree.children_prior_logits, node_index, axis=1, batch_dims=1)
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(visit_counts), tf.shape(prior_logits))
  completed_qvalues = qtransform(tree, node_index)

  table = tf.convert_to_tensor(seq_halving.get_table_of_considered_visits(
      max_num_considered_actions, num_simulations))
  num_valid_actions = tf.reduce_sum(
      1 - tf.cast(tree.root_invalid_actions, tf.int32), axis=-1)
  num_considered = tf.math.minimum(
      max_num_considered_actions, num_valid_actions)
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(num_considered), (batch_size))
  # At the root, the simulation_index is equal to the sum of visit counts.
  simulation_index = tf.reduce_sum(visit_counts, -1)
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(simulation_index), (batch_size))
  considered_visit = tf.gather(
    tf.gather(table, num_considered),
    simulation_index,
    axis=1,
    batch_dims=1)
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(considered_visit), (batch_size))
  gumbel = tree.extra_data.root_gumbel
  to_argmax = seq_halving.score_considered(
      considered_visit, gumbel, prior_logits, completed_qvalues,
      visit_counts)

  # Masking the invalid actions at the root.
  return masked_argmax(to_argmax, tree.root_invalid_actions)


def gumbel_muzero_interior_action_selection(
    rng_key: "base.PRNGKey",
    tree: tree_lib.Tree,
    node_index: tf.Tensor,
    depth: tf.Tensor,
    *,
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value,
) -> tf.Tensor:
  """Selects the action with a deterministic action selection.

  The action is selected based on the visit counts to produce visitation
  frequencies similar to softmax(prior_logits + qvalues).

  Args:
    rng_key: random number generator state.
    tree: [B, ...] MCTS tree state.
    node_index: scalar index of the node from which to take an action.
    depth: the scalar depth of the current node. The root has depth zero.
    qtransform: function to obtain completed Q-values for a node.

  Returns:
    action: the action selected from the given node.
  """
  del rng_key, depth
  batch_size = tree_lib.infer_batch_size(tree)
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(node_index), (batch_size))
  visit_counts = tf.gather(
    tree.children_visits, node_index, axis=1, batch_dims=1)
  prior_logits = tf.gather(
    tree.children_prior_logits, node_index, axis=1, batch_dims=1)
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(visit_counts), tf.shape(prior_logits))
  completed_qvalues = qtransform(tree, node_index)

  # The `prior_logits + completed_qvalues` provide an improved policy,
  # because the missing qvalues are replaced by v_{prior_logits}(node).
  to_argmax = _prepare_argmax_input(
      probs=tf.nn.softmax(prior_logits + completed_qvalues),
      visit_counts=tf.cast(visit_counts, tf.float32))

  tf.assert_rank(to_argmax, 2)
  return tf.argmax(to_argmax, axis=-1, output_type=tf.int32)


def masked_argmax(
    to_argmax: tf.Tensor,
    invalid_actions: Optional[tf.Tensor]) -> tf.Tensor:
  """Returns a valid action with the highest `to_argmax`."""
  if invalid_actions is not None:
    if tf.executing_eagerly():
      tf.assert_equal(tf.shape(to_argmax), tf.shape(invalid_actions))
    # The usage of the -inf inside the argmax does not lead to NaN.
    # Do not use -inf inside softmax, logsoftmax or cross-entropy.
    to_argmax = tf.where(invalid_actions, -np.inf, to_argmax)
  # If all actions are invalid, the argmax returns action 0.
  return tf.argmax(to_argmax, axis=-1, output_type=tf.int32)


def _prepare_argmax_input(probs, visit_counts):
  """Prepares the input for the deterministic selection.

  When calling argmax(_prepare_argmax_input(...)) multiple times
  with updated visit_counts, the produced visitation frequencies will
  approximate the probs.

  For the derivation, see Section 5 "Planning at non-root nodes" in
  "Policy improvement by planning with Gumbel":
  https://openreview.net/forum?id=bERaNdoegnO

  Args:
    probs: a policy or an improved policy. Shape `[num_actions]`.
    visit_counts: the existing visit counts. Shape `[num_actions]`.

  Returns:
    The input to an argmax. Shape `[num_actions]`.
  """
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(probs), tf.shape(visit_counts))
  to_argmax = probs - visit_counts / (
      1 + tf.reduce_sum(visit_counts, keepdims=True, axis=-1))
  return to_argmax
