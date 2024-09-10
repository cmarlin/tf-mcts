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
"""Tests for `policies.py`."""
import functools

from absl.testing import absltest
import numpy as np
import os
import tensorflow as tf
import mctx
from mctx._src import policies
from mctx._src import tree as tree_lib


def _make_bandit_recurrent_fn(rewards, dummy_embedding=()):
  """Returns a recurrent_fn with discount=0."""

  def recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key, embedding
    reward = tf.gather(rewards, action, axis=1, batch_dims=1)
    return mctx.RecurrentFnOutput(
        reward=reward,
        discount=tf.zeros_like(reward),
        prior_logits=tf.zeros_like(rewards),
        value=tf.zeros_like(reward),
    ), dummy_embedding

  return recurrent_fn


def _make_bandit_decision_and_chance_fns(rewards, num_chance_outcomes):

  def decision_recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key
    batch_size = action.shape[0]
    reward = tf.gather(rewards, action, axis=1, batch_dims=1)
    dummy_chance_logits = tf.fill(
        [batch_size, num_chance_outcomes],
        -float("inf"))
    dummy_chance_logits = tf.tensor_scatter_nd_update(
        dummy_chance_logits,
        tf.stack(
            [tf.range(batch_size), tf.zeros([batch_size], dtype=tf.int32)],
            -1),
        tf.ones([batch_size])
        )
    afterstate_embedding = (action, embedding)
    return mctx.DecisionRecurrentFnOutput(
        chance_logits=dummy_chance_logits,
        afterstate_value=tf.zeros_like(reward)), afterstate_embedding

  def chance_recurrent_fn(params, rng_key, chance_outcome,
                          afterstate_embedding):
    del params, rng_key, chance_outcome
    afterstate_action, embedding = afterstate_embedding

    reward = tf.gather(rewards, afterstate_action, axis=1, batch_dims=1)
    return mctx.ChanceRecurrentFnOutput(
        action_logits=tf.zeros_like(rewards),
        value=tf.zeros_like(reward),
        discount=tf.zeros_like(reward),
        reward=reward), embedding

  return decision_recurrent_fn, chance_recurrent_fn


def _get_deepest_leaf(tree, node_index):
  """Returns `(leaf, depth)` with maximum depth and visit count.

  Args:
    tree: _unbatched_ MCTS tree state.
    node_index: the node of the inspected subtree.

  Returns:
    `(leaf, depth)` of a deepest leaf. If multiple leaves have the same depth,
    the leaf with the highest visit count is returned.
  """
  np.testing.assert_equal(len(tree.children_index.shape), 2)
  leaf = node_index
  max_found_depth = 0
  for action in range(tree.children_index.shape[-1]):
    next_node_index = tree.children_index[node_index, action]
    if next_node_index != tree_lib.UNVISITED:
      found_leaf, found_depth = _get_deepest_leaf(tree, next_node_index)
      if ((1 + found_depth, tree.node_visits[found_leaf]) >
          (max_found_depth, tree.node_visits[leaf])):
        leaf = found_leaf
        max_found_depth = 1 + found_depth
  return leaf, max_found_depth


class PoliciesTest(absltest.TestCase):

  def setUp(self):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

  def test_apply_temperature_one(self):
    """Tests temperature=1."""
    logits = tf.cast(tf.range(6), dtype=tf.float32)
    new_logits = policies._apply_temperature(logits, temperature=1.0)
    np.testing.assert_allclose(logits - tf.reduce_max(logits), new_logits)

  def test_apply_temperature_two(self):
    """Tests temperature=2."""
    logits = tf.cast(tf.range(6), dtype=tf.float32)
    temperature = 2.0
    new_logits = policies._apply_temperature(logits, temperature)
    np.testing.assert_allclose((logits - tf.reduce_max(logits)) / temperature,
                               new_logits)

  def test_apply_temperature_zero(self):
    """Tests temperature=0."""
    logits = tf.cast(tf.range(4), dtype=tf.float32)
    new_logits = policies._apply_temperature(logits, temperature=0.0)
    np.testing.assert_allclose(
        tf.convert_to_tensor(
            [-2.552118e+38, -1.701412e+38, -8.507059e+37, 0.0]),
        new_logits,
        rtol=1e-3)

  def test_apply_temperature_zero_on_large_logits(self):
    """Tests temperature=0 on large logits."""
    logits = tf.convert_to_tensor(
        [100.0, 3.4028235e+38, -float("inf"), -3.4028235e+38],
        dtype=tf.float32)
    new_logits = policies._apply_temperature(logits, temperature=0.0)
    np.testing.assert_allclose(
        tf.convert_to_tensor(
            [-float("inf"), 0.0, -float("inf"), -float("inf")],
            dtype=tf.float32),
        new_logits)

  def test_mask_invalid_actions(self):
    """Tests action masking."""
    logits = tf.convert_to_tensor(
        [1e6, -float("inf"), 1e6 + 1, -100.0],
        dtype=tf.float32)
    invalid_actions = tf.convert_to_tensor([False, True, False, True])
    masked_logits = policies._mask_invalid_actions(
        logits, invalid_actions)
    valid_probs = tf.nn.softmax(tf.convert_to_tensor([0.0, 1.0]))
    np.testing.assert_allclose(
        tf.convert_to_tensor([valid_probs[0], 0.0, valid_probs[1], 0.0]),
        tf.nn.softmax(masked_logits))

  def test_mask_all_invalid_actions(self):
    """Tests a state with no valid action."""
    logits = tf.convert_to_tensor(
        [-float("inf"), -float("inf"), -float("inf"), -float("inf")],
        dtype=tf.float32)
    invalid_actions = tf.convert_to_tensor([True, True, True, True])
    masked_logits = policies._mask_invalid_actions(
        logits, invalid_actions)
    np.testing.assert_allclose(
        tf.convert_to_tensor([0.25, 0.25, 0.25, 0.25], dtype=tf.float32),
        tf.nn.softmax(masked_logits))

  def test_muzero_policy(self):
    root = mctx.RootFnOutput(
        prior_logits=tf.convert_to_tensor([
            [-1.0, 0.0, 2.0, 3.0],
        ]),
        value=tf.convert_to_tensor([0.0]),
        embedding=(),
    )
    rewards = tf.zeros_like(root.prior_logits)
    invalid_actions = tf.convert_to_tensor([
        [False, False, False, True],
    ])

    policy_output = mctx.muzero_policy(
        params=(),
        rng_key=tf.convert_to_tensor([0, 0]),
        root=root,
        recurrent_fn=_make_bandit_recurrent_fn(rewards),
        num_simulations=1,
        invalid_actions=invalid_actions,
        dirichlet_fraction=0.0)
    expected_action = tf.convert_to_tensor([2], dtype=tf.int32)
    np.testing.assert_array_equal(expected_action, policy_output.action)
    expected_action_weights = tf.convert_to_tensor([
        [0.0, 0.0, 1.0, 0.0],
    ])
    np.testing.assert_allclose(expected_action_weights,
                               policy_output.action_weights)

  def test_gumbel_muzero_policy(self):
    root_value = tf.convert_to_tensor([-5.0])
    root = mctx.RootFnOutput(
        prior_logits=tf.convert_to_tensor([
            [0.0, -1.0, 2.0, 3.0],
        ]),
        value=root_value,
        embedding=(),
    )
    rewards = tf.convert_to_tensor([
        [20.0, 3.0, -1.0, 10.0],
    ])
    invalid_actions = tf.convert_to_tensor([
        [True, False, False, True],
    ])

    value_scale = 0.05
    maxvisit_init = 60
    num_simulations = 17
    max_depth = 3
    qtransform = functools.partial(
        mctx.qtransform_completed_by_mix_value,
        value_scale=value_scale,
        maxvisit_init=maxvisit_init,
        rescale_values=True)
    policy_output = mctx.gumbel_muzero_policy(
        params=(),
        rng_key=tf.convert_to_tensor([42, 42]),
        root=root,
        recurrent_fn=_make_bandit_recurrent_fn(rewards),
        num_simulations=num_simulations,
        invalid_actions=invalid_actions,
        max_depth=max_depth,
        qtransform=qtransform,
        gumbel_scale=1.0)
    # Testing the action.
    expected_action = tf.convert_to_tensor([1], dtype=tf.int32)
    np.testing.assert_array_equal(expected_action, policy_output.action)

    # Testing the action_weights.
    probs = tf.nn.softmax(tf.where(
        invalid_actions, -float("inf"), root.prior_logits))
    mix_value = 1.0 / (num_simulations + 1) * (root_value + num_simulations * (
        probs[:, 1] * rewards[:, 1] + probs[:, 2] * rewards[:, 2]))

    completed_qvalues = tf.convert_to_tensor([
        [mix_value[0], rewards[0, 1], rewards[0, 2], mix_value[0]],
    ])
    max_value = tf.reduce_max(completed_qvalues, axis=-1, keepdims=True)
    min_value = tf.reduce_min(completed_qvalues, axis=-1, keepdims=True)
    total_value_scale = (maxvisit_init + np.ceil(num_simulations / 2)
                         ) * value_scale
    rescaled_qvalues = total_value_scale * (completed_qvalues - min_value) / (
        max_value - min_value)
    expected_action_weights = tf.nn.softmax(
        tf.where(invalid_actions,
                  -float("inf"),
                  root.prior_logits + rescaled_qvalues))
    np.testing.assert_allclose(expected_action_weights,
                               policy_output.action_weights,
                               atol=1e-6)

    # Testing the visit_counts.
    summary = policy_output.search_tree.summary()
    expected_visit_counts = tf.convert_to_tensor(
        [[0.0, np.ceil(num_simulations / 2), num_simulations // 2, 0.0]])
    np.testing.assert_array_equal(expected_visit_counts, summary.visit_counts)

    # Testing max_depth.
    leaf, max_found_depth = _get_deepest_leaf(
        tf.nest.map_structure(lambda x: x[0], policy_output.search_tree),
        tree_lib.ROOT_INDEX)
    self.assertEqual(max_depth, max_found_depth)
    self.assertEqual(6, policy_output.search_tree.node_visits[0, leaf])

  def test_gumbel_muzero_policy_without_invalid_actions(self):
    root_value = tf.convert_to_tensor([-5.0])
    root = mctx.RootFnOutput(
        prior_logits=tf.convert_to_tensor([
            [0.0, -1.0, 2.0, 3.0],
        ]),
        value=root_value,
        embedding=(),
    )
    rewards = tf.convert_to_tensor([
        [20.0, 3.0, -1.0, 10.0],
    ])

    value_scale = 0.05
    maxvisit_init = 60
    num_simulations = 17
    max_depth = 3
    qtransform = functools.partial(
        mctx.qtransform_completed_by_mix_value,
        value_scale=value_scale,
        maxvisit_init=maxvisit_init,
        rescale_values=True)
    policy_output = mctx.gumbel_muzero_policy(
        params=(),
        rng_key=tf.convert_to_tensor([0, 42]),
        root=root,
        recurrent_fn=_make_bandit_recurrent_fn(rewards),
        num_simulations=num_simulations,
        invalid_actions=None,
        max_depth=max_depth,
        qtransform=qtransform,
        gumbel_scale=1.0)
    # Testing the action.
    expected_action = tf.convert_to_tensor([3], dtype=tf.int32)
    np.testing.assert_array_equal(expected_action, policy_output.action)

    # Testing the action_weights.
    summary = policy_output.search_tree.summary()
    completed_qvalues = rewards
    max_value = tf.reduce_max(completed_qvalues, axis=-1, keepdims=True)
    min_value = tf.reduce_min(completed_qvalues, axis=-1, keepdims=True)
    total_value_scale = (maxvisit_init + tf.reduce_max(summary.visit_counts)
                         ) * value_scale
    rescaled_qvalues = total_value_scale * (completed_qvalues - min_value) / (
        max_value - min_value)
    expected_action_weights = tf.nn.softmax(
        root.prior_logits + rescaled_qvalues)
    np.testing.assert_allclose(expected_action_weights,
                               policy_output.action_weights,
                               atol=1e-6)

    # Testing the visit_counts.
    expected_visit_counts = tf.convert_to_tensor(
        [[6, 2, 2, 7]])
    np.testing.assert_array_equal(expected_visit_counts, summary.visit_counts)

  def test_stochastic_muzero_policy(self):
    """Tests that SMZ is equivalent to MZ with a dummy chance function."""
    root = mctx.RootFnOutput(
        prior_logits=tf.convert_to_tensor([
            [-1.0, 0.0, 2.0, 3.0],
            [0.0, 2.0, 5.0, -4.0],
        ]),
        value=tf.convert_to_tensor([1.0, 0.0]),
        embedding=tf.zeros([2, 4]),
    )
    rewards = tf.zeros_like(root.prior_logits)
    invalid_actions = tf.convert_to_tensor([
        [False, False, False, True],
        [True, False, True, False],
    ])

    num_simulations = 10

    policy_output = mctx.muzero_policy(
        params=(),
        rng_key=tf.convert_to_tensor([0, 42]),
        root=root,
        recurrent_fn=_make_bandit_recurrent_fn(
            rewards,
            dummy_embedding=tf.zeros_like(root.embedding)),
        num_simulations=num_simulations,
        invalid_actions=invalid_actions,
        dirichlet_fraction=0.0)

    num_chance_outcomes = 5

    decision_rec_fn, chance_rec_fn = _make_bandit_decision_and_chance_fns(
        rewards, num_chance_outcomes)

    stochastic_policy_output = mctx.stochastic_muzero_policy(
        params=(),
        rng_key=tf.convert_to_tensor([0, 42]),
        root=root,
        decision_recurrent_fn=decision_rec_fn,
        chance_recurrent_fn=chance_rec_fn,
        num_simulations=2 * num_simulations,
        invalid_actions=invalid_actions,
        dirichlet_fraction=0.0)

    np.testing.assert_array_equal(stochastic_policy_output.action,
                                  policy_output.action)

    np.testing.assert_allclose(stochastic_policy_output.action_weights,
                               policy_output.action_weights)


if __name__ == "__main__":
  absltest.main()
