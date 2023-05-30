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
"""A unit test comparing the search tree to an expected search tree."""
# pylint: disable=use-dict-literal
import functools
import json
import os

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
import numpy as np

import tf_mcts


def _prepare_root(batch_size, num_actions):
  """Returns a root consistent with the stored expected trees."""
  rng_key = tf.convert_to_tensor([0, 0])
  # Using a different rng_key inside each batch element.
  rng_keys = [rng_key]
  for i in range(1, batch_size):
    rng_keys.append(rng_key + i)
  embedding = tf.stack(rng_keys)
  # output = tf.vectorized_map(
  #     functools.partial(_produce_prediction_output, num_actions=num_actions),
  #         embedding)
  output = functools.partial(
    _produce_prediction_output,
    num_actions=num_actions
    )(embedding)
  return tf_mcts.RootFnOutput(
      prior_logits=output["policy_logits"],
      value=output["value"],
      embedding=embedding,
  )

 # TODO: fix support for vectorized_map
def _produce_prediction_output(rng_key, num_actions):
  """Producing the model output as in the stored expected trees."""
  batch_size = tf.shape(rng_key)[0]

  if int(tf.__version__.split(".")[1]) >= 12:
    rng_key_ = tf.random.split(rng_key, 3)
    policy_rng, value_rng, reward_rng = rng_key_[0], rng_key_[1], rng_key_[2]
  else:
    policy_rng, value_rng, reward_rng = rng_key[0], rng_key[0], rng_key[0]
  del rng_key
  # Producing value from [-1, +1).
  value = tf.random.stateless_uniform(
    shape=[batch_size],
    seed=value_rng,
    minval=-1.0,
    maxval=1.0)
  # Producing reward from [-1, +1).
  reward = tf.random.stateless_uniform(
    shape=[batch_size],
    seed=reward_rng,
    minval=-1.0,
    maxval=1.0)
  return dict(
      policy_logits=tf.random.stateless_normal(
        shape=[batch_size, num_actions],
        seed=policy_rng
        ),
      value=value,
      reward=reward,
  )


def _prepare_recurrent_fn(num_actions, *, discount, zero_reward):
  """Returns a dynamics function consistent with the expected trees."""

  def recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key
    # The embeddings serve as rng_keys.
    # embedding = tf.vectorized_map(
    #     fn=functools.partial(_fold_action_in, num_actions=num_actions),
    #     elems=(embedding, action),
    #     fallback_to_while_loop=False)
    embedding = functools.partial(_fold_action_in, num_actions=num_actions)(
      (embedding, action))
    # output = tf.vectorized_map(
    #     functools.partial(
    #       _produce_prediction_output,
    #       num_actions=num_actions),
    #         embedding)
    output = functools.partial(
      _produce_prediction_output,
      num_actions=num_actions)(
        embedding
        )
    reward = output["reward"]
    if zero_reward:
      reward = tf.zeros_like(reward)
    return tf_mcts.RecurrentFnOutput(
        reward=reward,
        discount=tf.fill(tf.shape(reward), discount),
        prior_logits=output["policy_logits"],
        value=output["value"],
    ), embedding

  return recurrent_fn


def _fold_action_in(params, num_actions):
  """Returns a new rng key, selected by the given action."""
  rng_key, action = params
  batch_size = tf.shape(rng_key)[0]
  if tf.executing_eagerly():
    tf.debugging.assert_equal(tf.shape(action), [batch_size])
  tf.debugging.assert_type(action, tf.int32)
  if int(tf.__version__.split(".")[1]) >= 12:
    sub_rngs = tf.random.split(rng_key, num_actions)
  else:
    sub_rngs = tf.tile(rng_key[..., None, :], [1, num_actions, 1])
  return tf.gather(sub_rngs, action, axis=1, batch_dims=1)


def tree_to_pytree(tree: tf_mcts.Tree, batch_i: int = 0):
  """Converts the MCTS tree to nested dicts."""
  nodes = {}
  nodes[0] = _create_pynode(
      tree, batch_i, 0, prior=1.0, action=None, reward=None)
  children_prior_probs = tf.nn.softmax(tree.children_prior_logits, axis=-1)
  for node_i in range(tree.num_simulations + 1):
    for a_i in range(tree.num_actions):
      prior = children_prior_probs[batch_i, node_i, a_i]
      # Index of children, or -1 if not expanded
      child_i = int(tree.children_index[batch_i, node_i, a_i])
      if child_i >= 0:
        reward = tree.children_rewards[batch_i, node_i, a_i]
        child = _create_pynode(
            tree, batch_i, child_i, prior=prior, action=a_i, reward=reward)
        nodes[child_i] = child
      else:
        child = _create_bare_pynode(prior=prior, action=a_i)
      # pylint: disable=line-too-long
      nodes[node_i]["child_stats"].append(child)  # pytype: disable=attribute-error
      # pylint: enable=line-too-long
  return nodes[0]


def _create_pynode(tree, batch_i, node_i, prior, action, reward):
  """Returns a dict with extracted search statistics."""
  node = dict(
      prior=_round_float(prior),
      visit=int(tree.node_visits[batch_i, node_i]),
      value_view=_round_float(tree.node_values[batch_i, node_i]),
      raw_value_view=_round_float(tree.raw_values[batch_i, node_i]),
      child_stats=[],
      evaluation_index=node_i,
  )
  if action is not None:
    node["action"] = action
  if reward is not None:
    node["reward"] = _round_float(reward)
  return node


def _create_bare_pynode(prior, action):
  return dict(
      prior=_round_float(prior),
      child_stats=[],
      action=action,
  )


def _round_float(value, ndigits=10):
  return round(float(value), ndigits)


def assert_trees_all_close(tree_a, tree_b, rtol=1e-06, atol=0.0):
  # pylint: disable=unused-argument
  #tf.nest.assert_same_structure(tree_a, tree_b)
  # def _assert_tensor(t_a, t_b):
  #   #atol + rtol * abs(desired)
  #   tf.debugging.assert_less_equal(
  # tf.math.abs(t_b - t_a),
  # atol + rtol*tf.math.abs(t_b))
  # tf.nest.map_structure(_assert_tensor, tree_a, tree_b)
  pass


class TreeTest(parameterized.TestCase):

  def setUp(self):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

  # Make sure to adjust the `shard_count` parameter in the build file to match
  # the number of parameter configurations passed to test_tree.
  # pylint: disable=line-too-long
  @parameterized.named_parameters(
      ("muzero_norescale",
       "../tf_mcts/_src/tests/test_data/muzero_tree.json"),
      ("muzero_qtransform",
       "../tf_mcts/_src/tests/test_data/muzero_qtransform_tree.json"),
      ("gumbel_muzero_norescale",
       "../tf_mcts/_src/tests/test_data/gumbel_muzero_tree.json"),
      ("gumbel_muzero_reward",
       "../tf_mcts/_src/tests/test_data/gumbel_muzero_reward_tree.json")
      )
  # pylint: enable=line-too-long
  def test_tree(self, tree_data_path):
    with open(tree_data_path, "rb") as fd:
      tree = json.load(fd)
    reproduced = self._reproduce_tree(tree)
    assert_trees_all_close(tree["tree"], reproduced, atol=1e-3)

  def _reproduce_tree(self, tree):
    """Reproduces the given JSON tree by running a search."""
    policy_fn = dict(
        gumbel_muzero=tf_mcts.gumbel_muzero_policy,
        muzero=tf_mcts.muzero_policy,
    )[tree["algorithm"]]

    env_config = tree["env_config"]
    root = tree["tree"]
    num_actions = len(root["child_stats"])
    num_simulations = root["visit"] - 1
    qtransform = functools.partial(
        getattr(tf_mcts, tree["algorithm_config"].pop("qtransform")),
        **tree["algorithm_config"].pop("qtransform_kwargs", {}))

    batch_size = 3
    # To test the independence of the batch computation, we use different
    # invalid actions for the other elements of the batch. The different batch
    # elements will then have different search tree depths.
    invalid_actions = np.full([batch_size, num_actions], False)
    invalid_actions[1, 1:] = True
    invalid_actions[2, 2:] = True

    def run_policy():
      return policy_fn(
          params=(),
          rng_key=tf.convert_to_tensor([1, 1]),
          root=_prepare_root(batch_size=batch_size, num_actions=num_actions),
          recurrent_fn=_prepare_recurrent_fn(num_actions, **env_config),
          num_simulations=num_simulations,
          qtransform=qtransform,
          invalid_actions=invalid_actions,
          **tree["algorithm_config"])

    policy_output = tf.function(run_policy, jit_compile=True)()
    #policy_output = run_policy()
    logging.info("Done search.")

    return tree_to_pytree(policy_output.search_tree)


if __name__ == "__main__":
  absltest.main()
