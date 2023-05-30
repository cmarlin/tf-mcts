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
"""A demo of Graphviz visualization of a search tree."""

from typing import Optional, Sequence

from absl import app
from absl import flags
import tensorflow as tf
import mctx
import pygraphviz

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("num_simulations", 32, "Number of simulations.")
flags.DEFINE_integer("max_num_considered_actions", 16,
                     "The maximum number of actions expanded at the root.")
flags.DEFINE_integer("max_depth", None, "The maximum search depth.")
flags.DEFINE_string("output_file", "/tmp/search_tree.png",
                    "The output file for the visualization.")


def convert_tree_to_graph(
    tree: mctx.Tree,
    action_labels: Optional[Sequence[str]] = None,
    batch_index: int = 0
) -> pygraphviz.AGraph:
  """Converts a search tree into a Graphviz graph.

  Args:
    tree: A `Tree` containing a batch of search data.
    action_labels: Optional labels for edges, defaults to the action index.
    batch_index: Index of the batch element to plot.

  Returns:
    A Graphviz graph representation of `tree`.
  """
  tf.debugging.assert_rank(tree.node_values, 2)
  batch_size = tree.node_values.shape[0]
  if action_labels is None:
    action_labels = range(tree.num_actions)
  elif len(action_labels) != tree.num_actions:
    raise ValueError(
        f"action_labels {action_labels} has the wrong number of actions "
        f"({len(action_labels)}). "
        f"Expecting {tree.num_actions}.")

  def node_to_str(node_i, reward=0, discount=1):
    return (f"{node_i}\n"
            f"Reward: {reward:.2f}\n"
            f"Discount: {discount:.2f}\n"
            f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"
            f"Visits: {tree.node_visits[batch_index, node_i]}\n")

  def edge_to_str(node_i, a_i):
    node_index = tf.fill([batch_size], node_i)
    probs = tf.nn.softmax(tree.children_prior_logits[batch_index, node_i])
    return (f"{action_labels[a_i]}\n"
            f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"  # pytype: disable=unsupported-operands  # always-use-return-annotations
            f"p: {probs[a_i]:.2f}\n")

  graph = pygraphviz.AGraph(directed=True)

  # Add root
  graph.add_node(0, label=node_to_str(node_i=0), color="green")
  # Add all other nodes and connect them up.
  for node_i in range(tree.num_simulations):
    for a_i in range(tree.num_actions):
      # Index of children, or -1 if not expanded
      children_i = tree.children_index[batch_index, node_i, a_i].numpy()
      if children_i >= 0:
        graph.add_node(
            children_i,
            label=node_to_str(
                node_i=children_i,
                reward=tree.children_rewards[batch_index, node_i, a_i],
                discount=tree.children_discounts[batch_index, node_i, a_i]),
            color="red")
        graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i))

  return graph


def _run_demo(rng_key: 'mctx.PRNGKey'):
  """Runs a search algorithm on a toy environment."""
  # We will define a deterministic toy environment.
  # The deterministic `transition_matrix` has shape `[num_states, num_actions]`.
  # The `transition_matrix[s, a]` holds the next state.
  transition_matrix = tf.convert_to_tensor([
      [1, 2, 3, 4],
      [0, 5, 0, 0],
      [0, 0, 0, 6],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
  ], dtype=tf.int32)
  # The `rewards` have shape `[num_states, num_actions]`. The `rewards[s, a]`
  # holds the reward for that (s, a) pair.
  rewards = tf.convert_to_tensor([
      [1, -1, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [10, 0, 20, 0],
  ], dtype=tf.float32)
  num_states = rewards.shape[0]
  # The discount for each (s, a) pair.
  discounts = tf.where(transition_matrix > 0, 1.0, 0.0)
  # Using optimistic initial values to encourage exploration.
  values = tf.fill([num_states], 15.0)
  # The prior policies for each state.
  all_prior_logits = tf.zeros_like(rewards)
  root, recurrent_fn = _make_batched_env_model(
      # Using batch_size=2 to test the batched search.
      batch_size=2,
      transition_matrix=transition_matrix,
      rewards=rewards,
      discounts=discounts,
      values=values,
      prior_logits=all_prior_logits)

  # Running the search.
  policy_output = mctx.gumbel_muzero_policy(
      params=(),
      rng_key=rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      num_simulations=FLAGS.num_simulations,
      max_depth=FLAGS.max_depth,
      max_num_considered_actions=FLAGS.max_num_considered_actions,
  )
  return policy_output


def _make_batched_env_model(
    batch_size: int,
    *,
    transition_matrix: tf.Tensor,
    rewards: tf.Tensor,
    discounts: tf.Tensor,
    values: tf.Tensor,
    prior_logits: tf.Tensor):
  """Returns a batched `(root, recurrent_fn)`."""
  tf.debugging.assert_equal(tf.shape(transition_matrix), tf.shape(rewards))
  tf.debugging.assert_equal(tf.shape(transition_matrix), tf.shape(discounts))
  tf.debugging.assert_equal(tf.shape(transition_matrix), tf.shape(prior_logits))
  num_states, num_actions = transition_matrix.shape
  tf.debugging.assert_equal(tf.shape(values), [num_states])
  # We will start the search at state zero.
  root_state = 0
  root = mctx.RootFnOutput(
      prior_logits=tf.tile(prior_logits[root_state][None], [batch_size, 1]),
      value=tf.fill([batch_size], values[root_state]),
      # The embedding will hold the state index.
      embedding=tf.zeros([batch_size], dtype=tf.int32),
  )

  def recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key
    tf.debugging.assert_equal(tf.shape(action), [batch_size])
    tf.debugging.assert_equal(tf.shape(embedding), [batch_size])
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=tf.gather(tf.gather(rewards, embedding), action, axis=1, batch_dims=1),  # rewards[embedding, action]
        discount=tf.gather(tf.gather(discounts, embedding), action, axis=1, batch_dims=1),  # discounts[embedding, action],
        prior_logits=tf.gather(prior_logits, embedding),  # prior_logits[embedding],
        value=tf.gather(values, embedding))  # values[embedding])
    next_embedding = tf.gather(tf.gather(transition_matrix, embedding), action, axis=1, batch_dims=1)  # transition_matrix[embedding, action]
    return recurrent_fn_output, next_embedding

  return root, recurrent_fn


def main(_):
  rng_key = tf.convert_to_tensor([0, FLAGS.seed])
  jitted_run_demo = tf.function(_run_demo, jit_compile=True)
  print("Starting search.")
  policy_output = jitted_run_demo(rng_key)
  batch_index = 0
  selected_action = policy_output.action[batch_index]
  q_value = policy_output.search_tree.summary().qvalues[
      batch_index, selected_action]
  print("Selected action:", selected_action)
  # To estimate the value of the root state, use the Q-value of the selected
  # action. The Q-value is not affected by the exploration at the root node.
  print("Selected action Q-value:", q_value)
  graph = convert_tree_to_graph(policy_output.search_tree)
  print("Saving tree diagram to:", FLAGS.output_file)
  graph.draw(FLAGS.output_file, prog="dot")


if __name__ == "__main__":
  app.run(main)
