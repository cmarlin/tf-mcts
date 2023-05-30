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
"""A demonstration of the policy improvement by planning with Gumbel."""

import functools
from typing import NamedTuple, Tuple

from absl import app
from absl import flags
import tensorflow as tf
import tf_mcts

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("num_actions", 82, "Number of actions.")
flags.DEFINE_integer("num_simulations", 4, "Number of simulations.")
flags.DEFINE_integer("max_num_considered_actions", 16,
                     "The maximum number of actions expanded at the root.")
flags.DEFINE_integer("num_runs", 1, "Number of runs on random data.")


class DemoOutput(NamedTuple):
  prior_policy_value: tf.Tensor
  prior_policy_action_value: tf.Tensor
  selected_action_value: tf.Tensor
  action_weights_policy_value: tf.Tensor


def _run_demo(
    rng_key:'tf_mcts.PRNGKey'
  ) -> Tuple['tf_mcts.PRNGKey', DemoOutput]:
  """Runs a search algorithm on random data."""
  batch_size = FLAGS.batch_size
  if int(tf.__version__.split('.')[1]) >= 12:
    rng_key_ = tf.random.split(rng_key, 4)
    rng_key, logits_rng, q_rng, search_rng = \
      rng_key_[0], rng_key_[1], rng_key_[2], rng_key_[3]
  else:
    rng_key, logits_rng, q_rng, search_rng = rng_key, rng_key, rng_key, rng_key
  # We will demonstrate the algorithm on random prior_logits.
  # Normally, the prior_logits would be produced by a policy network.
  prior_logits = tf.random.stateless_normal(
      seed=logits_rng, shape=[batch_size, FLAGS.num_actions])
  # Defining a bandit with random Q-values. Only the Q-values of the visited
  # actions will be revealed to the search algorithm.
  qvalues = tf.random.stateless_uniform(seed=q_rng, shape=prior_logits.shape)
  # If we know the value under the prior policy, we can use the value to
  # complete the missing Q-values. The completed Q-values will produce an
  # improved policy in `policy_output.action_weights`.
  raw_value = tf.reduce_sum(tf.nn.softmax(prior_logits) * qvalues, axis=-1)
  use_mixed_value = False

  # The root output would be the output of MuZero representation network.
  root = tf_mcts.RootFnOutput(
      prior_logits=prior_logits,
      value=raw_value,
      # The embedding is used only to implement the MuZero model.
      embedding=tf.zeros([batch_size]),
  )
  # The recurrent_fn would be provided by MuZero dynamics network.
  recurrent_fn = _make_bandit_recurrent_fn(qvalues)

  # Running the search.
  policy_output = tf_mcts.gumbel_muzero_policy(
      params=(),
      rng_key=search_rng,
      root=root,
      recurrent_fn=recurrent_fn,
      num_simulations=FLAGS.num_simulations,
      max_num_considered_actions=FLAGS.max_num_considered_actions,
      qtransform=functools.partial(
          tf_mcts.qtransform_completed_by_mix_value,
          use_mixed_value=use_mixed_value),
  )

  # Collecting the Q-value of the selected action.
  selected_action_value = tf.gather(qvalues, indices=policy_output.action, axis=1, batch_dims=1)

  # We will compare the selected action to the action selected by the
  # prior policy, while using the same Gumbel random numbers.
  gumbel = policy_output.search_tree.extra_data.root_gumbel
  prior_policy_action = tf.argmax(gumbel + prior_logits, axis=-1)
  prior_policy_action_value = tf.gather(qvalues, indices=prior_policy_action, axis=1, batch_dims=1)

  # Computing the policy value under the new action_weights.
  action_weights_policy_value = tf.reduce_sum(
      policy_output.action_weights * qvalues, axis=-1)

  output = DemoOutput(
      prior_policy_value=raw_value,
      prior_policy_action_value=prior_policy_action_value,
      selected_action_value=selected_action_value,
      action_weights_policy_value=action_weights_policy_value,
  )
  return rng_key, output


def _make_bandit_recurrent_fn(qvalues):
  """Returns a recurrent_fn for a determistic bandit."""

  def recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key
    # For the bandit, the reward will be non-zero only at the root.
    reward = tf.where(embedding == 0,
                       tf.gather(qvalues, indices=action, axis=1, batch_dims=1),
                       0.0)
    # On a single-player environment, use discount from [0, 1].
    # On a zero-sum self-play environment, use discount=-1.
    discount = tf.ones_like(reward)
    recurrent_fn_output = tf_mcts.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=tf.zeros_like(qvalues),
        value=tf.zeros_like(reward))
    next_embedding = embedding + 1
    return recurrent_fn_output, next_embedding

  return recurrent_fn


def main(_):
  rng_key = tf.convert_to_tensor([FLAGS.seed, FLAGS.seed])
  jitted_run_demo = tf.function(_run_demo, jit_compile=True)
  for _ in range(FLAGS.num_runs):
    rng_key, output = jitted_run_demo(rng_key)
    # Printing the obtained increase of the policy value.
    # The obtained increase should be non-negative.
    action_value_improvement = (
        output.selected_action_value - output.prior_policy_action_value)
    weights_value_improvement = (
        output.action_weights_policy_value - output.prior_policy_value)
    print("action value improvement:         %.3f (min=%.3f)" %
          (tf.reduce_mean(action_value_improvement), tf.reduce_min(action_value_improvement)))
    print("action_weights value improvement: %.3f (min=%.3f)" %
          (tf.reduce_mean(weights_value_improvement), tf.reduce_min(weights_value_improvement)))


if __name__ == "__main__":
  app.run(main)
