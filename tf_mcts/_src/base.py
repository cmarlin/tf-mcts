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
"""Core types used in tf_mcts."""

from typing import Any, Callable, NamedTuple, TypeVar, Tuple, Union

import tensorflow as tf

from tf_mcts._src import tree


PRNGKey = Any  # TypeVar("PRNGKey")


# Parameters are arbitrary nested structures of `chex.Array`.
# A nested structure is either a single object, or a collection (list, tuple,
# dictionary, etc.) of other nested structures.
Params = Union[tf.Tensor, Tuple[tf.Tensor, ...]]


# The model used to search is expressed by a `RecurrentFn` function that takes
# `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput` and
# the new state embedding.
class RecurrentFnOutput(NamedTuple):
  """The output of a `RecurrentFn`.

  reward: `[B]` an approximate reward from the state-action transition.
  discount: `[B]` the discount between the `reward` and the `value`.
  prior_logits: `[B, num_actions]` the logits produced by a policy network.
  value: `[B]` an approximate value of the state after the state-action
    transition.
  """
  reward: tf.Tensor
  discount: tf.Tensor
  prior_logits: tf.Tensor
  value: tf.Tensor


Action = tf.Tensor
RecurrentState = Any
RecurrentFn = Callable[
    [Params, PRNGKey, Action, RecurrentState],
    Tuple[RecurrentFnOutput, RecurrentState]]

class RootFnOutput(NamedTuple):
  """The output of a representation network.

  prior_logits: `[B, num_actions]` the logits produced by a policy network.
  value: `[B]` an approximate value of the current state.
  embedding: `[B, ...]` the inputs to the next `recurrent_fn` call.
  """
  prior_logits: tf.Tensor
  value: tf.Tensor
  embedding: RecurrentState

  def replace(self, **kwargs):
    return tree.namedtuple_replace(self, **kwargs)


# Action selection functions specify how to pick nodes to expand in the tree.
NodeIndices = tf.Tensor
Depth = tf.Tensor
RootActionSelectionFn = Callable[
    [PRNGKey, tree.Tree, NodeIndices], tf.Tensor]
InteriorActionSelectionFn = Callable[
    [PRNGKey, tree.Tree, NodeIndices, Depth],
    tf.Tensor]
QTransform = Callable[[tree.Tree, tf.Tensor], tf.Tensor]
# LoopFn has the same interface as jax.lax.fori_loop.
LoopFn = Callable[
    [int, int, Callable[[Any, Any], Any], Tuple[PRNGKey, tree.Tree]],
    Tuple[PRNGKey, tree.Tree]]

T = TypeVar("T")


class PolicyOutput(NamedTuple):  # Generic[T]
  """The output of a policy.

  action: `[B]` the proposed action.
  action_weights: `[B, num_actions]` the targets used to train a policy network.
    The action weights sum to one. Usually, the policy network is trained by
    cross-entropy:
    `cross_entropy(labels=stop_gradient(action_weights), logits=prior_logits)`.
  search_tree: `[B, ...]` the search tree of the finished search.
  """
  action: tf.Tensor
  action_weights: tf.Tensor
  search_tree: tree.Tree  # tree.Tree[T]


class DecisionRecurrentFnOutput(NamedTuple):
  """Output of the function for expanding decision nodes.

  Expanding a decision node takes an action and a state embedding and produces
  an afterstate, which represents the state of the environment after an action
  is taken but before the environment has updated its state. Accordingly, there
  is no discount factor or reward for transitioning from state `s` to afterstate
  `sa`.

  Attributes:
    chance_logits: `[B, C]` logits of `C` chance outcomes at the afterstate.
    afterstate_value: `[B]` values of the afterstates `v(sa)`.
  """
  chance_logits: tf.Tensor  # [B, C]
  afterstate_value: tf.Tensor  # [B]


class ChanceRecurrentFnOutput(NamedTuple):
  """Output of the function for expanding chance nodes.

  Expanding a chance node takes a chance outcome and an afterstate embedding
  and produces a state, which captures a potentially stochastic environment
  transition. When this transition occurs reward and discounts are produced as
  in a normal transition.

  Attributes:
    action_logits: `[B, A]` logits of different actions from the state.
    value: `[B]` values of the states `v(s)`.
    reward: `[B]` rewards at the states.
    discount: `[B]` discounts at the states.
  """
  action_logits: tf.Tensor  # [B, A]
  value: tf.Tensor  # [B]
  reward: tf.Tensor  # [B]
  discount: tf.Tensor  # [B]


class StochasticRecurrentState(NamedTuple):
  """Wrapper that enables different treatment of decision and chance nodes.

  In Stochastic MuZero tree nodes can either be decision or chance nodes, these
  nodes are treated differently during expansion, search and backup, and a user
  could also pass differently structured embeddings for each type of node. This
  wrapper enables treating chance and decision nodes differently and supports
  potential differences between chance and decision node structures.

  Attributes:
    state_embedding: `[B ...]` an optionally meaningful state embedding.
    afterstate_embedding: `[B ...]` an optionally meaningful afterstate
      embedding.
    is_decision_node: `[B]` whether the node is a decision or chance node. If it
      is a decision node, `afterstate_embedding` is a dummy value. If it is a
      chance node, `state_embedding` is a dummy value.
  """
  state_embedding: Tuple[tf.Tensor]  # [B, ...]
  afterstate_embedding: Tuple[tf.Tensor]  # [B, ...]
  is_decision_node: tf.Tensor  # [B]


RecurrentState = Tuple[tf.Tensor]

DecisionRecurrentFn = Callable[[Params, PRNGKey, Action, RecurrentState],
                               Tuple[DecisionRecurrentFnOutput, RecurrentState]]

ChanceRecurrentFn = Callable[[Params, PRNGKey, Action, RecurrentState],
                             Tuple[ChanceRecurrentFnOutput, RecurrentState]]
