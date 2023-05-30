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
"""A data structure used to hold / inspect search data for a batch of inputs."""

from __future__ import annotations
from typing import Any, NamedTuple, TypeVar

import tensorflow as tf


T = TypeVar("T")

# service to mimic chex dataclass
def namedtuple_replace(self: NamedTuple, **kwargs):
  dict_self = dict(zip(self._fields, self))
  for k, v in kwargs.items():
    dict_self[k] = v
  return self.__class__(**dict_self)


# The following attributes are class variables (and should not be set on
# Tree instances).
ROOT_INDEX: int = 0
NO_PARENT: int = -1
UNVISITED: int = -1


class Tree(NamedTuple):  # Generic[T]
  """State of a search tree.

  The `Tree` dataclass is used to hold and inspect search data for a batch of
  inputs. In the fields below `B` denotes the batch dimension, `N` represents
  the number of nodes in the tree, and `num_actions` is the number of discrete
  actions.

  node_visits: `[B, N]` the visit counts for each node.
  raw_values: `[B, N]` the raw value for each node.
  node_values: `[B, N]` the cumulative search value for each node.
  parents: `[B, N]` the node index for the parents for each node.
  action_from_parent: `[B, N]` action to take from the parent to reach each
    node.
  children_index: `[B, N, num_actions]` the node index of the children for each
    action.
  children_prior_logits: `[B, N, num_actions]` the action prior logits of each
    node.
  children_visits: `[B, N, num_actions]` the visit counts for children for
    each action.
  children_rewards: `[B, N, num_actions]` the immediate reward for each action.
  children_discounts: `[B, N, num_actions]` the discount between the
    `children_rewards` and the `children_values`.
  children_values: `[B, N, num_actions]` the value of the next node after the
    action.
  embeddings: `[B, N, ...]` the state embeddings of each node.
  root_invalid_actions: `[B, num_actions]` a mask with invalid actions at the
    root. In the mask, invalid actions have ones, and valid actions have zeros.
  extra_data: `[B, ...]` extra data passed to the search.
  """
  node_visits: tf.Tensor  # [B, N]
  raw_values: tf.Tensor  # [B, N]
  node_values: tf.Tensor  # [B, N]
  parents: tf.Tensor  # [B, N]
  action_from_parent: tf.Tensor  # [B, N]
  children_index: tf.Tensor  # [B, N, num_actions]
  children_prior_logits: tf.Tensor  # [B, N, num_actions]
  children_visits: tf.Tensor  # [B, N, num_actions]
  children_rewards: tf.Tensor  # [B, N, num_actions]
  children_discounts: tf.Tensor  # [B, N, num_actions]
  children_values: tf.Tensor  # [B, N, num_actions]
  embeddings: Any  # [B, N, ...]
  root_invalid_actions: tf.Tensor  # [B, num_actions]
  extra_data: Any  # T  # [B, ...]

  # The following attributes are class variables (and should not be set on
  # Tree instances).
  # ROOT_INDEX: ClassVar[int] = 0
  # NO_PARENT: ClassVar[int] = -1
  # UNVISITED: ClassVar[int] = -1

  @property
  def num_actions(self):
    return self.children_index.shape[-1]

  @property
  def num_simulations(self):
    return self.node_visits.shape[-1] - 1

  def qvalues(self, indices:tf.Tensor):
    """Compute q-values for any node indices in the tree."""
    # pytype: disable=wrong-arg-types  # jnp-type
    tf.debugging.assert_rank(self.children_discounts, 3)
    return tf.gather(self.children_rewards, indices, axis=1, batch_dims=1) + \
      tf.gather(self.children_discounts, indices, axis=1, batch_dims=1) * \
        tf.gather(self.children_values, indices, axis=1, batch_dims=1)
    # pytype: enable=wrong-arg-types

  def summary(self) -> SearchSummary:
    """Extract summary statistics for the root node."""
    # Get state and action values for the root nodes.
    tf.debugging.assert_rank(self.node_values, 2)
    value = self.node_values[:, ROOT_INDEX]
    batch_size, = value.shape
    root_indices = tf.fill((batch_size,), ROOT_INDEX)
    qvalues = self.qvalues(root_indices)
    # Extract visit counts and induced probabilities for the root nodes.
    visit_counts = tf.cast(self.children_visits[:, ROOT_INDEX], value.dtype)
    total_counts = tf.reduce_sum(visit_counts, axis=-1, keepdims=True)
    visit_probs = visit_counts / tf.reduce_max(total_counts, 1)[..., None]
    visit_probs = tf.where(total_counts > 0, visit_probs, 1 / self.num_actions)
    # Return relevant stats.
    return SearchSummary(  # pytype: disable=wrong-arg-types  # numpy-scalars
        visit_counts=visit_counts,
        visit_probs=visit_probs,
        value=value,
        qvalues=qvalues)

  def replace(self, **kwargs):
    return namedtuple_replace(self, **kwargs)


def infer_batch_size(tree: Tree) -> int:
  """Recovers batch size from `Tree` data structure."""
  tf.assert_rank(tree.node_values, 2, message="Input tree is not batched.")
  batch_size = tf.shape(tree.node_values)[0]
  if tf.executing_eagerly():
    tf.nest.map_structure(
      lambda t: tf.debugging.assert_equal(tf.shape(t)[0], batch_size),
      tree)
  return batch_size


# A number of aggregate statistics and predictions are extracted from the
# search data and returned to the user for further processing.
class SearchSummary(NamedTuple):
  """Stats from MCTS search."""
  visit_counts: tf.Tensor
  visit_probs: tf.Tensor
  value: tf.Tensor
  qvalues: tf.Tensor

