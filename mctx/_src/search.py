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
"""A JAX implementation of batched MCTS."""
from typing import Any, NamedTuple, Optional, Tuple, TypeVar

import tensorflow as tf

from mctx._src import action_selection
from mctx._src import base
from mctx._src import tree as tree_lib

Tree = tree_lib.Tree
T = TypeVar('T')


def search(
    params: base.Params,
    rng_key: base.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[tf.Tensor] = None,
    extra_data: Any = None,
    # loop_fn: base.LoopFn = tf.while_loop
    ) -> Tree:
  """Performs a full search and returns sampled actions.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = tf.cast(tf.zeros_like(root.prior_logits), tf.bool)

  def body_fun(sim, loop_state):
    rng_key, tree = loop_state
    if int(tf.__version__.split('.')[1]) >= 12:
      rng_key_ = tf.random.split(rng_key, 3)
      rng_key, simulate_key, expand_key = \
        rng_key_[0], rng_key_[1], rng_key_[2]
    else:
      rng_key, simulate_key, expand_key = \
        tf.identity(rng_key), rng_key, rng_key
    # simulate is vmapped and expects batched rng keys.
    # if int(tf.__version__.split('.')[1]) >= 12:
    #   simulate_keys = tf.random.split(simulate_key, batch_size)
    # else:
    simulate_keys = simulate_key
    parent_index, action = simulate(
        simulate_keys,
        tree,
        action_selection_fn,
        max_depth
        )
    # A node first expanded on simulation `i`, will have node index `i`.
    # Node 0 corresponds to the root node.
    next_node_index = tf.gather(
      tf.gather(tree.children_index, parent_index, axis=1, batch_dims=1),
      action,
      axis=1,
      batch_dims=1
      )
    next_node_index = tf.where(
      next_node_index == tree_lib.UNVISITED,
      sim + 1,
      next_node_index
      )
    tree = expand(
        params, expand_key, tree, recurrent_fn, parent_index,
        action, next_node_index)
    tree = backward(tree, next_node_index)
    loop_state = rng_key, tree
    return sim+1, loop_state

  # Allocate all necessary storage.
  # None is unsupported by tf.while_loop
  tree = instantiate_tree_from_root(root, num_simulations,
    root_invalid_actions=invalid_actions,
    extra_data=extra_data if \
      extra_data is not None else \
        tf.zeros([tf.shape(root.value)[0], 0])
    )

  _, (_, tree) = tf.while_loop(
      lambda *args: True,
      body_fun,
      (0, (rng_key, tree)),
      maximum_iterations=num_simulations
      )

  return tree


class _SimulationState(NamedTuple):
  """The state for the simulation while loop."""
  # rng_key: base.PRNGKey
  node_index: tf.Tensor  # int
  action: tf.Tensor  # int
  next_node_index: tf.Tensor  # int
  depth: tf.Tensor  # int
  is_continuing: tf.Tensor  # bool


def simulate(
    rng_key: base.PRNGKey,
    tree: Tree,
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int) -> Tuple[tf.Tensor, tf.Tensor]:
  """Traverses the tree until reaching an unvisited action or `max_depth`.

  Each simulation starts from the root and keeps selecting actions traversing
  the tree until a leaf or `max_depth` is reached.

  Args:
    rng_key: random number generator state, the key is consumed.
    tree: [B, ...] MCTS tree state.
    action_selection_fn: function used to select an action during simulation.
    max_depth: maximum search tree depth allowed during simulation.

  Returns:
    `(parent_index, action)` tuple, where `parent_index` is the index of the
    node reached at the end of the simulation, and the `action` is the action to
    evaluate from the `parent_index`.
  """
  def cond_fun(rng_key, state):
    # pylint: disable=unused-argument
    return tf.reduce_any(state.is_continuing)

  def body_fun(rng_key, state):
    # Preparing the next simulation state.
    node_index = state.next_node_index
    if int(tf.__version__.split('.')[1]) >= 12:
      rng_key_ = tf.random.split(rng_key, 2)
      rng_key, action_selection_key = rng_key_[0], rng_key_[1]
    else:
      rng_key, action_selection_key = tf.identity(rng_key), rng_key

    # mask to avoid access to invalid indices
    masked_node_index = tf.where(state.is_continuing, node_index, 0)
    action = action_selection_fn(
      action_selection_key,
      tree,
      masked_node_index,
      state.depth)
    next_node_index = tf.gather(
      tf.gather(tree.children_index, masked_node_index, axis=1, batch_dims=1),
      action,
      axis=1,
      batch_dims=1
      )
    # The returned action will be visited.
    depth = state.depth + 1
    is_before_depth_cutoff = depth < max_depth
    is_visited = next_node_index != tf.constant(tree_lib.UNVISITED)[..., None]
    is_continuing = tf.logical_and(is_visited, is_before_depth_cutoff)
    ret = _SimulationState(  # pytype: disable=wrong-arg-types  # jax-types
        # rng_key=rng_key,
        node_index=node_index,
        action=action,
        next_node_index=next_node_index,
        depth=depth,
        is_continuing=is_continuing)
    next_state = tf.nest.map_structure(
      lambda t_true, t_false: tf.where(state.is_continuing, t_true, t_false),
      ret,
      state
      )
    return rng_key, next_state

  batch_size = tf.shape(tree.node_values)[0]
  node_index = tf.fill([batch_size], tree_lib.ROOT_INDEX)  # dtype=tf.int32
  depth = tf.zeros(batch_size, dtype=tree.children_prior_logits.dtype)
  initial_state = _SimulationState(
      # rng_key=rng_key,
      node_index=tf.fill([batch_size], tree_lib.NO_PARENT),
      action=tf.fill([batch_size], tree_lib.NO_PARENT),
      next_node_index=node_index,
      depth=depth,
      is_continuing=tf.fill([batch_size], True))
  _, end_state = tf.while_loop(cond_fun, body_fun, (rng_key, initial_state))

  # Returning a node with a selected action.
  # The action can be already visited, if the max_depth is reached.
  return end_state.node_index, end_state.action


def expand(
    params: base.Params,
    rng_key: base.PRNGKey,
    tree: Tree,  # Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: tf.Tensor,
    action: tf.Tensor,
    next_node_index: tf.Tensor
    ) -> Tree:  # Tree[T]
  """Create and evaluate child nodes from given nodes and unvisited actions.

  Args:
    params: params to be forwarded to recurrent function.
    rng_key: random number generator state.
    tree: the MCTS tree state to update.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    parent_index: the index of the parent node, from which the action will be
      expanded. Shape `[B]`.
    action: the action to expand. Shape `[B]`.
    next_node_index: the index of the newly expanded node. This can be the index
      of an existing node, if `max_depth` is reached. Shape `[B]`.

  Returns:
    tree: updated MCTS tree state.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(parent_index), (batch_size,))
    tf.assert_equal(tf.shape(action), (batch_size,))
    tf.assert_equal(tf.shape(next_node_index), (batch_size,))

  # Retrieve states for nodes to be evaluated.
  embedding = tf.nest.map_structure(
      lambda x: tf.gather(x, parent_index, axis=1, batch_dims=1),
      tree.embeddings)

  # Evaluate and create a new node.
  step, embedding = recurrent_fn(params, rng_key, action, embedding)
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(step.prior_logits), [batch_size, tree.num_actions])
    tf.assert_equal(tf.shape(step.reward), [batch_size])
    tf.assert_equal(tf.shape(step.discount), [batch_size])
    tf.assert_equal(tf.shape(step.value), [batch_size])

  tree = update_tree_node(
    tree, next_node_index, step.prior_logits, step.value, embedding)

  # Return updated tree topology.
  return tree.replace(
      children_index=batch_update(
          tree.children_index, next_node_index, parent_index, action),
      children_rewards=batch_update(
          tree.children_rewards, step.reward, parent_index, action),
      children_discounts=batch_update(
          tree.children_discounts, step.discount, parent_index, action),
      parents=batch_update(tree.parents, parent_index, next_node_index),
      action_from_parent=batch_update(
          tree.action_from_parent, action, next_node_index))


def backward(
    tree: Tree,  # Tree[T]
    leaf_index: tf.Tensor
    ) -> Tree:  # Tree[T]
  """Goes up and updates the tree until all nodes reached the root.

  Args:
    tree: the MCTS tree state to update, without the batch size.
    leaf_index: the node index from which to do the backward.

  Returns:
    Updated MCTS tree state.
  """

  def cond_fun(tree, leaf_value, index):
    # pylint: disable=unused-argument
    return tf.reduce_any(index != tree_lib.ROOT_INDEX)

  def body_fun(tree, leaf_value, index):
    cond_mask = index != tree_lib.ROOT_INDEX
    # Here we update the value of our parent, so we start by reversing.
    parent = tf.gather(tree.parents, index, axis=1, batch_dims=1)
    masked_parent = tf.where(cond_mask, parent, 0)
    count = tf.gather(tree.node_visits, masked_parent, axis=1, batch_dims=1)
    action = tf.gather(tree.action_from_parent, index, axis=1, batch_dims=1)
    masked_action = tf.where(cond_mask, action, 0)
    reward = tf.gather(
      tf.gather(tree.children_rewards, masked_parent, axis=1, batch_dims=1),
      masked_action,
      axis=1,
      batch_dims=1
      )
    leaf_value = reward + \
      tf.gather(
        tf.gather(
          tree.children_discounts,
          masked_parent,
          axis=1,
          batch_dims=1),
        masked_action,
        axis=1,
        batch_dims=1
      ) * leaf_value
    parent_value = (
        tf.gather(tree.node_values, masked_parent, axis=1, batch_dims=1) * \
          tf.cast(count,tf.float32) + \
            leaf_value) / \
          (tf.cast(count, tf.float32) + 1.0)
    children_values = tf.gather(tree.node_values, index, axis=1, batch_dims=1)
    children_counts = tf.gather(
      tf.gather(tree.children_visits, masked_parent, axis=1, batch_dims=1),
      masked_action,
      axis=1,
      batch_dims=1
      ) + 1

    tree = tree.replace(
        node_values=tf.where(cond_mask[..., None],
            batch_update(
              tree.node_values,
              parent_value,
              masked_parent
              ),
            tree.node_values),
        node_visits=tf.where(cond_mask[..., None],
            batch_update(
              tree.node_visits,
              count + 1,
              masked_parent
              ),
            tree.node_visits),
        children_values=tf.where(cond_mask[..., None, None],
            batch_update(
              tree.children_values,
              children_values,
              masked_parent,
              masked_action
              ),
            tree.children_values),
        children_visits=tf.where(cond_mask[..., None, None],
            batch_update(
              tree.children_visits,
              children_counts,
              masked_parent,
              masked_action
              ),
            tree.children_visits))

    return tree, leaf_value, masked_parent

  loop_state = (
    tree,
    tf.gather(tree.node_values, leaf_index, axis=1, batch_dims=1),
    leaf_index
  )
  tree, _, _ = tf.while_loop(cond_fun, body_fun, loop_state)

  return tree


def batch_update(x, vals, *indices):
  batch_size = vals.shape[0]
  indices2d = tf.stack([tf.range(batch_size), *list(indices)], -1)
  return tf.tensor_scatter_nd_update(
    x,
    indices2d,
    vals)



def update_tree_node(
    tree: Tree,  # Tree[T]
    node_index: tf.Tensor,
    prior_logits: tf.Tensor,
    value: tf.Tensor,
    embedding: tf.Tensor
    ) -> Tree:  # Tree[T]
  """Updates the tree at node index.

  Args:
    tree: `Tree` to whose node is to be updated.
    node_index: the index of the expanded node. Shape `[B]`.
    prior_logits: the prior logits to fill in for the new node, of shape
      `[B, num_actions]`.
    value: the value to fill in for the new node. Shape `[B]`.
    embedding: the state embeddings for the node. Shape `[B, ...]`.

  Returns:
    The new tree with updated nodes.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(prior_logits), (batch_size, tree.num_actions))

  # When using max_depth, a leaf can be expanded multiple times.
  new_visit = tf.gather(tree.node_visits, node_index, axis=1, batch_dims=1) + 1
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits=batch_update(
          tree.children_prior_logits, prior_logits, node_index),
      raw_values=batch_update(
          tree.raw_values, value, node_index),
      node_values=batch_update(
          tree.node_values, value, node_index),
      node_visits=batch_update(
          tree.node_visits, new_visit, node_index),
      embeddings=tf.nest.map_structure(
          lambda t, s: batch_update(t, s, node_index),
          tree.embeddings, embedding))

  return tree.replace(**updates)


def instantiate_tree_from_root(
    root: base.RootFnOutput,
    num_simulations: int,
    root_invalid_actions: tf.Tensor,
    extra_data: Any) -> Tree:
  """Initializes tree state at search root."""
  if tf.executing_eagerly():
    tf.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape
  if tf.executing_eagerly():
    tf.assert_equal(tf.shape(root.value), [batch_size])
  num_nodes = num_simulations + 1
  data_dtype = root.value.dtype
  batch_node = (batch_size, num_nodes)
  batch_node_action = (batch_size, num_nodes, num_actions)

  def _zeros(x):
    return tf.zeros(batch_node + x.shape[1:], dtype=x.dtype)

  # Create a new empty tree state and fill its root.
  logits_dtype = root.prior_logits.dtype
  tree = Tree(
      node_visits=tf.zeros(batch_node, dtype=tf.int32),
      raw_values=tf.zeros(batch_node, dtype=data_dtype),
      node_values=tf.zeros(batch_node, dtype=data_dtype),
      parents=tf.fill(batch_node, tree_lib.NO_PARENT),
      action_from_parent=tf.fill(batch_node, tree_lib.NO_PARENT),
      children_index=tf.fill(batch_node_action, tree_lib.UNVISITED),
      children_prior_logits=tf.zeros(batch_node_action, dtype=logits_dtype),
      children_values=tf.zeros(batch_node_action, dtype=data_dtype),
      children_visits=tf.zeros(batch_node_action, dtype=tf.int32),
      children_rewards=tf.zeros(batch_node_action, dtype=data_dtype),
      children_discounts=tf.zeros(batch_node_action, dtype=data_dtype),
      embeddings=tf.nest.map_structure(_zeros, root.embedding),
      root_invalid_actions=root_invalid_actions,
      extra_data=extra_data)

  root_index = tf.fill([batch_size], tree_lib.ROOT_INDEX)
  tree = update_tree_node(
      tree, root_index, root.prior_logits, root.value, root.embedding)
  return tree
  