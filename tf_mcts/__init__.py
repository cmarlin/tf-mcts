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
"""tf_mcts: Monte Carlo tree search in Tensorflow."""

from tf_mcts._src.action_selection import gumbel_muzero_interior_action_selection
from tf_mcts._src.action_selection import gumbel_muzero_root_action_selection
from tf_mcts._src.action_selection import GumbelMuZeroExtraData
from tf_mcts._src.action_selection import muzero_action_selection
from tf_mcts._src.base import ChanceRecurrentFnOutput
from tf_mcts._src.base import DecisionRecurrentFnOutput
from tf_mcts._src.base import InteriorActionSelectionFn
from tf_mcts._src.base import LoopFn
from tf_mcts._src.base import PolicyOutput
from tf_mcts._src.base import RecurrentFn
from tf_mcts._src.base import RecurrentFnOutput
from tf_mcts._src.base import RecurrentState
from tf_mcts._src.base import RootActionSelectionFn
from tf_mcts._src.base import RootFnOutput
from tf_mcts._src.policies import gumbel_muzero_policy
from tf_mcts._src.policies import muzero_policy
from tf_mcts._src.policies import stochastic_muzero_policy
from tf_mcts._src.qtransforms import qtransform_by_min_max
from tf_mcts._src.qtransforms import qtransform_by_parent_and_siblings
from tf_mcts._src.qtransforms import qtransform_completed_by_mix_value
from tf_mcts._src.search import search
from tf_mcts._src.tree import Tree

__version__ = "0.0.1"

__all__ = (
    "ChanceRecurrentFnOutput",
    "DecisionRecurrentFnOutput",
    "GumbelMuZeroExtraData",
    "InteriorActionSelectionFn",
    "LoopFn",
    "PolicyOutput",
    "RecurrentFn",
    "RecurrentFnOutput",
    "RecurrentState",
    "RootActionSelectionFn",
    "RootFnOutput",
    "Tree",
    "gumbel_muzero_interior_action_selection",
    "gumbel_muzero_policy",
    "gumbel_muzero_root_action_selection",
    "muzero_action_selection",
    "muzero_policy",
    "qtransform_by_min_max",
    "qtransform_by_parent_and_siblings",
    "qtransform_completed_by_mix_value",
    "search",
    "stochastic_muzero_policy",
)

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the tf_mcts public API.    /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
