# Copyright 2023 Spotify AB
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

import collections
import numpy as np


class BayesianBandit:

    """Thompson-sampling bandit."""

    def __init__(self, beliefs):
        """Initialize bandit."""
        self.beliefs = beliefs  # Dict mapping item ID to belief object.
        self.t = 0  # Step counter.

    def act(self, n_actions=1, admissible=None):
        """Choose a number of actions given current beliefs.

        Parameters
        ----------
        n_actions : int, optional
            Number of actions to sample. Default: 1.
        admissible : list, optional
            List of admissible actions. If `None` (default), all actions are
            admissible.
        """
        actions = list()
        if admissible is None:
            # All actions are admissible.
            admissible = list(self.beliefs.keys())
        for _ in range(n_actions):
            # Get a sample from the current belief distribution for each item, and
            # keep the item with the highest sampled value.
            samples = [
                self.beliefs[a].sample_from_posterior(self.t) for a in admissible
            ]
            idx = np.argmax(samples)
            actions.append(admissible[idx])
        return actions

    def update(self, item, traces):
        """Update belief based on observation."""
        self.beliefs[item].update(traces, self.t)

    def step(self):
        """increment step counter."""
        self.t += 1
