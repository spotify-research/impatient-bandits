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


class Environment:
    def __init__(self, dists):
        # Dict mapping item ID to distribution of activity traces.
        self.dists = dists
        # Dict mapping timestep to actions.
        self.history = collections.defaultdict(list)

    def step(self, action, t, n_samples=1):
        self.history[t].append(action)
        return self.dists[action].sample(n=n_samples)

    def reset(self):
        for dist in self.dists.values():
            dist.reset()
        self.history = collections.defaultdict(list)
