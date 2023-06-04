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

import numpy as np


class StickinessHelper:

    """
    Helper for estimating the hyperparameters: prior and noise covariance,
    prior mean.
    """

    def __init__(self, prior_mvec, prior_cmat, noise_cmat=None):
        self.prior_mvec = prior_mvec
        self.prior_cmat = prior_cmat
        self.noise_cmat = noise_cmat

    @classmethod
    def from_data(cls, data):
        cmats = np.array([np.cov(mat.T) for mat in data.values()])
        noise_cmat = np.mean(cmats, axis=0)
        mvecs = np.array([np.mean(mat, axis=0) for mat in data.values()])
        prior_mvec = np.mean(mvecs, axis=0)
        prior_cmat = np.cov(mvecs.T)
        return cls(prior_mvec, prior_cmat, noise_cmat)
