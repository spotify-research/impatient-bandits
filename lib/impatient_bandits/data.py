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

import abc
import itertools
import numpy as np
import scipy.stats as sps
import warnings


class Distribution(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, n=1):
        """Generate samples from the distribution."""

    def reset(self):
        pass


class EmpiricalDistribution(Distribution):

    """Empirical distribution of a dataset."""

    def __init__(self, traces, seed=42):
        self.rng = np.random.default_rng(seed=seed)
        self.traces = traces
        self.mvec = np.mean(traces, axis=0)
        self.reset()

    @property
    def mean_sum(self):
        """Compute ground truth stickiness."""
        # Add 1.0 as we assume activity on the first day.
        return 1.0 + np.sum(self.mvec)

    def sample(self, n=1):
        """Sample user traces from the dataset without replacement."""
        traces = np.array(list(itertools.islice(self.itr, n)))
        self.consumed += n
        if self.consumed > len(self.traces):
            warnings.warn("sampled more user traces than available")
        return traces

    def reset(self):
        """Reset the sampler."""
        self.rng.shuffle(self.traces)
        self.itr = itertools.cycle(self.traces)
        self.consumed = 0


class IIDBernoulliDistribution(Distribution):

    """IID Bernoullis with prescribed probability."""

    def __init__(self, p, w=59, seed=42):
        self.p = p
        self.w = w
        self.rng = np.random.default_rng(seed=seed)

    @property
    def mean_sum(self):
        """Compute ground truth stickiness."""
        # Add 1.0 as we assume activity on the first day.
        return 1.0 + (self.w * self.p)

    def sample(self, n=1):
        return self.rng.binomial(1, self.p, size=(n, self.w)).astype(float)


class GaussianDistribution(Distribution):

    """Multivariate Gaussian distribution over real-valued vectors."""

    def __init__(self, mvec, cmat, seed=42):
        self.chol = np.linalg.cholesky(cmat)
        self.mvec = mvec
        self.w = len(mvec)
        self.rng = np.random.default_rng(seed=seed)

    @property
    def mean_sum(self):
        """Compute ground truth stickiness."""
        # Add 1.0 as we assume activity on the first day.
        return 1.0 + np.sum(self.mvec)

    def sample(self, n=1):
        return self.mvec + self.rng.normal(size=(n, self.w)) @ self.chol.T

    @classmethod
    def niw_generator(cls, mu, psi, nu, lm, seed=42):
        rng = np.random.default_rng(seed=seed)
        while True:
            cmat = sps.invwishart.rvs(df=nu, scale=psi, random_state=rng)
            mvec = rng.multivariate_normal(mu, cmat / lm)
            yield cls(mvec, cmat, seed=rng)


class BinaryDistribution(Distribution):

    """
    Distribution over binary vectors with prescribed mean and correlation
    matrix.
    """

    def __init__(self, mat, seed=42):
        self.rng = np.random.default_rng(seed=seed)
        # `mat` is the Gaussian covariance matrix H defined here:
        # <https://mathoverflow.net/a/219427>
        self.chol = np.linalg.cholesky(mat)
        self.mvec = np.arcsin(mat[-1, :-1]) / np.pi + 0.5

    @property
    def mean_sum(self):
        """Compute ground truth stickiness."""
        # Add 1.0 as we assume activity on the first day.
        return 1.0 + np.sum(self.mvec)

    def sample(self, n=1):
        samples = np.sign(self.rng.normal(size=(n, len(self.chol))) @ self.chol.T)
        samples = samples[:, :-1] * samples[:, -1:]
        return (samples > 0).astype(float)

    @classmethod
    def from_traces(cls, traces, seed=42):
        traces = 2 * traces - 1  # Transform from {0, 1} to {-1, +1}.
        mvec = np.mean(traces, axis=0)
        smat = traces.T @ traces / len(traces)
        return cls.from_moments(mvec, smat, seed=seed)

    @classmethod
    def from_moments(cls, mvec, smat, seed=42):
        w = len(mvec)
        mat = np.zeros((w + 1, w + 1))
        mat[:w, :w] = smat
        mat[:w, w] = mat[w, :w] = mvec
        mat[w, w] = 1.0
        transf = BinaryDistribution.project(np.sin(np.pi * mat / 2))
        return cls(transf, seed=seed)

    @staticmethod
    def project(mat):
        """Project Gaussian covariance matrix onto feasible set."""
        # Find closest PD matrix w.r.t. Frobenius norm.
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals[eigvals < 1e-6] = 1e-6
        mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Rescale matrix to ensure unit diagonal
        scale = np.diag(np.sqrt(1 / np.diag(mat)))
        return scale @ mat @ scale
