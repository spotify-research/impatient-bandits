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
import numpy as np
import scipy.linalg as spl


class GaussianBelief(metaclass=abc.ABCMeta):

    """Abstract base class for Gaussian stickiness beliefs."""

    def __init__(self, mean, var, seed=None):
        self.mean = mean
        self.var = var
        # Random number generator
        self.rng = np.random.default_rng(seed=seed)
        # Timestep at which posterior last computed.
        self.t_post = 0
        # Whether there is new data which needs to be conditioned upon.
        self.has_new_obs = False

    @abc.abstractmethod
    def compute_posterior(self, t):
        """Compute posterior using data up to timestep `t`."""

    def sample_from_posterior(self, t):
        # Only re-compute the posterior if a) we have new observations available
        # or b) we have not already computed the posterior for this timestep.
        if self.has_new_obs or t != self.t_post:
            self.compute_posterior(t)
        return self.mean + np.sqrt(self.var) * self.rng.standard_normal()


class ProgressiveBelief(GaussianBelief):

    """Stickiness belief making use of partial observations."""

    def __init__(
        self,
        prior_mvec,
        prior_cmat,
        noise_cmat,
        cov_estimator="fixed",
        seed=None,
    ):
        self.prior_mvec = prior_mvec  # w-dim prior mean vector.
        self.prior_cmat = prior_cmat  # w x w prior covariance matrix.
        self.noise_cmat = noise_cmat  # w x w noise covariance matrix.
        self.w = len(prior_mvec)  # Stickiness window size / feedback delay.
        self.cov_estimator = cov_estimator  # Options: fixed, regularized.
        # Data.
        self.traces = np.zeros((0, self.w))  # Activity traces.
        self.ts = np.zeros(0, dtype=int)  # Starting timesteps.
        super().__init__(
            mean=np.sum(self.prior_mvec),
            var=np.sum(self.prior_cmat),
            seed=seed,
        )

    def update(self, traces, t):
        """Add observation.

        Parameters
        ----------
        traces : ndarray, shape (m, w) or (w,)
            User activity traces.
        t : int
            Timestep of first observation.
        """
        m = np.atleast_2d(traces).shape[0]
        self.traces = np.vstack((self.traces, traces))
        self.ts = np.hstack((self.ts, np.repeat(t, m)))
        self.has_new_obs = True

    def compute_posterior(self, t):
        """Compute mean & variance of belief at step `t`."""
        # Step 1: filter observations.
        m = len(self.ts)
        # Binary mask indicating entries observed at time `t`.
        mask = np.tile(np.arange(self.w), (m, 1)) < (t - self.ts)[:, None]
        mat = np.zeros_like(self.traces)
        mat[mask] = self.traces[mask]
        n_users = np.sum(mask, axis=0)

        # Step 2: Compute empirical mean and noise covariance.
        # `k` is the first timestep with no observations.
        k = np.min(np.argwhere(n_users == 0), initial=self.w)
        empirical_mvec = mat[:, :k].sum(axis=0) / n_users[:k]
        denom_min = np.minimum(n_users[:k, None], n_users[None, :k])
        denom_max = np.maximum(n_users[:k, None], n_users[None, :k])
        if self.cov_estimator == "fixed":
            noise_cmat = self.noise_cmat[:k, :k] / denom_max
        elif self.cov_estimator == "regularized":
            tmp = mat[:, :k] - empirical_mvec
            noise_cmat = (tmp.T @ tmp + self.noise_cmat[:k, :k]) / (
                (denom_min + 1) * denom_max
            )
        else:
            raise ValueError(f"method {self.cov_estimator} unknown")

        # Step 3: Combine with prior to compute posterior.
        # Kernel plus noise.
        kpn = self.prior_cmat[:k, :k] + noise_cmat
        # Inverse kernel plus noise.
        ikpn = spl.solve(kpn, np.eye(len(kpn)), assume_a="pos")
        post_mvec = self.prior_mvec + self.prior_cmat[:, :k] @ ikpn @ (
            empirical_mvec - self.prior_mvec[:k]
        )
        post_cmat = (
            self.prior_cmat - self.prior_cmat[:, :k] @ ikpn @ self.prior_cmat[:k, :]
        )

        # Update univariate stickiness belief.
        self.mean = np.sum(post_mvec)
        self.var = np.sum(post_cmat)
        self.t_post = t
        self.has_new_obs = False


class UnivariateBelief(GaussianBelief):

    """Base class for univariate stickiness beliefs."""

    def __init__(
        self,
        prior_mvec,
        prior_cmat,
        noise_cmat,
        cov_estimator="fixed",
        seed=None,
    ):
        self.prior_mean = np.sum(prior_mvec)
        self.prior_var = np.sum(prior_cmat)
        self.noise_var = np.sum(noise_cmat)
        self.cov_estimator = cov_estimator  # Options: fixed, regularized.
        self.w = len(prior_mvec)  # Stickiness window size / feedback delay.
        # Data.
        self.ys = np.zeros(0)  # Scalar observations.
        self.ts = np.zeros(0, dtype=int)  # Timestep of observation.
        super().__init__(
            mean=self.prior_mean,
            var=self.prior_var,
            seed=seed,
        )

    @abc.abstractmethod
    def update(self, traces, t):
        """Add observation.

        Parameters
        ----------
        traces : ndarray, shape (m, w) or (w,)
            User activity traces.
        t : int
            Timestep of first observation.
        """

    def compute_posterior(self, t):
        """Compute mean & variance of belief at step `t`."""
        ys = self.ys[self.ts < t]
        m = len(ys)
        # If no observations, set posterior to the prior
        if m == 0:
            self.var = self.prior_var
            self.mean = self.prior_mean
        # Otherwise, compute posterior by conditioning on observations
        else:
            if self.cov_estimator == "fixed":
                noise_var = self.noise_var / m
            elif self.cov_estimator == "regularized":
                noise_var = (np.var(ys) + self.noise_var / m) / (m + 1)
            else:
                raise ValueError(f"method {self.cov_estimator} unknown")
            self.var = 1 / (1 / self.prior_var + 1 / noise_var)
            self.mean = self.var * (
                self.prior_mean / self.prior_var + np.mean(ys) / noise_var
            )
        self.t_post = t
        self.has_new_obs = False


class OracleBelief(UnivariateBelief):

    """Stickiness belief assuming immediate access to the long-term reward."""

    def update(self, traces, t):
        traces = np.atleast_2d(traces)
        self.ys = np.hstack((self.ys, np.sum(traces, axis=1)))
        self.ts = np.hstack((self.ts, np.repeat(t, traces.shape[0])))
        self.has_new_obs = True


class DelayedBelief(UnivariateBelief):

    """Stickiness belief assuming immediate access to the long-term reward."""

    def update(self, traces, t):
        traces = np.atleast_2d(traces)
        self.ys = np.hstack((self.ys, np.sum(traces, axis=1)))
        # Add delay to timestamp.
        self.ts = np.hstack((self.ts, np.repeat(t + self.w, traces.shape[0])))
        self.has_new_obs = True


class DayTwoBelief(UnivariateBelief):

    """Using day-two activity as a proxy for stickiness."""

    def __init__(
        self,
        prior_mvec,
        prior_cmat,
        noise_cmat,
        cov_estimator="fixed",
        seed=None,
    ):
        self.prior_mean = prior_mvec[0]
        self.prior_var = prior_cmat[0, 0]
        self.noise_var = noise_cmat[0, 0]
        self.cov_estimator = cov_estimator  # Options: fixed, regularized.
        # Data.
        self.ys = np.zeros(0)  # Scalar observations.
        self.ts = np.zeros(0, dtype=int)  # Timestep of observation.
        GaussianBelief.__init__(
            self,
            mean=self.prior_mean,
            var=self.prior_var,
            seed=seed,
        )

    def update(self, traces, t):
        traces = np.atleast_2d(traces)
        # Pick out activity on the first day.
        self.ys = np.hstack((self.ys, traces[:, 0]))
        self.ts = np.hstack((self.ts, np.repeat(t, traces.shape[0])))
        self.has_new_obs = True


class DummyBelief:

    """Dummy belief that makes the agent act randomly."""

    def __init__(self, seed=None, **kwargs):
        self.rng = np.random.default_rng(seed=seed)

    def update(self, traces, t):
        pass

    def sample_from_posterior(self, t):
        return self.rng.standard_normal()
