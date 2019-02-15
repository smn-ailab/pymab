"""This Module contains Contextual Bandit Policies."""
import copy
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


class ContextualPolicyInterface(ABC):
    """Abstract base class for all contextual policies in pymab."""

    @abstractmethod
    def select_action(self, x: np.array) -> int:
        """Select action for new data."""
        pass

    @abstractmethod
    def update_params(self, x: np.array, action: int, reward: Union[int, float]) -> None:
        """Update parameters."""
        pass


class LinUCB(ContextualPolicyInterface):
    """Linear Upper Confidence Bound.

    Parameters
    ----------
    n_actions: int
        The number of given bandit actions.

    n_features: int
        The dimention of context vectors.

    alpha: float, optional(default=1.0)
        The hyper-parameter which represents how often the algorithm explores.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    References
    -------
    [1] L. Li, W. Chu, J. Langford, and E. Schapire.
        A contextual-bandit approach to personalized news article recommendation.
        In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.

    """

    def __init__(self, n_actions: int, n_features: int, alpha: float=1.0, batch_size: int=1) -> None:
        """Initialize class."""
        self.n_actions = n_actions
        self.n_features = n_features
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.action_counts_temp = np.zeros(self.n_actions, dtype=int)
        self.observation_interval = batch_size
        self.num_iter = 0
        self.alpha = alpha
        self.name = f"LinUCB(alpha={self.alpha})"
        self.theta_hat = np.zeros((self.n_features, self.n_actions))  # d * k
        self.A_inv = np.concatenate([np.identity(self.n_features)
                                     for i in np.arange(self.n_actions)]).reshape(self.n_actions, self.n_features, self.n_features)  # k * d * d
        self.b = np.zeros((self.n_features, self.n_actions))  # d * k
        self.A_inv_temp = np.copy(self.A_inv)
        self.b_temp = np.copy(self.b)

    def select_action(self, x: np.ndarray) -> int:
        """Select action for new data.

        Parameters
        ----------
        x : array-like, shape = (n_features, )
            A test sample.

        Returns
        -------
        result: int
            The selected action.

        """
        if 0 in self.action_counts:
            result = np.argmin(self.action_counts)
        else:
            x = np.expand_dims(x, axis=1)
            self.theta_hat = np.concatenate([self.A_inv_temp[i] @ np.expand_dims(self.b_temp[:, i], axis=1)
                                             for i in np.arange(self.n_actions)], axis=1)  # user_dim * n_actions
            sigma_hat = np.sqrt((x.T @ self.A_inv_temp @ x)).reshape((self.n_actions, 1))  # n_actions * 1
            result = np.argmax(self.theta_hat.T @ x + self.alpha * sigma_hat)
        return result

    def update_params(self, x: np.ndarray, action: int, reward: Union[int, float]) -> None:
        """Update parameters.

        Parameters
        ----------
        x : array-like, shape = (n_features, )
            A test sample.

        action: int
            The selected action.

        reward: int, float
            The observed reward value from the selected action.

        """
        x = np.expand_dims(x, axis=1)
        self.num_iter += 1
        self.action_counts[action] += 1
        self.A_inv[action] -= self.A_inv[action] @ x @ x.T @ self.A_inv[action] / (1 + x.T @ self.A_inv[action] @ x)  # d * d
        self.b[:, action] += np.ravel(x) * reward  # d * 1
        if self.num_iter % self.observation_interval == 0:
            self.action_counts_temp = np.copy(self.action_counts)
            self.A_inv_temp, self.b_temp = np.copy(self.A_inv), np.copy(self.b)  # d * d, d * 1


class HybridLinUCB(ContextualPolicyInterface):
    """Hybrid Linear Upper Confidence Bound.

    Parameters
    ----------
    n_actions: int
        The number of given bandit actions.

    z_dim: int,
        The dimensions of context vectors which are common to all actions.

    x_dim:, int
        The dimentions of context vectors which are unique to earch action.

    alpha: float, optional(default=1.0)
        The hyper-parameter which represents how often the algorithm explores.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    References
    -------
    [1] L. Li, W. Chu, J. Langford, and E. Schapire.
        A contextual-bandit approach to personalized news article recommendation.
        In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.

    """

    def __init__(self, n_actions: int, z_dim: int, x_dim: int, alpha: float=1.0, batch_size: int=1) -> None:
        """Initialize class."""
        self.n_actions = n_actions
        self.z_dim = z_dim  # k
        self.x_dim = x_dim  # d
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.action_counts_temp = np.zeros(self.n_actions, dtype=int)
        self.observation_interval = batch_size
        self.num_iter = 0
        self.alpha = alpha
        self.name = f"HybridLinUCB(alpha={self.alpha})"

        self.beta = np.zeros(self.z_dim)
        self.theta_hat = np.zeros((self.x_dim, self.n_actions))  # d * k

        # matrices which are common to all context
        self.A0, self.b0 = np.identity(self.z_dim), np.zeros((self.z_dim, 1))  # k * k, k * 1
        self.A_inv = np.concatenate([np.identity(self.x_dim)
                                     for i in np.arange(self.n_actions)]).reshape(self.n_actions, self.x_dim, self.x_dim)  # k * d * d
        self.B = np.zeros((self.n_actions, self.x_dim, self.z_dim))
        self.b = np.zeros((self.x_dim, self.n_actions))
        self.A0_temp, self.b0_temp = np.copy(self.A0), np.copy(self.b0)
        self.A_inv_temp, self.B_temp, self.b_temp = np.copy(self.A_inv), np.copy(self.B), np.copy(self.b)

    def select_action(self, x: np.ndarray) -> int:
        """Select actions according to the policy for new data.

        Parameters
        ----------
        x : array-like, shape = (n_features, )
            A test sample.

        Returns
        -------
        result: int
            The selected action.

        """
        if 0 in self.action_counts:
            result = np.argmin(self.action_counts)
        else:
            z, x = np.expand_dims(x[:self.z_dim]), np.expand_dims(x[self.z_dim:])
            self.beta = np.linalg.inv(self.A0) @ self.b0  # k * 1
            self.theta_hat = np.concatenate([(self.A_inv[i] @ (np.expand_dims(self.b[:, i], axis=1) - self.B[i] @ self.beta))
                                             for i in np.arange(self.n_actions)], axis=1)
            sigma1 = z.T @ np.linalg.inv(self.A0) @ z
            sigma2 = - 2 * np.concatenate([z.T @ np.linalg.inv(self.A0) @ self.B[i].T @ self.A_inv[i] @ x
                                           for i in np.arange(self.n_actions)], axis=1)
            sigma3 = np.concatenate([x.T @ self.A_inv[i] @ x for i in np.arange(self.n_actions)], axis=1)
            sigma4 = np.concatenate([x.T @ self.A_inv[i] @ self.B[i] @ np.linalg.inv(self.A0) @ self.B[i].T @ self.A_inv[i] @ x
                                     for i in np.arange(self.n_actions)], axis=1)
            sigma_hat = sigma1 + sigma2 + sigma3 + sigma4
            result = np.argmax(z.T @ self.beta + x.T @ self.theta_hat + self.alpha * sigma_hat)
        return result

    def update_params(self, x: np.ndarray, action: int, reward: float) -> None:
        """Update parameters.

        Parameters
        ----------
        x : array-like, shape = (n_features, )
            A test sample.

        action: int
            The selected action.

        reward: int, float
            The observed reward value from the selected action.

        """
        z, x = np.expand_dims(x[:self.z_dim]), np.expand_dims(x[self.z_dim:])
        self.num_iter += 1
        self.action_counts[action] += 1
        self.A0 += self.B[action].T @ self.A_inv[action] @ self.B[action]
        self.b0 += self.B[action].T @ self.A_inv[action] @ self.b[action]
        self.A_inv[action] -= self.A_inv[action] @ x @ x.T @ self.A_inv[action] / (1 + x.T @ self.A_inv[action] @ x)
        self.B[action] += x @ z.T
        self.b[:, action] += np.ravel(x) * reward
        self.A0 += z @ z.T - self.B[action].T @ self.A_inv[action] @ self.B[action]
        self.b0 += z * reward - self.B[action].T @ self.A_inv[action] @ np.expand_dims(self.b[:, action], axis=1)

        if self.num_iter % self.observation_interval == 0:
            self.A0_temp, self.b0_temp = np.copy(self.A0), np.copy(self.b0)
            self.A_inv_temp, self.B_temp, self.b_temp = np.copy(self.A_inv), np.copy(self.B), np.copy(self.b)


class LinTS(ContextualPolicyInterface):
    """Linear Thompson Sampling.

    Parameters
    ----------
    n_actions: int
        The number of given bandit actions.

    n_features: int
        The dimention of context vectors.

    sigma: float, optional(default=1.0)
        The variance of prior gaussian distribution.

    sample_batch: int, optional (default=1)
        How often the policy sample new parameters.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    References
    -------
    [1] 本多淳也, 中村篤祥. バンディット問題の理論とアルゴリズム. 講談社 機械学習プロフェッショナルシリーズ. 2016.

    """

    def __init__(self, n_actions: int, n_features: int, sigma: float=1.0, sample_batch: int=1, batch_size: int=1) -> None:
        """Initialize class."""
        self.n_actions = n_actions
        self.n_features = n_features
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.action_counts_temp = np.zeros(self.n_actions, dtype=int)
        self.observation_interval = batch_size
        self.num_iter = 0
        self.sigma = sigma
        self.sample_batch = sample_batch
        self.name = f"LinTS(sigma={self.sigma})"
        self.theta_hat, self.theta_tilde = np.zeros((self.n_features, self.n_actions)), np.zeros((self.n_features, self.n_actions))
        self.A_inv = np.concatenate([np.identity(self.n_features)
                                     for i in np.arange(self.n_actions)]).reshape(self.n_actions, self.n_features, self.n_features)  # k * d * d
        self.b = np.zeros((self.n_features, self.n_actions))  # d * k
        self.A_inv_temp = np.copy(self.A_inv)
        self.b_temp = np.copy(self.b)

    def select_action(self, x: np.ndarray) -> int:
        """Select actions according to the policy for new data.

        Parameters
        ----------
        x : array-like, shape = (n_features, )
            A test sample.

        Returns
        -------
        result: int
            The selected action.

        """
        if 0 in self.action_counts:
            result = np.argmin(self.action_counts)
        else:
            if self.num_iter % self.sample_batch == 0:
                x = np.expand_dims(x, axis=1)
                self.theta_hat = np.concatenate([self.A_inv[i] @ np.expand_dims(self.b[:, i], axis=1)
                                                 for i in np.arange(self.n_actions)], axis=1)
                self.theta_tilde = np.concatenate([np.expand_dims(np.random.multivariate_normal(self.theta_hat[:, i], self.A_inv[i]), axis=1)
                                                   for i in np.arange(self.n_actions)], axis=1)
            result = np.argmax(x.T @ self.theta_tilde)

        return result

    def update_params(self, x: np.ndarray, action: int, reward: Union[int, float]) -> None:
        """Update parameters.

        Parameters
        ----------
        x : array-like, shape = (n_features, )
            A test sample.

        action: int
            The selected action.

        reward: int, float
            The observed reward value from the selected action.

        """
        x = np.expand_dims(x, axis=1)
        self.num_iter += 1
        self.action_counts[action] += 1
        self.A_inv[action] -= self.A_inv[action] @ x @ x.T @ self.A_inv[action] / (1 + x.T @ self.A_inv[action] @ x)  # d * d
        self.b[:, action] += np.ravel(x) * reward  # d * 1
        if self.num_iter % self.observation_interval == 0:
            self.action_counts_temp = np.copy(self.action_counts)
            self.A_inv_temp, self.b_temp = np.copy(self.A_inv), np.copy(self.b)  # d * d, d * 1


class LogisticTS(ContextualPolicyInterface):
    """Logistic Thompson Sampling.

    Parameters
    ----------
    n_actions: int
        The number of given bandit actions.

    n_features: int
        The dimention of context vectors.

    sigma: float, optional(default=1.0)
        The variance of prior gaussian distribution.

    n_iter: int, optional(default=1)
        The num of iteration of newton method in each parameter update.

    sample_batch: int, optional (default=1)
        How often the policy sample new parameters.

    batch_size: int, optional (default=1)
        The number of data given in each batch.

    References
    -------
    [1] 本多淳也, 中村篤祥. バンディット問題の理論とアルゴリズム. 講談社 機械学習プロフェッショナルシリーズ, 2016.

    [2] O. Chapelle, L. Li. An Empirical Evaluation of Thompson Sampling. In NIPS, pp. 2249–2257, 2011.

    """

    def __init__(self, n_actions: int, n_features: int, sigma: float=0.1,
                 n_iter: int=1, sample_batch: int=1,  batch_size: int=1) -> None:
        """Initialize Class."""
        self.n_actions = n_actions
        self.n_features = n_features
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.action_counts_temp = np.zeros(self.n_actions, dtype=int)
        self.observation_interval = batch_size
        self.num_iter = 0
        self.sigma = sigma
        self.n_iter = n_iter
        self.sample_batch = sample_batch
        self.name = f"LogisticTS(sigma={self.sigma})"
        self.data_stock: list = [[] for i in np.arange(self.n_actions)]
        self.reward_stock: list = [[] for i in np.arange(self.n_actions)]
        # array - (n_actions * user_dim),
        self.theta_hat, self.theta_tilde = np.zeros((self.n_features, self.n_actions)), np.zeros((self.n_features, self.n_actions))
        self.hessian_inv = np.concatenate([np.identity(self.n_features)
                                           for i in np.arange(self.n_actions)]).reshape(self.n_actions, self.n_features, self.n_features)

    def select_action(self, x: np.ndarray) -> int:
        """Select actions according to the policy for new data.

        Parameters
        ----------
        x : array-like, shape = (n_features, )
            A test sample.

        Returns
        -------
        result: int
            The selected action.

        """
        if 0 in self.action_counts:
            result = np.argmin(self.action_counts)
        else:
            if self.num_iter % self.sample_batch == 0:
                x = np.expand_dims(x, axis=1)
                self.theta_tilde = np.concatenate([np.expand_dims(np.random.multivariate_normal(self.theta_hat[:, i], self.hessian_inv[i]), axis=1)
                                                   for i in np.arange(self.n_actions)], axis=1)
            result = np.argmax(x.T @ self.theta_tilde)
        return result

    def update_params(self, x: np.ndarray, action: int, reward: Union[int, float]) -> None:
        """Update parameters.

        Parameters
        ----------
        x : array-like, shape = (n_features, )
            A test sample.

        action: int
            The selected action.

        reward: int, float
            The observed reward value from the selected action.

        """
        x = np.expand_dims(x, axis=1)
        self.action_counts[action] += 1
        self.data_stock[action].append(x)  # (user_dim + action_dim) * 1
        self.reward_stock[action].append(reward)
        self.num_iter += 1

        if self.num_iter % self.observation_interval == 0:
            for i in np.arange(self.n_iter):
                self.theta_hat[:, action], self.hessian_inv[action] = \
                    self._update_theta_hat(action, self.theta_hat[:, action])

    def _calc_gradient(self, action: int, theta_hat: np.ndarray) -> np.ndarray:
        _hat = np.expand_dims(theta_hat, axis=1)
        _gradient = _hat / self.sigma
        _data = np.concatenate(self.data_stock[action], axis=1)  # action_dim * n_user
        _gradient += np.expand_dims(np.sum(_data * (np.exp(_hat.T @ _data) / (1 + np.exp(_hat.T @ _data))), axis=1), axis=1)
        _gradient -= np.expand_dims(np.sum(_data[:, np.array(self.reward_stock[action]) == 1], axis=1), axis=1)
        return _gradient

    def _calc_hessian(self, action: int, theta_hat: np.ndarray) -> np.ndarray:
        _hat = np.expand_dims(theta_hat, axis=1)
        _hessian = np.identity(self.n_features) / self.sigma
        _data = np.concatenate(self.data_stock[action], axis=1)
        mat = [np.expand_dims(_data[:, i], axis=1) @ np.expand_dims(_data[:, i], axis=1).T
               for i in np.arange(self.action_counts[action])]
        weight = np.ravel(np.exp(_hat.T @ _data) / (1 + np.exp(_hat.T @ _data)) ** 2)  # 1 * data_size
        _hessian += np.sum(
            np.concatenate([_mat * w for _mat, w in zip(mat, weight)], axis=0).reshape(self.action_counts[action],
                                                                                       self.n_features,
                                                                                       self.n_features), axis=0)
        return _hessian

    def _update_theta_hat(self, action: int, theta_hat: np.ndarray) -> np.ndarray:
        _theta_hat = np.expand_dims(theta_hat, axis=1)  # (user_dim * action_dim) * 1
        _gradient = self._calc_gradient(action, theta_hat)
        _hessian_inv = np.linalg.inv(self._calc_hessian(action, theta_hat))
        _theta_hat -= _hessian_inv @ _gradient
        return np.ravel(_theta_hat), _hessian_inv
