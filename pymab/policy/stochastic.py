"""This Module contains Stochastic Bandit Policies."""
from abc import ABC, abstractmethod
from typing import Union

import numpy as np


class PolicyInterface(ABC):
    """Abstract Base class for all stochastic policies."""

    @abstractmethod
    def select_action(self) -> int:
        """Select action for new data."""
        pass

    @abstractmethod
    def update_params(self, chosen_arm: int, reward: Union[int, float]) -> None:
        """Update parameters."""
        pass


class EpsilonGreedy(PolicyInterface):
    """Epsilon-Greedy.

    Parameters
    ----------
    n_actions: int
        The number of actions.

    epsilon: float
        Probability of taking a random action.

    observation_interval: int, optional (default=1)
        The number of data given in each batch.

    """

    _policy_type = "stochastic"

    def __init__(self, n_actions: int, epsilon: float, observation_interval: int=1) -> None:
        """Initialize class."""
        self.n_actions = n_actions
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.observation_interval = observation_interval
        self.num_iter = 0
        self.epsilon = epsilon
        self.estimated_rewards = np.zeros(self.n_actions)
        self.observed_rewards = np.zeros(self.n_actions)
        self.name = f"EpsilonGreedy(eps={self.epsilon})"

    def select_action(self) -> int:
        """Select action for new data.

        Returns
        -------
        result: int
            The selected action.

        """
        result = np.random.randint(self.n_actions)
        if np.random.rand() > self.epsilon:
            result = np.argmax(self.observed_rewards)
        return result

    def update_params(self, action: int, reward: Union[int, float]) -> None:
        """Update the reward information about earch action.

        Parameters
        ----------
        action: int
            The selected action.

        reward: int, float
            The observed reward value.

        """
        self.num_iter += 1
        self.action_counts[action] += 1
        n, old_reward = self.action_counts[action], self.estimated_rewards[action]
        self.estimated_rewards[action] = (old_reward * (n - 1) / n) + (reward / n)

        if self.num_iter % self.observation_interval == 0:
            self.observed_rewards = np.copy(self.estimated_rewards)


class SoftMax(PolicyInterface):
    """SoftMax.

    Parameters
    ----------
    n_actions: int
        The number of given bandit actions.

    tau: float
        Softmax hyper-parameter.

    observation_interval: int, optional (default=1)
        The number of data given in each batch.

    """

    _policy_type = "stochastic"

    def __init__(self, n_actions: int, tau: float, observation_interval: int=1) -> None:
        """Initialize class."""
        self.n_actions = n_actions
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.observation_interval = observation_interval
        self.num_iter = 0
        self.tau = tau
        self.estimated_rewards = np.zeros(self.n_actions)
        self.observed_rewards = np.zeros(self.n_actions)
        self.name = f"SoftMax(tau={self.tau})"

    def select_action(self) -> int:
        """Select action for new data.

        Returns
        -------
        result: int
            The selected action.

        """
        z = np.sum(np.exp(self.observed_rewards) / self.tau)
        probs = (np.exp(self.estimated_rewards) / self.tau) / z
        return np.random.choice(self.n_actions, p=probs)

    def update_params(self, action: int, reward: Union[int, float]) -> None:
        """Update parameters.

        Parameters
        ----------
        action: int
            The selected action.

        reward: int, float
            The observed reward value.

        """
        self.num_iter += 1
        self.action_counts[action] += 1
        n, old_reward = self.action_counts[action], self.estimated_rewards[action]
        self.estimated_rewards[action] = (old_reward * (n - 1) / n) + (reward / n)

        if self.num_iter % self.observation_interval == 0:
            self.observed_rewards = np.copy(self.estimated_rewards)


class UCB(PolicyInterface):
    """Upper Confidence Bound.

    Parameters
    ----------
    n_actions: int
        The number of given bandit actions.

    observation_interval: int, optional (default=1)
        The number of data given in each batch.

    """

    _policy_type = "stochastic"

    def __init__(self, n_actions: int, alpha: float=0.5, observation_interval: int=1) -> None:
        """Initialize class."""
        self.n_actions = n_actions
        self.alpha = alpha
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.observed_action_counts = np.zeros(self.n_actions, dtype=int)
        self.observation_interval = observation_interval
        self.num_iter = 0
        self.estimated_rewards = np.zeros(self.n_actions)
        self.observed_rewards = np.zeros(self.n_actions)
        name = f"UCB(alpha={self.alpha})"

    def select_action(self) -> int:
        """Select actions according to the policy for new data.

        Returns
        -------
        result: int
            The selected action.

        """
        if 0 in self.action_counts:
            result = np.argmin(self.action_counts)
        else:
            ucb_values = np.zeros(self.n_actions)
            bounds = np.sqrt(self.alpha * np.log(np.sum(self.observed_action_counts)) / self.observed_action_counts)
            result = np.argmax(self.observed_rewards + bounds)

        return result

    def update_params(self, action: int, reward: Union[int, float]) -> None:
        """Update parameters.

        Parameters
        ----------
        action: int
            The selected action.

        reward: int, float
            The observed reward value.

        """
        self.num_iter += 1
        self.action_counts[action] += 1
        n, old_reward = self.action_counts[action], self.estimated_rewards[action]
        self.estimated_rewards[action] = (old_reward * (n - 1) / n) + (reward / n)

        if self.num_iter % self.observation_interval == 0:
            self.observed_action_counts = np.copy(self.action_counts)
            self.observed_rewards = np.copy(self.estimated_rewards)


class UCBTuned(PolicyInterface):
    """Uppler Confidence Bound Tuned.

    Parameters
    ----------
    n_actions: int
        The number of given bandit actions.

    observation_interval: int, optional (default=1)
        The number of data given in each batch.

    """

    _policy_type = "stochastic"
    name = "UCBTuned"

    def __init__(self, n_actions: int, observation_interval: int=1) -> None:
        """Initialize class."""
        self.n_actions = n_actions
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.observed_action_counts = np.zeros(self.n_actions, dtype=int)
        self.observation_interval = observation_interval
        self.num_iter = 0
        self.estimated_rewards = np.zeros(self.n_actions)
        self.observed_rewards = np.zeros(self.n_actions)
        self.sigma = np.zeros(self.n_actions, dtype=float)
        self.sigma_temp = np.zeros(self.n_actions, dtype=float)

    def select_action(self) -> int:
        """Select action for new data.

        Returns
        -------
        result: int
            The selected action.

        """
        if 0 in self.action_counts:
            result = np.argmin(self.observed_action_counts)
        else:
            ucb_values, total_counts = np.zeros(self.n_actions), np.sum(self.observed_action_counts)
            bounds1 = np.log(total_counts) / self.observed_action_counts
            bounds2 = np.minimum(1 / 4, self.sigma_temp + 2 * np.log(total_counts) / self.observed_action_counts)
            result = np.argmax(self.observed_rewards + np.sqrt(bounds1 * bounds2))

        return result

    def update_params(self, action: int, reward: Union[int, float]) -> None:
        """Update parameters.

        Parameters
        ----------
        action: int
            The selected action.

        reward: int, float
            The observed reward value.

        """
        self.num_iter += 1
        self.action_counts[action] += 1
        n, old_reward = self.action_counts[action], self.estimated_rewards[action]
        self.estimated_rewards[action] = (old_reward * (n - 1) / n) + (reward / n)
        new_sigma = ((n * ((self.sigma[action] ** 2) + (self.estimated_rewards[action] ** 2)) + reward ** 2) / (n + 1)) - self.estimated_rewards[action] ** 2
        self.sigma[action] = new_sigma

        if self.num_iter % self.observation_interval == 0:
            self.observed_action_counts = np.copy(self.action_counts)
            self.observed_rewards = np.copy(self.estimated_rewards)
            self.sigma_temp = np.copy(self.sigma)


class BernoulliTS(PolicyInterface):
    """Thompson Sampling for Bernoulli Distribution.

    Parameters
    ----------
    n_actions: int
        The number of given bandit actions.

    alpha: float (default=1.0)
        Hyperparameter alpha for beta distribution.

    beta: float (default=1.0)
        Hyperparameter beta for beta distribution.

    observation_interval: int, optional (default=1)
        The number of data given in each batch.

    """

    _policy_type = "stochastic"
    name = "BernoulliThompsonSampling"

    def __init__(self, n_actions: int, alpha: float=1.0, beta: float=1.0, observation_interval: int=1) -> None:
        """Initialize class."""
        self.n_actions = n_actions
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.alpha = alpha
        self.beta = beta
        self.observation_interval = observation_interval
        self.num_iter = 0
        self.reward_counts = np.zeros(self.n_actions)

    def select_action(self) -> int:
        """Select action for new data.

        Returns
        -------
        result: int
            The selected action.

        """
        theta = np.random.beta(a=self.reward_counts + self.alpha,
                               b=(self.action_counts - self.reward_counts) + self.beta)
        result = np.argmax(theta)
        return result

    def update_params(self, action: int, reward: Union[int, float]) -> None:
        """Update parameters.

        Parameters
        ----------
        action: int
            The selected action.

        reward: int, float
            The observed reward value.

        """
        self.num_iter += 1
        if self.num_iter % self.observation_interval == 0:
            self.action_counts[action] += 1
            self.reward_counts[action] += reward


class GaussianTS(PolicyInterface):
    """Thompson Sampling for Gaussian Distribution.

    Parameters
    ----------
    n_actions: int
        The number of given bandit actions.

    mu_prior: float (default=1.0)
        The hyperparameter mu for prior gaussian distribution.

    lam_likelihood: float (default=1.0)
        The hyperparameter lamda for likelihood gaussian distribution.

    lam_prior: float (defaut=1.0)
        The hyperparameter lamda for prior gaussian distribution.

    observation_interval: int, optional (default=1)
        The number of data given in each batch.

    """

    _policy_type = "stochastic"
    name = "GaussianThompsonSampling"

    def __init__(self, n_actions: int, mu_prior: float=0.0, lam_likelihood: float=1.0, lam_prior: float=1.0, observation_interval: int=1) -> None:
        """Initialize class."""
        self.n_actions = n_actions
        self.action_counts = np.zeros(self.n_actions, dtype=int)
        self.observation_interval = observation_interval
        self.num_iter = 0
        self.reward_sums = np.zeros(self.n_actions, dtype=float)
        self.mu = np.zeros(self.n_actions, dtype=float)
        self.lam = np.ones(self.n_actions, dtype=float) * lam_prior
        self.mu_prior = mu_prior
        self.lam_prior = lam_prior
        self.lam_likelihood = lam_likelihood

    def select_action(self) -> int:
        """Select action for new data.

        Returns
        -------
        result: int
            The selected action.

        """
        theta = np.random.normal(loc=self.mu, scale=(1.0 / self.lam))
        result = np.argmax(theta)
        return result

    def update_params(self, action: int, reward: Union[int, float]) -> None:
        """Update parameters.

        Parameters
        ----------
        action: int
            The selected action.

        reward: int, float
            The observed reward value.

        """
        self.num_iter += 1
        self.reward_sums[action] += reward

        if self.num_iter % self.observation_interval == 0:
            self.lam = self.action_counts * self.lam_likelihood + self.lam_prior
            self.mu = (self.lam_likelihood * self.reward_sums + self.lam_prior * self.mu_prior) / self.lam
