import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution
import numpy as np
import d4rl
import random
import gym
import matplotlib.pyplot as plt
import matplotlib
from typing import Callable, List, Tuple, Dict, Union, Tuple, Optional
import os.path as path

matplotlib.use('WebAgg')

CONST_EPS = 1e-10


def plot_distributions(original, predicted, feature_name):
    plt.figure(figsize=(10, 6))
    plt.hist(original.flatten(), bins=50, alpha=0.5, label='Original')
    plt.hist(predicted.flatten(), bins=50, alpha=0.5, label='Predicted')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Original vs Predicted {feature_name}')
    plt.legend()
    plt.show()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def orthogonal_initWeights(
    net: nn.Module,
    ) -> None:
    for e in net.parameters():
        if len(e.size()) >= 2:
            nn.init.orthogonal_(e)


def log_prob_func(
    dist: Distribution, action: torch.Tensor
    ) -> torch.Tensor:
    log_prob = dist.log_prob(action)
    if len(log_prob.shape) == 1:
        return log_prob
    else:
        return log_prob.sum(-1, keepdim=True)
    

def load_demonstrations_d4rl(env, normalize=False, normalize_reward=False):
    dataset = d4rl.qlearning_dataset(env)
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    states_mean = None
    states_std = None
    if normalize:
        states_mean = np.mean(dataset['observations'], axis=0)
        states_std = np.std(dataset['observations'], axis=0) + 1e-3

    new_dataset = dict(states=[], actions=[], rewards=[], next_states=[], timesteps=[], dones=[])

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    trajectories = []
    timestep = 0
    t = dict(states=[], actions=[], rewards=[], next_states=[], timesteps=[], dones=[])

    for state, action, reward, next_state, done in zip(dataset['observations'],
                                                          dataset['actions'], dataset['rewards'],
                                                          dataset['next_observations'], dataset['terminals']):

        if states_mean is not None:
            state = (state - states_mean) / states_std
            next_state = (next_state - states_mean) / states_std

        t['states'].append(state)
        t['next_states'].append(next_state)
        t['rewards'].append(reward)
        t['actions'].append(action)
        t['timesteps'].append(timestep)

        timestep += 1
        if use_timeouts:
            final_timestep = dataset['timeouts'][timestep]
        else:
            final_timestep = (timestep == 1000 - 1)

        if done or final_timestep:
            t['actions'] = np.asarray(t['actions'])
            t['rewards'] = np.asarray(t['rewards'])
            t['states'] = np.asarray(t['states'])
            t['next_states'] = np.asarray(t['next_states'])
            t['timesteps'] = np.asarray(t['timesteps'])
            t['dones'].append(True)
            trajectories.append(t)
            t = dict(states=[], actions=[], rewards=[], dones=[], timesteps=[], next_states=[])
            timestep = 0
        else:
            t['dones'].append(False)
    
    if len(t['states']) > 0:
        t['actions'] = np.asarray(t['actions'])
        t['rewards'] = np.asarray(t['rewards'])
        t['states'] = np.asarray(t['states'])
        t['next_states'] = np.asarray(t['next_states'])
        t['timesteps'] = np.asarray(t['timesteps'])
        t['dones'].append(True)
        trajectories.append(t)

    print("There are {} transitions in total in this dataset".format(len(dataset['observations'])))
    print("There are {} trajectories in total in this dataset".format(len(trajectories)))

    random.shuffle(trajectories)

    # Populate the dataset of transitions

    new_dataset['states'] = np.asarray([s for t in trajectories for s in t['states']])
    new_dataset['states'] = np.reshape(new_dataset['states'], (-1, state_size))
    new_dataset['actions'] = np.asarray([a for t in trajectories for a in t['actions']])
    new_dataset['actions'] = np.reshape(new_dataset['actions'], (-1, action_size))
    new_dataset['rewards'] = np.asarray([r for t in trajectories for r in t['rewards']])
    new_dataset['rewards'] = np.reshape(new_dataset['rewards'], (-1, 1))
    new_dataset['next_states'] = np.asarray([ns for t in trajectories for ns in t['next_states']])
    new_dataset['next_states'] = np.reshape(new_dataset['next_states'], (-1, state_size))
    new_dataset['timesteps'] = np.asarray([ts for t in trajectories for ts in t['timesteps']])
    new_dataset['timesteps'] = np.reshape(new_dataset['timesteps'], (-1, 1))
    new_dataset['dones'] = np.asarray([d for t in trajectories for d in t['dones']])
    new_dataset['dones'] = np.reshape(new_dataset['dones'], (-1, 1))

    if normalize_reward:
        new_dataset['rewards'] = (new_dataset['rewards'] - np.min(new_dataset['rewards'])) / (np.max(new_dataset['rewards']) - np.min(new_dataset['rewards']))

    print("We will use {} transitions in this dataset".format(len(new_dataset['states'])))
    print("We will use {} trajectories in this dataset".format(len(trajectories)))

    print("Max reward: {}".format(np.max(new_dataset['rewards'])))
    print("Min reward: {}".format(np.min(new_dataset['rewards'])))
    print("Mean reward: {}".format(np.mean(new_dataset['rewards'])))
    # input("Press any key to continue... ")
    return new_dataset, trajectories, states_mean, states_std


# class StandardScaler(object):
#     def __init__(self, mu=None, std=None):
#         self.mu = mu
#         self.std = std

#     def fit(self, data):
#         """Runs two ops, one for assigning the mean of the data to the internal mean, and
#         another for assigning the standard deviation of the data to the internal standard deviation.
#         This function must be called within a 'with <session>.as_default()' block.

#         Arguments:
#         data (np.ndarray): A numpy array containing the input

#         Returns: None.
#         """
#         self.mu = np.mean(data, axis=0, keepdims=True)
#         self.std = np.std(data, axis=0, keepdims=True)
#         self.std[self.std < 1e-12] = 1.0

#     def transform(self, data):
#         """Transforms the input matrix data using the parameters of this scaler.

#         Arguments:
#         data (np.array): A numpy array containing the points to be transformed.

#         Returns: (np.array) The transformed dataset.
#         """
#         return (data - self.mu) / self.std

#     def inverse_transform(self, data):
#         """Undoes the transformation performed by this scaler.

#         Arguments:
#         data (np.array): A numpy array containing the points to be transformed.

#         Returns: (np.array) The transformed dataset.
#         """
#         return self.std * data + self.mu
    
#     def save_scaler(self, save_path):
#         mu_path = path.join(save_path, "mu.npy")
#         std_path = path.join(save_path, "std.npy")
#         np.save(mu_path, self.mu)
#         np.save(std_path, self.std)
    
#     def load_scaler(self, load_path):
#         mu_path = path.join(load_path, "mu.npy")
#         std_path = path.join(load_path, "std.npy")
#         self.mu = np.load(mu_path)
#         self.std = np.load(std_path)

#     def transform_tensor(self, data: torch.Tensor):
#         device = data.device
#         data = self.transform(data.cpu().numpy())
#         data = torch.tensor(data, device=device)
#         return data



class StandardScaler(object):
    def __init__(self, mu=None, std=None):
        self.mu = mu
        self.std = std
        self.n = 0  # To keep track of the number of samples

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray or torch.Tensor): A numpy array or tensor containing the input

        Returns: None.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0
        self.n = data.shape[0]

    def partial_fit(self, data):
        """Updates the mean and standard deviation with new data.

        Arguments:
        data (np.ndarray or torch.Tensor): A numpy array or tensor containing the new input

        Returns: None.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        if self.mu is None or self.std is None:
            # If the scaler is not initialized, use fit
            self.fit(data)
        else:
            # Update mean and std incrementally
            new_n = self.n + data.shape[0]
            new_mu = (self.mu * self.n + np.sum(data, axis=0, keepdims=True)) / new_n
            new_ssd = self.std**2 * (self.n - 1) + np.sum((data - new_mu)**2, axis=0, keepdims=True)
            new_std = np.sqrt(new_ssd / new_n)
            new_std[new_std < 1e-12] = 1.0

            self.mu = new_mu
            self.std = new_std
            self.n = new_n

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.ndarray or torch.Tensor): A numpy array or tensor containing the points to be transformed.

        Returns: (np.ndarray or torch.Tensor) The transformed dataset.
        """
        if isinstance(data, torch.Tensor):
            device = data.device
            data = data.cpu().numpy()
            data = (data - self.mu) / self.std
            return torch.tensor(data, device=device)
        else:
            return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.ndarray or torch.Tensor): A numpy array or tensor containing the points to be transformed.

        Returns: (np.ndarray or torch.Tensor) The transformed dataset.
        """
        if isinstance(data, torch.Tensor):
            device = data.device
            data = data.cpu().numpy()
            data = self.std * data + self.mu
            return torch.tensor(data, device=device)
        else:
            return self.std * data + self.mu
    
    def save_scaler(self, save_path):
        mu_path = path.join(save_path, "mu.npy")
        std_path = path.join(save_path, "std.npy")
        np.save(mu_path, self.mu)
        np.save(std_path, self.std)
    
    def load_scaler(self, load_path):
        mu_path = path.join(load_path, "mu.npy")
        std_path = path.join(load_path, "std.npy")
        self.mu = np.load(mu_path)
        self.std = np.load(std_path)
        self.n = len(self.mu)  # Assuming mu and std are updated with the number of samples

    def transform_tensor(self, data: torch.Tensor):
        return self.transform(data)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }
    
    def __len__(self) -> int:
        return self._size
