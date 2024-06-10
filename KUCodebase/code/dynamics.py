import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as path
from typing import Callable, List, Tuple, Dict, Union, Tuple, Optional
import argparse
import gym
import d4rl
import sys
import os
import random
import time
from tqdm import tqdm

from dynamics_utils import soft_clamp, Swish, StandardScaler


class BaseDynamics(object):

    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer
    ) -> None:
        super().__init__()
        self.model = model
        self.optim = optim
    
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        raise NotImplementedError
    
class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric"
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))

        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        
        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info
    
    @ torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act)
        mean, logvar = self.model(obs_act)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1]
        return next_obss

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        return inputs, targets

    ### ORIGINAL TRAIN FUNCTION
    # def train(
    #     self,
    #     data: Dict,
    #     # logger: Logger,
    #     max_epochs: Optional[float] = None,
    #     max_epochs_since_update: int = 5,
    #     batch_size: int = 256,
    #     holdout_ratio: float = 0.2,
    #     logvar_loss_coef: float = 0.01
    # ) -> None:
    #     inputs, targets = self.format_samples_for_training(data)
    #     data_size = inputs.shape[0]
    #     holdout_size = min(int(data_size * holdout_ratio), 1000)
    #     train_size = data_size - holdout_size
    #     train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
    #     train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
    #     holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]

    #     self.scaler.fit(train_inputs)
    #     train_inputs = self.scaler.transform(train_inputs)
    #     holdout_inputs = self.scaler.transform(holdout_inputs)
    #     holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

    #     data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
    #     def shuffle_rows(arr):
    #         idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    #         return arr[np.arange(arr.shape[0])[:, None], idxes]

    #     epoch = 0
    #     cnt = 0
    #     # logger.log("Training dynamics:")
    #     while True:
    #         epoch += 1
    #         start_time = time.time()
    #         train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size, logvar_loss_coef)
    #         new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
    #         holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
    #         # logger.logkv("loss/dynamics_train_loss", train_loss)
    #         # logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
    #         # logger.set_timestep(epoch)
    #         # logger.dumpkvs(exclude=["policy_training_progress"])

            # print(f"Time for epoch: {epoch} = {time.time() - start_time}, train_loss: {train_loss}, holdout_loss: {holdout_loss}")

    #         # shuffle data for each base learner
    #         data_idxes = shuffle_rows(data_idxes)

    #         indexes = []
    #         for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
    #             improvement = (old_loss - new_loss) / old_loss
    #             if improvement > 0.01:
    #                 indexes.append(i)
    #                 holdout_losses[i] = new_loss
            
    #         if len(indexes) > 0:
    #             self.model.update_save(indexes)
    #             cnt = 0
    #         else:
    #             cnt += 1
            
    #         if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
    #             break

    #     indexes = self.select_elites(holdout_losses)
    #     self.model.set_elites(indexes)
    #     self.model.load_save()
    #     # self.save(logger.model_dir)
    #     self.model.eval()
    #     # logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))
    
    def train(
        self,
        data: Dict,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:
        inputs, targets = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        progress_bar = tqdm(total=max_epochs if max_epochs else float('inf'), desc="Training Dynamics", unit="epoch")
        
        while True:
            epoch += 1
            start_time = time.time()
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size, logvar_loss_coef)
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            
            progress_bar.set_postfix({
                "epoch": epoch,
                "train_loss": train_loss,
                "holdout_loss": holdout_loss,
                "time_per_epoch": time.time() - start_time
            })

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            progress_bar.update(1)
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        progress_bar.close()
        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.model.eval()


    def train_on_memory(
        self,
        data,
        max_epochs: Optional[float] = 1,
        max_epochs_since_update: int = 0,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:
        inputs = np.concatenate((data[0].cpu(), data[1].cpu()), axis=-1)
        targets = np.concatenate((data[3].cpu(), data[2].cpu()), axis=-1)

        # inputs = torch.from_numpy(inputs).float().to(self.model.device)
        # targets = torch.from_numpy(targets).float().to(self.model.device)

        # print(f"inputs.shape: {inputs.shape}, targets.shape: {targets.shape}")
        # input()

        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]
        # print(f"train_inputs.shape: {train_inputs.shape}, train_targets.shape: {train_targets.shape}")
        # input()

        self.scaler.partial_fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        # print(f"Train after transform: {train_inputs.shape}")
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        # print(f"data_idxes.shape: {data_idxes.shape}| {train_inputs[data_idxes].shape}")
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        # progress_bar = tqdm(total=max_epochs if max_epochs else float('inf'), desc="Training Dynamics", unit="epoch")
        
        while True:
            epoch += 1
            start_time = time.time()
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size, logvar_loss_coef)
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            # print(f"Time for epoch: {epoch} = {time.time() - start_time}, train_loss: {train_loss}, holdout_loss: {holdout_loss}", flush=True)
            # progress_bar.set_postfix({
            #     "epoch": epoch,
            #     "train_loss": train_loss,
            #     "holdout_loss": holdout_loss,
            #     "time_per_epoch": time.time() - start_time
            # })

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            # progress_bar.update(1)
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        # progress_bar.close()
        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.model.eval()




    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)

    def learn_single(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        
        inputs = torch.as_tensor(inputs, dtype=torch.float32).to(self.model.device)
        targets = torch.as_tensor(targets, dtype=torch.float32).to(self.model.device)
        
        mean, logvar = self.model(inputs)
        inv_var = torch.exp(-logvar)
        # Average over batch and dim, sum over ensembles.
        mse_loss_inv = (torch.pow(mean - targets, 2) * inv_var).mean(dim=(1, 2))
        var_loss = logvar.mean(dim=(1, 2))
        loss = mse_loss_inv.sum() + var_loss.sum()
        loss = loss + self.model.get_decay_loss()
        loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()
    
    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)

class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias

        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss

class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: nn.Module = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_reward: bool = True,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        # self.device = torch.device(device)
        self.device = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")

        self.activation = activation()

        assert len(weight_decays) == (len(hidden_dims) + 1)

        module_list = []
        hidden_dims = [obs_dim+action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weight_decay in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
        self.backbones = nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * (obs_dim + self._with_reward),
            num_ensemble,
            weight_decays[-1]
        )

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * -10, requires_grad=True)
        )

        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )

        self.to(self.device)

    def forward(self, obs_action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self.device)
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs

