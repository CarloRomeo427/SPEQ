import copy
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from redq.algos.core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, ReplayBuffer, \
    mbpo_target_entropy_dict, soft_update_policy
import wandb

from redq.algos.core import test_agent


def get_probabilistic_num_min(num_mins):
    floored_num_mins = np.floor(num_mins)
    if num_mins - floored_num_mins > 0.001:
        prob_for_higher_value = num_mins - floored_num_mins
        if np.random.uniform(0, 1) < prob_for_higher_value:
            return int(floored_num_mins + 1)
        else:
            return int(floored_num_mins)
    else:
        return num_mins


class REDQSACAgent(object):
    def __init__(self, env_name, obs_dim, act_dim, act_limit, device,
                 hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
                 lr=3e-4, gamma=0.99, polyak=0.995,
                 alpha=0.2, auto_alpha=True, target_entropy='mbpo',
                 start_steps=5000, delay_update_steps='auto',
                 utd_ratio=20, num_Q=10, num_min=2, q_target_mode='min',
                 policy_update_delay=20,
                 target_drop_rate=0.0, layer_norm=False,
                 utd_ratio_offline=None):
        self.policy_net = TanhGaussianPolicy(obs_dim, act_dim, (256, 256), action_limit=act_limit).to(device)
        self.q_net_list, self.q_target_net_list = [], []
        for q_i in range(num_Q):
            new_q_net = Mlp(obs_dim + act_dim, 1, hidden_sizes, target_drop_rate=target_drop_rate,
                            layer_norm=layer_norm).to(device)
            self.q_net_list.append(new_q_net)
            new_q_target_net = Mlp(obs_dim + act_dim, 1, hidden_sizes, target_drop_rate=target_drop_rate,
                                   layer_norm=layer_norm).to(device)
            new_q_target_net.load_state_dict(new_q_net.state_dict())
            self.q_target_net_list.append(new_q_target_net)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optimizer_list = [optim.Adam(q.parameters(), lr=lr) for q in self.q_net_list]
        self.auto_alpha = auto_alpha
        if auto_alpha:
            if target_entropy == 'auto':
                self.target_entropy = - act_dim
            if target_entropy == 'mbpo':
                mbpo_target_entropy_dict['AntTruncatedObs-v2'] = -4
                mbpo_target_entropy_dict['HumanoidTruncatedObs-v2'] = -2
                try:
                    self.target_entropy = mbpo_target_entropy_dict[env_name]
                except:
                    self.target_entropy = -2
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.cpu().exp().item()
        else:
            self.alpha = alpha
            self.target_entropy, self.log_alpha, self.alpha_optim = None, None, None
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.val_replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        self.mse_criterion = nn.MSELoss()
        self.start_steps = start_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.lr = lr
        self.hidden_sizes = hidden_sizes
        self.gamma = gamma
        self.polyak = polyak
        self.replay_size = replay_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_min = num_min
        self.num_Q = num_Q
        self.utd_ratio = utd_ratio
        self.delay_update_steps = self.start_steps if delay_update_steps == 'auto' else delay_update_steps
        self.q_target_mode = q_target_mode
        self.policy_update_delay = policy_update_delay
        self.device = device
        self.utd_ratio_offline = utd_ratio_offline if utd_ratio_offline else utd_ratio
        self.target_drop_rate = target_drop_rate
        self.layer_norm = layer_norm

    def __get_current_num_data(self):
        return self.replay_buffer.size

    def get_exploration_action(self, obs, env):
        with torch.no_grad():
            if self.__get_current_num_data() > self.start_steps:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
                action_tensor = self.policy_net.forward(obs_tensor, deterministic=False, return_log_prob=False)[0]
                action = action_tensor.cpu().numpy().reshape(-1)
            else:
                action = env.action_space.sample()
        return action

    def get_test_action(self, obs):
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy_net.forward(obs_tensor, deterministic=True, return_log_prob=False)[0]
            action = action_tensor.cpu().numpy().reshape(-1)
        return action

    def get_action_and_logprob_for_bias_evaluation(self, obs):
        if obs is None or len(obs) == 0:
            raise ValueError("Observation is empty")
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
            action_tensor, _, _, log_prob_a_tilda, _, _, = self.policy_net.forward(obs_tensor, deterministic=False,
                                                                                   return_log_prob=True)
            action = action_tensor.cpu().numpy().reshape(-1)
        return action, log_prob_a_tilda

    def get_ave_q_prediction_for_bias_evaluation(self, obs_tensor, acts_tensor):
        q_prediction_list = []
        for q_i in range(self.num_Q):
            q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
            q_prediction_list.append(q_prediction)
        q_prediction_cat = torch.cat(q_prediction_list, dim=1)
        average_q_prediction = torch.mean(q_prediction_cat, dim=1)
        return average_q_prediction

    def store_data(self, o, a, r, o2, d):
        self.replay_buffer.store(o, a, r, o2, d)

    def store_val_data(self, o, a, r, o2, d):
        self.val_replay_buffer.store(o, a, r, o2, d)

    def sample_data(self, batch_size):
        batch = self.replay_buffer.sample_batch(batch_size)
        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def get_td_error(self):
        with torch.no_grad():
            batch = self.replay_buffer.sample_all()
            result = np.zeros([1, len(batch['obs1'])])
            for i in range(0, len(batch['obs1']), 1000):
                obs_tensor = Tensor(batch['obs1'][i:i + 1000]).to(self.device)
                obs_next_tensor = Tensor(batch['obs2'][i:i + 1000]).to(self.device)
                acts_tensor = Tensor(batch['acts'][i:i + 1000]).to(self.device)
                rews_tensor = Tensor(batch['rews'][i:i + 1000]).unsqueeze(1).to(self.device)
                done_tensor = Tensor(batch['done'][i:i + 1000]).unsqueeze(1).to(self.device)

                y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
                q_prediction_list = []
                for q_i in range(self.num_Q):
                    q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                    q_prediction_list.append(q_prediction)
                q_prediction_cat = torch.cat(q_prediction_list, dim=1)
                y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
                q_loss_all = ((q_prediction_cat - y_q) ** 2).mean(1)
                result[0, i:i + 1000] = q_loss_all.cpu().numpy()

        return result

    def get_redq_q_target_no_grad(self, obs_next_tensor, rews_tensor, done_tensor):
        num_mins_to_use = get_probabilistic_num_min(self.num_min)
        sample_idxs = np.random.choice(self.num_Q, num_mins_to_use, replace=False)
        with torch.no_grad():
            if self.q_target_mode == 'min':
                a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor)
                q_prediction_next_list = []
                for sample_idx in sample_idxs:
                    q_prediction_next = self.q_target_net_list[sample_idx](
                        torch.cat([obs_next_tensor, a_tilda_next], 1))
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
                min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
                next_q_with_log_prob = min_q - self.alpha * log_prob_a_tilda_next
                y_q = rews_tensor + self.gamma * (1 - done_tensor) * next_q_with_log_prob
            if self.q_target_mode == 'ave':
                a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor)
                q_prediction_next_list = []
                for q_i in range(self.num_Q):
                    q_prediction_next = self.q_target_net_list[q_i](torch.cat([obs_next_tensor, a_tilda_next], 1))
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_ave = torch.cat(q_prediction_next_list, 1).mean(dim=1).reshape(-1, 1)
                next_q_with_log_prob = q_prediction_next_ave - self.alpha * log_prob_a_tilda_next
                y_q = rews_tensor + self.gamma * (1 - done_tensor) * next_q_with_log_prob
            if self.q_target_mode == 'rem':
                a_tilda_next, _, _, log_prob_a_tilda_next, _, _ = self.policy_net.forward(obs_next_tensor)
                q_prediction_next_list = []
                for q_i in range(self.num_Q):
                    q_prediction_next = self.q_target_net_list[q_i](torch.cat([obs_next_tensor, a_tilda_next], 1))
                    q_prediction_next_list.append(q_prediction_next)
                q_prediction_next_cat = torch.cat(q_prediction_next_list, 1)
                rem_weight = Tensor(np.random.uniform(0, 1, q_prediction_next_cat.shape)).to(device=self.device)
                normalize_sum = rem_weight.sum(1).reshape(-1, 1).expand(-1, self.num_Q)
                rem_weight = rem_weight / normalize_sum
                q_prediction_next_rem = (q_prediction_next_cat * rem_weight).sum(dim=1).reshape(-1, 1)
                next_q_with_log_prob = q_prediction_next_rem - self.alpha * log_prob_a_tilda_next
                y_q = rews_tensor + self.gamma * (1 - done_tensor) * next_q_with_log_prob
        return y_q, sample_idxs

    def train(self, logger):
        num_update = 0 if self.__get_current_num_data() <= self.delay_update_steps else self.utd_ratio
        for i_update in range(num_update):
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)

            """Q loss"""
            y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction_list = []
            for q_i in range(self.num_Q):
                q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                q_prediction_list.append(q_prediction)
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()

            """policy and alpha loss"""
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = self.policy_net.forward(
                    obs_tensor)
                q_a_tilda_list = []
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(False)
                    q_a_tilda = self.q_net_list[sample_idx](torch.cat([obs_tensor, a_tilda], 1))
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(True)

                if self.auto_alpha:
                    alpha_loss = -(self.log_alpha * (log_prob_a_tilda + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.cpu().exp().item()
                else:
                    alpha_loss = Tensor([0])

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                self.policy_optimizer.step()

            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)

            if i_update == num_update - 1:
                logger.store(LossPi=policy_loss.cpu().item(), LossQ1=q_loss_all.cpu().item() / self.num_Q,
                             LossAlpha=alpha_loss.cpu().item(),
                             Q1Vals=q_prediction.detach().cpu().numpy(),  # NOTE: use parentheses for .numpy()
                             Alpha=self.alpha, LogPi=log_prob_a_tilda.detach().cpu().numpy(),
                             PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1))

                wandb.log({"policy_loss": policy_loss.cpu().item(),
                           "mean_loss_q": q_loss_all.cpu().item() / self.num_Q})

        if num_update == 0:
            logger.store(LossPi=0, LossQ1=0, LossAlpha=0, Q1Vals=0, Alpha=0, LogPi=0, PreTanh=0)

    def finetune_offline(self, epochs, test_env=None):
        num_update = 0 if self.__get_current_num_data() <= self.delay_update_steps else self.utd_ratio_offline
        initial_val_loss = self.evaluate_validation_loss(batch_size=self.batch_size)
        wandb.log({'ValQloss': initial_val_loss})
        steps_since_last_check = 0
        for e in range(epochs):
            
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)

            """Q loss"""
            y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction_list = []
            for q_i in range(self.num_Q):
                q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                q_prediction_list.append(q_prediction)
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q

            # Revert to MSE loss
            q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)

            steps_since_last_check += 1

            # Check validation loss after a certain interval
            if steps_since_last_check >= 1000:
                current_val_loss = self.evaluate_validation_loss(batch_size=self.batch_size)
                wandb.log({'ValQloss': current_val_loss})
                steps_since_last_check = 0


                

    def evaluate_validation_loss(self, batch_size=256, heldout_ratio=0.0):
        """
        Evaluate Q-loss on the entire validation replay buffer by iterating over it in batches.
        Returns the average Q-loss across all validation data.
        """
        if heldout_ratio > 0.0:
            val_data = self.val_replay_buffer.sample_all()
        else:
            val_data = self.replay_buffer.sample_all()
        obs_full = val_data['obs1']
        obs_next_full = val_data['obs2']
        acts_full = val_data['acts']
        rews_full = val_data['rews']
        done_full = val_data['done']

        num_samples = len(obs_full)
        if num_samples == 0:
            return 0.0  # No validation data available

        num_batches = (num_samples + batch_size - 1) // batch_size
        total_loss = 0.0

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            obs_tensor = torch.Tensor(obs_full[start_idx:end_idx]).to(self.device)
            obs_next_tensor = torch.Tensor(obs_next_full[start_idx:end_idx]).to(self.device)
            acts_tensor = torch.Tensor(acts_full[start_idx:end_idx]).to(self.device)
            rews_tensor = torch.Tensor(rews_full[start_idx:end_idx]).unsqueeze(1).to(self.device)
            done_tensor = torch.Tensor(done_full[start_idx:end_idx]).unsqueeze(1).to(self.device)

            # Compute Q-loss for this batch
            with torch.no_grad():
                y_q, _ = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
                q_prediction_list = [
                    q_net(torch.cat([obs_tensor, acts_tensor], 1)) for q_net in self.q_net_list
                ]
                q_prediction_cat = torch.cat(q_prediction_list, dim=1)
                y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q

                # MSE loss for validation data
                q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q
                total_loss += q_loss_all.item()

        average_loss = total_loss / num_batches
        return average_loss

    def early_finetuning(self, epochs, threshold, improvement=0.05, patience=5, check_interval=1000, heldout_ratio=0.0):
        """
        Offline fine-tuning with a two-phase early stopping:
        
        Phase 1: Continue training until current_val_loss < initial_val_loss * x (lower_bound).
        Phase 2: Once lower_bound is reached at least once, 
                require 5% improvement over the previous best to continue.
                If no 5% improvement for `patience` checks, stop early.
        """

        initial_val_loss = self.evaluate_validation_loss(batch_size=self.batch_size, heldout_ratio=heldout_ratio)
        lower_bound = initial_val_loss * threshold
        if heldout_ratio > 0.0:
            wandb.log({'ValDSSize': self.val_replay_buffer.size})
        else:
            wandb.log({'ValDSSize': self.replay_buffer.size})

        

        print(f"Initial validation Q-loss: {initial_val_loss:.4f}, Lower bound: {lower_bound:.4f}")
        wandb.log({'ValQloss': initial_val_loss})

        # Phase 1 tracking
        threshold_reached = False
        # best_val_loss = None  # Will be set once threshold is reached
        best_val_loss = initial_val_loss

        # Phase 2 tracking (once threshold is reached)
        no_improvement_count = 0

        steps_since_last_check = 0
        last_epoch = 0

        for e in range(epochs):
            last_epoch = e

            # Sample a batch from main replay buffer
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)

            # Compute Q-loss on this batch with MSE
            y_q, _ = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction_list = [
                q_net(torch.cat([obs_tensor, acts_tensor], 1))
                for q_net in self.q_net_list
            ]
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

            # Update Q-networks
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            # Soft update Q-target networks
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)

            steps_since_last_check += 1

            # Check validation loss after a certain interval
            if steps_since_last_check >= check_interval:
                current_val_loss = self.evaluate_validation_loss(batch_size=self.batch_size)
                wandb.log({'ValQloss': current_val_loss})
                steps_since_last_check = 0

                # if not threshold_reached:
                #     # Phase 1: Check if we've reached below lower_bound
                #     if current_val_loss < lower_bound:
                #         # Threshold reached, enter Phase 2
                #         threshold_reached = True
                #         best_val_loss = current_val_loss
                #         no_improvement_count = 0
                #         print(f"Reached lower bound at epoch {e}: current_val_loss {current_val_loss:.4f} < {lower_bound:.4f}")
                # else:
                #     # Phase 2: Check for significant improvements over best_val_loss
                #     # Must improve by at least 5%
                #     if current_val_loss < best_val_loss * (1 - improvement):
                #         # Significant improvement detected
                #         best_val_loss = current_val_loss
                #         no_improvement_count = 0
                #         print(f"Significant improvement at epoch {e}: new best_val_loss {best_val_loss:.4f}")
                #     else:
                #         # No significant improvement
                #         no_improvement_count += 1
                #         print(f"No significant improvement at epoch {e}: {no_improvement_count}/{patience}")

                #         # If we have gone 'patience' checks with no improvement, stop early
                #         if no_improvement_count >= patience:
                #             wandb.log({'OffEpochs': e})
                #             print(f"Stopping early due to no 5% improvement for {no_improvement_count} checks.")
                #             return


                if current_val_loss < best_val_loss*(1-improvement):
                    
                    best_val_loss = current_val_loss
                    no_improvement_count = 0
                    print(f"Significant improvement at epoch {e}: new best_val_loss {best_val_loss:.4f}")
                else:
                    # No significant improvement
                    no_improvement_count += 1
                    print(f"No significant improvement at epoch {e}: {no_improvement_count}/{patience}")

                    # If we have gone 'patience' checks with no improvement, stop early
                    if no_improvement_count >= patience:
                        wandb.log({'OffEpochs': e})
                        print(f"Stopping early due to no 5% improvement for {no_improvement_count} checks.")
                        return
        wandb.log({'OffEpochs': last_epoch})


    def early_finetuning_with_bias(self, epochs, patience=5, check_interval=1000, 
                                    init_bias=None, bias_state=None, bias_action=None, mc_values=None):
        best_val = np.mean(init_bias)
        steps_since_last_check = 0
        no_improvement_count = 0
        for e in range(epochs):
            last_epoch = e

            # Sample a batch from main replay buffer
            obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)

            # Compute Q-loss on this batch with MSE
            y_q, _ = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction_list = [
                q_net(torch.cat([obs_tensor, acts_tensor], 1))
                for q_net in self.q_net_list
            ]
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

            # Update Q-networks
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            # Soft update Q-target networks
            for q_i in range(self.num_Q):
                soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)

            steps_since_last_check += 1

            # Check validation loss after a certain interval
            if steps_since_last_check >= check_interval:
                with torch.no_grad():
                    current_qs = self.get_ave_q_prediction_for_bias_evaluation(bias_state, bias_action).cpu().numpy().reshape(-1)
                current_bias = current_qs - mc_values
                # print(f"Current bias: {current_bias}")
                wandb.log({'ValQloss': current_bias})
                steps_since_last_check = 0
                current_bias = np.mean(current_bias)
                goal_bias = best_val * 0.99
                # print(f"Current bias: {current_bias:.4f}")
                # print(f"Best bias: {best_val:.4f}")
                # print(f"Best bias * 0.99: {goal_bias:.4f}")
                if current_bias < goal_bias:
                    # print(f"IS current bias < best_val*0.99: {current_bias < goal_bias}")
                    # input()
                    best_val = current_bias
                    no_improvement_count = 0
                    print(f"Significant improvement at epoch {e}: new best_val_loss {best_val:.4f}")
                else:
                    # No significant improvement
                    no_improvement_count += 1
                    print(f"No significant improvement at epoch {e}: {no_improvement_count}/{patience}")

                    # If we have gone 'patience' checks with no improvement, stop early
                    if no_improvement_count >= patience:
                        # wandb.log({'OffEpochs': e})
                        print(f"Stopping early due to no 5% improvement for {no_improvement_count} checks.")
                        break
        wandb.log({'OffEpochs': last_epoch})
