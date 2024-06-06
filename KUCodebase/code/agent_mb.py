import os
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from rltorch.memory import PrioritizedMemory
from memory import MultiStepMemory
import time
import copy

from model import TwinnedQNetwork, GaussianPolicy, RandomizedEnsembleNetwork
from utils import grad_false, hard_update, soft_update, to_batch, update_params, RunningMeanStats

from collections import deque
import itertools
import math

class SacAgent:

    def __init__(self, env, log_dir, dynamics, num_steps=3000000, batch_size=256,
                 lr=0.0003, hidden_units=[256, 256], memory_size=1e6,
                 gamma=0.99, tau=0.005, entropy_tuning=True, ent_coef=0.2,
                 multi_step=1, per=False, alpha=0.6, beta=0.4,
                 beta_annealing=0.0001, grad_clip=None, updates_per_step=1,
                 start_steps=10000, log_interval=10, target_update_interval=1,
                 eval_interval=1000, cuda=0, seed=0,
                 eval_runs=1, huber=0, layer_norm=0,
                 method=None, target_entropy=None, target_drop_rate=0.0, critic_update_delay=1):
        self.env = env

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        torch.backends.cudnn.deterministic = True  # It harms a performance.
        torch.backends.cudnn.benchmark = False

        self.method = method
        self.critic_update_delay = critic_update_delay
        self.target_drop_rate = target_drop_rate

        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
        self.dynamics = dynamics

        # policy
        self.policy = GaussianPolicy(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)

        # Q functions
        kwargs_q = {"num_inputs": self.env.observation_space.shape[0],
                    "num_actions": self.env.action_space.shape[0],
                    "hidden_units": hidden_units,
                    "layer_norm": layer_norm,
                    "drop_rate": self.target_drop_rate}
        if self.method == "redq":
            self.critic = RandomizedEnsembleNetwork(**kwargs_q).to(self.device)
            self.critic_target = RandomizedEnsembleNetwork(**kwargs_q).to(self.device)
        else:
            self.critic = TwinnedQNetwork(**kwargs_q).to(self.device)
            self.critic_target = TwinnedQNetwork(**kwargs_q).to(self.device)
        if self.target_drop_rate <= 0.0:
            self.critic_target = self.critic_target.eval()
        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        # optimizer
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        if self.method == "redq":
            for i in range(self.critic.N):
                setattr(self, "q"+str(i)+"_optim",
                        Adam(getattr(self.critic, "Q"+str(i)).parameters(), lr=lr))
        else:
            self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
            self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)

        if entropy_tuning:
            if not (target_entropy is None):
                self.target_entropy = torch.prod(torch.Tensor([target_entropy]).to(self.device)).item()
            else:
                # Target entropy is -|A|.
                self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)

        if per:
            # replay memory with prioritied experience replay
            self.memory = PrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:
            # replay memory without prioritied experience replay
            self.memory = MultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step)

        # expanded memory for rollouts
        self.expanded_memory = MultiStepMemory(
            memory_size, self.env.observation_space.shape,
            self.env.action_space.shape, self.device, gamma, multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_rewards = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
        #
        self.eval_runs = eval_runs
        self.huber = huber
        self.multi_step = multi_step

    def run(self):
        while True:
            start_time = time.time()
            self.train_episode()
            print(f"Time for current episode: {np.round(time.time() - start_time)}", flush=True)
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and self.steps >= self.start_steps

    def act(self, state):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        # act without randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, action = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            if self.method == "sac" or self.method == "redq":
                next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies
            elif self.method == "duvn":
                next_q = next_q1 + self.alpha * next_entropies # discard q2
            elif self.method == "monosac":
                next_q2, _ = self.critic_target(next_states, next_actions)
                next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies
            else:
                raise NotImplementedError()
        target_q = (rewards / (self.multi_step * 1.0)) + (1.0 - dones) * self.gamma_n * next_q
        return target_q

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()

        while not done:
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
#---------#
            episode_reward += reward

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            if self.per:
                batch = to_batch(
                    state, action, reward, next_state, masked_done,
                    self.device)
                with torch.no_grad():
                    curr_q1, curr_q2 = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = (0.5 * torch.abs(curr_q1 - target_q) + 0.5 * torch.abs(curr_q2 - target_q)).item()
                self.memory.append(state, action, reward, next_state, masked_done, error, episode_done=done)
            else:
                self.memory.append(state, action, reward, next_state, masked_done, episode_done=done)

            if self.is_update():
                self.learn()

            if self.steps % self.eval_interval == 0:
                self.evaluate()
                self.save_models()
                self.dynamics.train_on_memory(self.memory.get_all(), max_epochs=10)
                self.copy_and_expand_memory()
                print(f"Memory size: {len(self.memory)} | Expanded memory size: {len(self.expanded_memory)}")

            state = next_state

        self.train_rewards.append(episode_reward)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar('reward/train', self.train_rewards.get(), self.steps)

        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'reward: {episode_reward:<5.1f}')

    def learn(self):
        self.learning_steps += 1

        if (self.learning_steps - 1) % self.critic_update_delay == 0:
            for _ in range(self.updates_per_step):
                if self.per:
                    batch, indices, weights = self.expanded_memory.sample(self.batch_size)
                else:
                    batch = self.expanded_memory.sample(self.batch_size)
                    weights = 1.

                if self.method == "redq":
                    losses, errors, mean_q1, mean_q2 = self.calc_critic_4redq_loss(batch, weights)
                    for i in range(self.critic.N):
                        update_params(getattr(self, "q" + str(i) + "_optim"),
                                      getattr(self.critic, "Q" + str(i)),
                                      losses[i], self.grad_clip)
                else:
                    q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(batch, weights)
                    update_params(self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
                    update_params(self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)

                if self.learning_steps % self.target_update_interval == 0:
                    soft_update(self.critic_target, self.critic, self.tau)

                if self.per:
                    self.memory.update_priority(indices, errors.cpu().numpy())

        if self.per:
            batch, indices, weights = self.expanded_memory.sample(self.batch_size)
        else:
            batch = self.expanded_memory.sample(self.batch_size)
            weights = 1.

        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        update_params(self.policy_optim, self.policy, policy_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()

    def copy_and_expand_memory(self):
        self.expanded_memory = copy.deepcopy(self.memory)
        self.perform_rollouts()

    def perform_rollouts(self, n_rollouts=1):
        batch = self.expanded_memory.sample(self.batch_size)
        new_states, new_actions, new_rewards, new_next_states, new_dones = [], [], [], [], []

        batch_states = batch[0].cpu().numpy()
        batch_actions = batch[1].cpu().numpy()
        batch_rewards = batch[2].cpu().numpy()
        batch_next_states = batch[3].cpu().numpy()
        batch_dones = batch[4].cpu().numpy()

        for i in range(len(batch_states)):
            for j in range(n_rollouts):
                state = batch_states[i].reshape(1, -1)
                action, _, _ = self.policy.sample(torch.tensor(state, dtype=torch.float32).to(self.device))
                action = action.detach().cpu().numpy()
                next_state, reward, done, _ = self.dynamics.step(state, action)

                new_states.append(next_state)
                new_actions.append(action)
                new_rewards.append(reward)
                new_next_states.append(next_state)
                new_dones.append(done)

                if done:
                    break

        new_states = np.array(new_states)
        new_actions = np.array(new_actions)
        new_rewards = np.array(new_rewards).reshape(-1, 1)
        new_next_states = np.array(new_next_states)
        new_dones = np.array(new_dones).reshape(-1, 1)

        for state, action, reward, next_state, done in zip(new_states, new_actions, new_rewards, new_next_states, new_dones):
            self.expanded_memory.append(state, action, reward, next_state, done, episode_done=done)

    def calc_critic_4redq_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        curr_qs = self.critic.allQs(states, actions)
        target_q = self.calc_target_q(*batch)
        errors = torch.abs(curr_qs[0].detach() - target_q)
        mean_q1 = curr_qs[0].detach().mean().item()
        mean_q2 = curr_qs[1].detach().mean().item()

        losses = []
        for curr_q in curr_qs:
            losses.append(torch.mean((curr_q - target_q).pow(2) * weights))
        return losses, errors, mean_q1, mean_q2

    def calc_critic_loss(self, batch, weights):
        assert self.method in ["sac", "duvn", "monosac"], "This method is only for sac, duvn, or monosac method"
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)
        errors = torch.abs(curr_q1.detach() - target_q)
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        sampled_action, entropy, _ = self.policy.sample(states)

        if self.method == "redq":
            q = self.critic.averageQ(states, sampled_action)
        else:
            q1, q2 = self.critic(states, sampled_action)
            if self.method in ["duvn", "monosac"]:
                q2 = q1
            q = torch.min(q1, q2)
        policy_loss = torch.mean((- q - self.alpha * entropy) * weights)
        return policy_loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        entropy_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropy).detach() * weights)
        return entropy_loss

    def evaluate(self):
        episodes = self.eval_runs
        returns = np.zeros((episodes,), dtype=np.float32)
        sar_buf = [[] for _ in range(episodes)]

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                sar_buf[i].append([state, action, reward])

            returns[i] = episode_reward

        mean_return = np.mean(returns)

        mc_discounted_return = [deque() for _ in range(episodes)]
        for i in range(episodes):
            for re_tran in reversed(sar_buf[i]):
                if len(mc_discounted_return[i]) > 0:
                    mcret = re_tran[2] + self.gamma_n * mc_discounted_return[i][0]
                else:
                    mcret = re_tran[2]
                mc_discounted_return[i].appendleft(mcret)

        norm_coef = np.mean(list(itertools.chain.from_iterable(mc_discounted_return)))
        norm_coef = math.fabs(norm_coef) + 0.000001

        norm_scores = [[] for _ in range(episodes)]
        for i in range(episodes):
            states = np.array(sar_buf[i], dtype="object")[:, 0].tolist()
            actions = np.array(sar_buf[i], dtype="object")[:, 1].tolist()
            with torch.no_grad():
                state = torch.FloatTensor(states).to(self.device)
                action = torch.FloatTensor(actions).to(self.device)
                if self.method == "redq":
                    q = self.critic.averageQ(state, action)
                else:
                    #--------#
                    q1, q2 = self.critic(state, action)
                    q = 0.5 * (q1 + q2)
                qs = q.to('cpu').numpy()
            for j in range(len(sar_buf[i])):
                score = (qs[j][0] - mc_discounted_return[i][j]) / norm_coef
                norm_scores[i].append(score)

        flatten_norm_score = list(itertools.chain.from_iterable(norm_scores))
        mean_norm_score = np.mean(flatten_norm_score)
        std_norm_score = np.std(flatten_norm_score)

        self.writer.add_scalar('reward/test', mean_return, self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  reward: {mean_return:<5.1f}')
        print('-' * 60)

        with open(self.log_dir + "/reward.csv", "a") as f:
            f.write(f"{self.steps},{mean_return},\n")
            f.flush()

        with open(self.log_dir + "/avrbias.csv", "a") as f:
            f.write(f"{self.steps},{mean_norm_score},\n")
            f.flush()

        with open(self.log_dir + "/stdbias.csv", "a") as f:
            f.write(f"{self.steps},{std_norm_score},\n")
            f.flush()

    def save_models(self):
        self.policy.save(os.path.join(self.model_dir, 'policy.pth'))
        self.critic.save(os.path.join(self.model_dir, 'critic.pth'))
        self.critic_target.save(os.path.join(self.model_dir, 'critic_target.pth'))

    def __del__(self):
        self.writer.close()
        self.env.close()

