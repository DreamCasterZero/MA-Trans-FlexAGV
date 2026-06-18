import numpy as np
import config
import torch
from gnn_actor import JobActorNetwork, AGVActorNetwork
from gnn_critic import CriticNetwork
from torch.optim.lr_scheduler import LinearLR


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.padding_masks = []
        self.action_masks = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def sample(self):
        # 确保至少有一个 batch
        n_states = len(self.states)
        if n_states < self.batch_size:
            return None  # 或者抛出异常，视情况而定

        batch_step = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_step]

        return np.array([s.cpu().detach().numpy() for s in self.states]), \
            np.array([pm.cpu().detach().numpy() for pm in self.padding_masks]), \
            np.array([am.cpu().detach().numpy() for am in self.action_masks]), \
            np.array(self.actions), \
            np.array([p.cpu().detach().numpy() for p in self.probs]), \
            np.array([v.cpu().detach().numpy() for v in self.vals]), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    def push(self, state, padding_mask, action_mask, action, probs, vals, reward, done):
        self.states.append(state)
        self.padding_masks.append(padding_mask)
        self.action_masks.append(action_mask)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.padding_masks = []
        self.action_masks = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class Job_Agent:
    def __init__(self, input_size, batch_size=32, n_epochs=5, policy_clip=0.2, gamma=0.99, alpha=2e-4, gae_lambda=0.95,
                 device=config.device, total_updates=3000, initial_step=0):
        self.n_epochs = n_epochs
        self.policy_clip = policy_clip
        self.gamma = gamma
        self.lr = alpha
        self.gae_lambda = gae_lambda
        self.device = device
        self.total_updates = total_updates
        self.current_update = 0
        self.learn_step_counter = initial_step

        # === 🟢 初始化 GNN 网络 ===
        # 注意: 这里的 JobActorNetwork 已经是 import from gnn_actor
        self.job_actor = JobActorNetwork(input_size).to(config.device)
        self.job_critic = CriticNetwork(input_size, name='job_critic').to(config.device)

        self.memory = PPOMemory(batch_size)
        self.job_actor_optimizer = torch.optim.Adam(self.job_actor.parameters(), lr=alpha)
        self.job_critic_optimizer = torch.optim.Adam(self.job_critic.parameters(), lr=alpha)

        # 线性衰减 Scheduler
        self.job_actor_scheduler = LinearLR(
            self.job_actor_optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_updates
        )
        self.job_critic_scheduler = LinearLR(
            self.job_critic_optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_updates
        )
        self.writer = None
        self.learn_step_counter = 0

    def choose_action(self, state, padding_mask, action_mask, greedy=False):
        state = state.to(self.device)
        padding_mask = padding_mask.to(self.device)
        action_mask = action_mask.to(self.device)

        dist = self.job_actor(state, padding_mask, action_mask)

        if greedy:
            action = torch.argmax(dist.probs)
        else:
            action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action))
        action_scalar = torch.squeeze(action).item() + 1

        value = self.job_critic(state, padding_mask)
        value = torch.squeeze(value)

        return action_scalar, probs, value

    def remember(self, state, padding_mask, action_mask, action, probs, vals, reward, done):
        self.memory.push(state, padding_mask, action_mask, action, probs, vals, reward, done)

    def save(self, i):
        self.job_actor.save_checkpoint(i)
        self.job_critic.save_checkpoint(i)

    def learn(self):
        # 1. 动态计算熵系数
        ent_start = 0.05
        ent_end = 0.01
        # === 🟢 修改为：带 Warmup 的滞后衰减 (与 Transformer 对齐) ===
        progress = min(1.0, self.current_update / self.total_updates)
        entropy_coef = ent_start - (ent_start - ent_end) * progress
        #
        # # 前 30% 保持高探索率 (0.05)，防止过早收敛
        # if progress < 0.3:
        #     entropy_coef = ent_start
        # else:
        #     # 剩下的 70% 时间再线性衰减
        #     decay_progress = (progress - 0.3) / 0.7
        #     entropy_coef = ent_start - (ent_start - ent_end) * decay_progress

        # 2. 从 Memory 采样
        # 注意处理 sample 可能为空的情况（虽然按逻辑不太可能）
        sample_result = self.memory.sample()
        if sample_result is None:
            return

        state_arr, padding_mask_arr, action_mask_arr, action_arr, \
            old_prob_arr, vals_arr, reward_arr, dones_arr, batches = sample_result

        values = vals_arr

        # 3. GAE 计算
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        gae = 0
        for t in reversed(range(len(reward_arr) - 1)):
            mask = 1.0 - dones_arr[t]
            delta = reward_arr[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantage[t] = gae

        advantage = torch.tensor(advantage).to(self.device)
        # 归一化 Advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        values = torch.tensor(values).to(self.device)

        # 4. Mini-batch 更新
        for batch in batches:
            batch_indices = torch.tensor(batch, dtype=torch.long).to(self.device)
            states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
            old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
            actions = torch.tensor(action_arr[batch] - 1).to(self.device)  # 0-based
            padding_masks = torch.tensor(padding_mask_arr[batch]).to(self.device)
            action_masks = torch.tensor(action_mask_arr[batch]).to(self.device)

            # --- 前向传播 (GNN) ---
            dist = self.job_actor(states, padding_masks, action_masks)
            critic_value = self.job_critic(states, padding_masks)

            critic_value = torch.squeeze(critic_value)
            new_probs = dist.log_prob(actions)
            prob_ratio = new_probs.exp() / old_probs.exp()

            # 原代码:
            # dist_entropy = dist.entropy().mean()

            # === 🟢 修改为: Masked Entropy (与 Transformer 保持一致) ===
            # 1. 算出 Log Softmax 和 Softmax
            log_probs = torch.log_softmax(dist.logits, dim=-1)
            probs = torch.softmax(dist.logits, dim=-1)

            # 2. 只计算 action_mask 为 True (合法动作) 的部分的熵
            # 注意: action_masks 在 gnn_ppo 里如果是 bool 型，可能需要转 float
            # entropy = -torch.sum(probs * log_probs * action_masks, dim=-1)
            # 稳健写法:
            entropy = -torch.sum(probs * log_probs * action_masks.float(), dim=-1)

            # 3. 归一化
            valid_count = action_masks.sum(dim=-1)
            dist_entropy = (entropy / (valid_count + 1e-9)).mean()
            # ========================================================

            current_advantage = advantage[batch_indices]
            weighted_probs = current_advantage * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * current_advantage

            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean() - entropy_coef * dist_entropy

            returns = current_advantage + values[batch_indices]
            critic_loss = (returns - critic_value) ** 2
            critic_loss = critic_loss.mean()

            total_loss = actor_loss + 0.5 * critic_loss

            self.job_actor_optimizer.zero_grad()
            self.job_critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.job_actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.job_critic.parameters(), 0.5)
            self.job_actor_optimizer.step()
            self.job_critic_optimizer.step()

            if self.writer:
                self.writer.add_scalar("job_Loss/actor", actor_loss.item(), self.learn_step_counter)
                self.writer.add_scalar("job_Loss/critic", critic_loss.item(), self.learn_step_counter)
                self.writer.add_scalar("job_Loss/total", total_loss.item(), self.learn_step_counter)
                self.writer.add_scalar("job_Stats/advantage_mean", advantage.mean().item(), self.learn_step_counter)
                self.writer.add_scalar("job_Stats/entropy", dist_entropy.item(), self.learn_step_counter)
            self.learn_step_counter += 1

        # 5. 更新 Scheduler 和 计数器
        self.job_actor_scheduler.step()
        self.job_critic_scheduler.step()
        self.current_update += 1  # [关键] 更新次数自增
        self.memory.clear()


class AGV_Agent:
    def __init__(self, input_size, batch_size=32, n_epochs=5, policy_clip=0.2, gamma=0.99, alpha=2e-4, gae_lambda=0.95,
                 device=config.device, total_updates=3000, initial_step=0):
        self.n_epochs = n_epochs
        self.policy_clip = policy_clip
        self.gamma = gamma
        self.lr = alpha
        self.gae_lambda = gae_lambda
        self.device = device
        self.total_updates = total_updates
        self.current_update = 0
        self.learn_step_counter = initial_step
        # === 🟢 初始化 AGV GNN 网络 ===
        self.agv_actor = AGVActorNetwork(input_size).to(config.device)
        self.agv_critic = CriticNetwork(input_size, name='agv_critic').to(config.device)

        self.memory = PPOMemory(batch_size)
        self.agv_actor_optimizer = torch.optim.Adam(self.agv_actor.parameters(), lr=alpha)
        self.agv_critic_optimizer = torch.optim.Adam(self.agv_critic.parameters(), lr=alpha)

        self.agv_actor_scheduler = LinearLR(self.agv_actor_optimizer, start_factor=1.0, end_factor=0.1,
                                            total_iters=total_updates)
        self.agv_critic_scheduler = LinearLR(
            self.agv_critic_optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_updates
        )
        self.writer = None
        self.learn_step_counter = 0

    def choose_action(self, state, padding_mask, action_mask, greedy=False):
        state = state.to(self.device)
        padding_mask = padding_mask.to(self.device)
        action_mask = action_mask.to(self.device)

        dist = self.agv_actor(state, padding_mask, action_mask)

        if greedy:
            action = torch.argmax(dist.probs)
        else:
            action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action))
        action_scalar = torch.squeeze(action).item() + 1

        value = self.agv_critic(state, padding_mask)
        value = torch.squeeze(value)
        return action_scalar, probs, value

    def remember(self, state, padding_mask, action_mask, action, probs, vals, reward, done):
        self.memory.push(state, padding_mask, action_mask, action, probs, vals, reward, done)

    def save(self, i):
        self.agv_actor.save_checkpoint(i)
        self.agv_critic.save_checkpoint(i)

    def learn(self):
        ent_start = 0.05
        ent_end = 0.01
        # === 🟢 修改为：带 Warmup 的滞后衰减 (与 Transformer 对齐) ===
        progress = min(1.0, self.current_update / self.total_updates)
        entropy_coef = ent_start - (ent_start - ent_end) * progress

        sample_result = self.memory.sample()
        if sample_result is None:
            return

        state_arr, padding_mask_arr, action_mask_arr, action_arr, \
            old_prob_arr, vals_arr, reward_arr, dones_arr, batches = sample_result

        values = vals_arr

        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        gae = 0
        for t in reversed(range(len(reward_arr) - 1)):
            mask = 1.0 - dones_arr[t]
            delta = reward_arr[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantage[t] = gae

        advantage = torch.tensor(advantage).to(self.device)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        values = torch.tensor(values).to(self.device)

        for batch in batches:
            batch_indices = torch.tensor(batch, dtype=torch.long).to(self.device)
            states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
            old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
            actions = torch.tensor(action_arr[batch] - 1).to(self.device)
            padding_masks = torch.tensor(padding_mask_arr[batch]).to(self.device)
            action_masks = torch.tensor(action_mask_arr[batch]).to(self.device)

            dist = self.agv_actor(states, padding_masks, action_masks)
            critic_value = self.agv_critic(states, padding_masks)

            critic_value = torch.squeeze(critic_value)
            new_probs = dist.log_prob(actions)
            prob_ratio = new_probs.exp() / old_probs.exp()

            # 原代码:
            # dist_entropy = dist.entropy().mean()

            # === 🟢 修改为: Masked Entropy (与 Transformer 保持一致) ===
            # 1. 算出 Log Softmax 和 Softmax
            log_probs = torch.log_softmax(dist.logits, dim=-1)
            probs = torch.softmax(dist.logits, dim=-1)

            # 2. 只计算 action_mask 为 True (合法动作) 的部分的熵
            # 注意: action_masks 在 gnn_ppo 里如果是 bool 型，可能需要转 float
            # entropy = -torch.sum(probs * log_probs * action_masks, dim=-1)
            # 稳健写法:
            entropy = -torch.sum(probs * log_probs * action_masks.float(), dim=-1)

            # 3. 归一化
            valid_count = action_masks.sum(dim=-1)
            dist_entropy = (entropy / (valid_count + 1e-9)).mean()
            # ========================================================
            current_advantage = advantage[batch_indices]
            weighted_probs = current_advantage * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * current_advantage
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean() - entropy_coef * dist_entropy

            returns = current_advantage + values[batch_indices]
            critic_loss = (returns - critic_value) ** 2
            critic_loss = critic_loss.mean()

            total_loss = actor_loss + 0.5 * critic_loss

            self.agv_actor_optimizer.zero_grad()
            self.agv_critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agv_actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.agv_critic.parameters(), 0.5)
            self.agv_actor_optimizer.step()
            self.agv_critic_optimizer.step()

            if self.writer:
                self.writer.add_scalar("agv_Loss/actor", actor_loss.item(), self.learn_step_counter)
                self.writer.add_scalar("agv_Loss/critic", critic_loss.item(), self.learn_step_counter)
                self.writer.add_scalar("agv_Loss/total", total_loss.item(), self.learn_step_counter)
                self.writer.add_scalar("agv_Stats/advantage_mean", advantage.mean().item(), self.learn_step_counter)
                self.writer.add_scalar("agv_Stats/entropy", dist_entropy.item(), self.learn_step_counter)
            self.learn_step_counter += 1

        self.agv_actor_scheduler.step()
        self.agv_critic_scheduler.step()
        self.current_update += 1  # [关键]
        self.memory.clear()