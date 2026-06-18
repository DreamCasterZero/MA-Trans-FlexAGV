import numpy as np
import config
import torch
from actor import JobActorNetwork, AGVActorNetwork
from critic import CriticNetwork
from torch.optim.lr_scheduler import LinearLR
import os
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.padding_masks = [] # [新增] 存储 Padding Mask
        self.action_masks = []  # [新增] 存储 Action Mask
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def sample(self):
        batch_step = np.arange(0, len(self.states), self.batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_step]

        # 注意：Mask 取出来后转为 numpy array
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
        self.padding_masks.append(padding_mask) # [新增]
        self.action_masks.append(action_mask)   # [新增]
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.padding_masks = [] # [新增]
        self.action_masks = []  # [新增]
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class Job_Agent:
    def __init__(self, input_size, batch_size=32, n_epochs=5, policy_clip=0.2, gamma=0.99,
                 alpha=2e-4, gae_lambda=0.95, device=config.device, total_updates=3000, initial_step=0):
        self.n_epochs = n_epochs
        self.policy_clip = policy_clip
        self.gamma = gamma
        self.lr = alpha
        self.gae_lambda = gae_lambda
        self.device = device
        # 保存总更新次数，用于计算衰减进度
        self.total_updates = total_updates
        self.current_update = 0  # [新增] 当前更新次数计数器
        self.learn_step_counter = initial_step

        self.job_actor = JobActorNetwork(input_size).to(config.device)
        self.job_critic = CriticNetwork(input_size, name='job_critic').to(config.device)
        self.memory = PPOMemory(batch_size)
        self.job_actor_optimizer = torch.optim.Adam(self.job_actor.parameters(), lr=alpha)
        self.job_critic_optimizer = torch.optim.Adam(self.job_critic.parameters(), lr=alpha)

        # 学习率衰减：每训练1000次，学习率乘以0.9
        self.job_actor_scheduler = LinearLR(
            self.job_actor_optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_updates  # <--- 关键修改
        )
        self.job_critic_scheduler = LinearLR(
            self.job_critic_optimizer,
            start_factor=1.0,
            end_factor=0.1,  # 保持同步，也是降到 10%
            total_iters=total_updates
        )
        self.writer = None
        self.learn_step_counter = 0

    def choose_action(self, state, padding_mask, action_mask, greedy=False, temperature=1.0):
        state = state.to(self.device)
        padding_mask = padding_mask.to(self.device)
        action_mask = action_mask.to(self.device)
        dist = self.job_actor(state, padding_mask, action_mask, temperature=temperature)
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
        # 存入 memory
        self.memory.push(state, padding_mask, action_mask, action, probs, vals, reward, done)

    def save(self,i):
        self.job_actor.save_checkpoint(i)
        self.job_critic.save_checkpoint(i)

    def learn(self):
        # 策略: 0.05 -> 0.01 线性衰减
        ent_start = 0.05
        ent_end = 0.01
        # 计算进度 (0.0 -> 1.0)
        progress = min(1.0, self.current_update / self.total_updates)
        entropy_coef = ent_start - (ent_start - ent_end) * progress

        for _ in range(self.n_epochs):
            state_arr, padding_mask_arr, action_mask_arr, action_arr, \
                old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.sample()
            values = vals_arr
            # === GAE 计算 ===
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            gae = 0
            # 从最后一步往前推
            for t in reversed(range(len(reward_arr) - 1)):
                mask = 1.0 - dones_arr[t]
                delta = reward_arr[t] + self.gamma * values[t + 1] * mask - values[t]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                advantage[t] = gae
            advantage = torch.tensor(advantage).to(self.device)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            values = torch.tensor(values).to(self.device)

            for batch in batches:
                # 这样后续索引 advantage 时就不会报警，且运行更稳健
                batch_indices = torch.tensor(batch, dtype=torch.long).to(self.device)
                # 转换数据为 Tensor
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                # [关键] 动作还原：Memory存的是1-based，这里要转回0-based给网络计算log_prob
                actions = torch.tensor(action_arr[batch] - 1).to(self.device)
                # [关键] Mask 转换为 Tensor
                padding_masks = torch.tensor(padding_mask_arr[batch]).to(self.device)
                action_masks = torch.tensor(action_mask_arr[batch]).to(self.device)
                # === 前向传播 (带 Mask) ===
                dist = self.job_actor(states, padding_masks, action_masks)
                critic_value = self.job_critic(states, padding_masks)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # dist_entropy = dist.entropy().mean()
                # === 改为 Masked Entropy ===
                # 1. 算出刚才 forward 时的 softmax 概率 (注意 dist 内部已经是 masked logits)
                log_probs = torch.log_softmax(dist.logits, dim=-1)
                probs = torch.softmax(dist.logits, dim=-1)
                # 2. 只计算 action_mask 为 True (合法动作) 的部分的熵
                # action_masks 是 (Batch, Seq_Len)
                entropy = -torch.sum(probs * log_probs * action_masks, dim=-1)
                # 3. 归一化 (可选)：除以合法动作的数量，让 entropy 变成“平均每个合法动作的不确定性”
                valid_count = action_masks.sum(dim=-1)
                # 避免除以 0
                dist_entropy = (entropy / (valid_count + 1e-9)).mean()

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
                torch.nn.utils.clip_grad_norm_(self.job_actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.job_critic.parameters(), 1.0)
                self.job_actor_optimizer.step()
                self.job_critic_optimizer.step()
                if self.writer:
                    self.writer.add_scalar("job_Loss/actor", actor_loss.item(), self.learn_step_counter)
                    self.writer.add_scalar("job_Loss/critic", critic_loss.item(), self.learn_step_counter)
                    self.writer.add_scalar("job_Loss/total", total_loss.item(), self.learn_step_counter)
                    self.writer.add_scalar("job_Stats/advantage_mean", advantage.mean().item(), self.learn_step_counter)
                    self.writer.add_scalar("job_Stats/entropy", dist_entropy.item(), self.learn_step_counter)
                self.learn_step_counter += 1
        # Scheduler 更新
        self.job_actor_scheduler.step()
        self.job_critic_scheduler.step()
        self.memory.clear()
        self.current_update += 1

    def save_snapshot(self, path):
        """
        保存完整的训练状态（包含模型权重 + 优化器状态）
        path: 外部传入的完整文件路径，例如 './run/job_agent_1000.pt'
        """
        torch.save({
            'actor_state': self.job_actor.state_dict(),
            'critic_state': self.job_critic.state_dict(),
            'optim_actor_state': self.job_actor_optimizer.state_dict(),  # 🔥 保存 Actor 优化器
            'optim_critic_state': self.job_critic_optimizer.state_dict()  # 🔥 保存 Critic 优化器
        }, path)

    def load_snapshot(self, path):
        """加载完整的训练状态"""
        if not os.path.exists(path):
            print(f"⚠️ 文件不存在: {path}")
            return

        checkpoint = torch.load(path)

        # 1. 加载网络权重
        self.job_actor.load_state_dict(checkpoint['actor_state'])
        self.job_critic.load_state_dict(checkpoint['critic_state'])

        # 2. 🔥 加载优化器状态 (这是断点续训的关键)
        self.job_actor_optimizer.load_state_dict(checkpoint['optim_actor_state'])
        self.job_critic_optimizer.load_state_dict(checkpoint['optim_critic_state'])

        print(f"✅ 成功加载 Job_Agent 断点: {path}")


class AGV_Agent:
    def __init__(self, input_size, batch_size=32, n_epochs=5, policy_clip=0.2, gamma=0.99, alpha=2e-4, gae_lambda=0.95,
                 device=config.device, total_updates=3000, initial_step=0):
        self.n_epochs = n_epochs
        self.policy_clip = policy_clip
        self.gamma = gamma
        self.lr = alpha
        self.gae_lambda = gae_lambda
        self.device = device
        # [新增] 计数器
        self.total_updates = total_updates
        self.current_update = 0
        self.learn_step_counter = initial_step
        self.agv_actor = AGVActorNetwork(input_size).to(config.device)
        # [修改] 传入 name='agv_critic'
        self.agv_critic = CriticNetwork(input_size, name='agv_critic').to(config.device)

        self.memory = PPOMemory(batch_size)
        self.agv_actor_optimizer = torch.optim.Adam(self.agv_actor.parameters(), lr=alpha)
        self.agv_critic_optimizer = torch.optim.Adam(self.agv_critic.parameters(), lr=alpha)

        self.agv_actor_scheduler = LinearLR(self.agv_actor_optimizer, start_factor=1.0, end_factor=0.1,
                                            total_iters=total_updates)
        self.agv_critic_scheduler = LinearLR(
            self.agv_critic_optimizer,
            start_factor=1.0,
            end_factor=0.1,  # 保持同步，也是降到 10%
            total_iters=total_updates
        )
        self.writer = None
        self.learn_step_counter = 0

    def choose_action(self, state, padding_mask, action_mask, greedy=False, temperature=1.0):
        state = state.to(self.device)
        padding_mask = padding_mask.to(self.device)
        action_mask = action_mask.to(self.device)

        # 传入所有 Mask
        dist = self.agv_actor(state, padding_mask, action_mask, temperature=temperature)

        if greedy:
            action = torch.argmax(dist.probs)
        else:
            action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action))
        action_scalar = torch.squeeze(action).item() + 1

        # Critic 带 Mask
        value = self.agv_critic(state, padding_mask)
        value = torch.squeeze(value)
        return action_scalar, probs, value

    def remember(self, state, padding_mask, action_mask, action, probs, vals, reward, done):
        self.memory.push(state, padding_mask, action_mask, action, probs, vals, reward, done)

    def save(self, i):
        self.agv_actor.save_checkpoint(i)
        self.agv_critic.save_checkpoint(i)

    def learn(self):
        # [新增] 动态计算熵系数
        ent_start = 0.05
        ent_end = 0.01
        progress = min(1.0, self.current_update / self.total_updates)
        entropy_coef = ent_start - (ent_start - ent_end) * progress
        for _ in range(self.n_epochs):
            # [修改] 解包 mask
            state_arr, padding_mask_arr, action_mask_arr, action_arr, \
                old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.sample()

            values = vals_arr

            # GAE (保持不变)
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
                # 这样后续索引 advantage 时就不会报警，且运行更稳健
                batch_indices = torch.tensor(batch, dtype=torch.long).to(self.device)
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                # 动作 1-based 转 0-based
                actions = torch.tensor(action_arr[batch] - 1).to(self.device)

                # Mask 转 Tensor
                padding_masks = torch.tensor(padding_mask_arr[batch]).to(self.device)
                action_masks = torch.tensor(action_mask_arr[batch]).to(self.device)

                # === 前向传播 (带 Mask) ===
                dist = self.agv_actor(states, padding_masks, action_masks)
                critic_value = self.agv_critic(states, padding_masks)

                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                # dist_entropy = dist.entropy().mean()
                # === 改为 Masked Entropy ===
                # 1. 算出刚才 forward 时的 softmax 概率 (注意 dist 内部已经是 masked logits)
                log_probs = torch.log_softmax(dist.logits, dim=-1)
                probs = torch.softmax(dist.logits, dim=-1)
                # 2. 只计算 action_mask 为 True (合法动作) 的部分的熵
                # action_masks 是 (Batch, Seq_Len)
                entropy = -torch.sum(probs * log_probs * action_masks, dim=-1)
                # 3. 归一化 (可选)：除以合法动作的数量，让 entropy 变成“平均每个合法动作的不确定性”
                valid_count = action_masks.sum(dim=-1)
                # 避免除以 0
                dist_entropy = (entropy / (valid_count + 1e-9)).mean()

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
                torch.nn.utils.clip_grad_norm_(self.agv_actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.agv_critic.parameters(), 1.0)
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
        self.memory.clear()
        self.current_update += 1

    def save_snapshot(self, path):
        """
        保存完整的训练状态（包含模型权重 + 优化器状态）
        path: 外部传入的完整文件路径，例如 './run/job_agent_1000.pt'
        """
        torch.save({
            'actor_state': self.agv_actor.state_dict(),
            'critic_state': self.agv_critic.state_dict(),
            'optim_actor_state': self.agv_actor_optimizer.state_dict(),  # 🔥 保存 Actor 优化器
            'optim_critic_state': self.agv_critic_optimizer.state_dict()  # 🔥 保存 Critic 优化器
        }, path)

    def load_snapshot(self, path):
        """加载完整的训练状态"""
        if not os.path.exists(path):
            print(f"⚠️ 文件不存在: {path}")
            return

        checkpoint = torch.load(path)

        # 1. 加载网络权重
        self.agv_actor.load_state_dict(checkpoint['actor_state'])
        self.agv_critic.load_state_dict(checkpoint['critic_state'])

        # 2. 🔥 加载优化器状态 (这是断点续训的关键)
        self.agv_actor_optimizer.load_state_dict(checkpoint['optim_actor_state'])
        self.agv_critic_optimizer.load_state_dict(checkpoint['optim_critic_state'])

        print(f"✅ 成功加载 Job_Agent 断点: {path}")

