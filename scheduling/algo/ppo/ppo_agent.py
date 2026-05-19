import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from .rollout_buffer import RolloutBuffer


class PPOAgent:
    def __init__(self, actor_net, critic_net, cfg, device):
        self.actor   = actor_net.to(device)
        self.critic  = critic_net.to(device)
        self.cfg     = cfg.ppo
        self.device  = device
        self.buffer  = RolloutBuffer()

        self.actor_optim  = Adam(self.actor.parameters(),  lr=self.cfg.learning_rate)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.cfg.learning_rate)
        self.actor_scheduler  = None
        self.critic_scheduler = None

        self.current_update = 0
        self.learn_step     = 0

    def init_schedulers(self, total_updates):
        self.actor_scheduler  = LinearLR(
            self.actor_optim,  start_factor=1.0, end_factor=0.1, total_iters=total_updates)
        self.critic_scheduler = LinearLR(
            self.critic_optim, start_factor=1.0, end_factor=0.1, total_iters=total_updates)

    @torch.no_grad()
    def choose_action(self, state, padding_mask, action_mask, greedy=False):
        state        = state.to(self.device)
        padding_mask = padding_mask.to(self.device)
        action_mask  = action_mask.to(self.device)

        dist  = self.actor(state, padding_mask, action_mask)
        value = self.critic(state, padding_mask)

        action   = dist.probs.argmax() if greedy else dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.squeeze(), value.squeeze()

    def push(self, state, padding_mask, action_mask, action, log_prob, value, reward, done):
        self.buffer.push(state, padding_mask, action_mask, action, log_prob, value, reward, done)

    def learn(self, total_updates, mini_batch_size, writer=None, prefix=""):
        progress     = min(1.0, self.current_update / (total_updates + 1))
        entropy_coef = self.cfg.ent_coef_start - (self.cfg.ent_coef_start - self.cfg.ent_coef_end) * progress

        (states_arr, pad_masks_arr, act_masks_arr,
         actions_arr, old_lp_arr, values_arr,
         rewards_arr, dones_arr, batches) = self.buffer.sample(mini_batch_size)

        # GAE（在所有 epoch 前统一计算）
        advantage = np.zeros(len(rewards_arr), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards_arr) - 1)):
            delta       = (rewards_arr[t]
                           + self.cfg.gamma * values_arr[t + 1] * (1 - dones_arr[t])
                           - values_arr[t])
            gae         = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - dones_arr[t]) * gae
            advantage[t] = gae

        advantage_t = torch.tensor(advantage, device=self.device)
        advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-8)
        values_t    = torch.tensor(values_arr, device=self.device)

        self.actor.train()
        self.critic.train()

        for _ in range(self.cfg.n_epochs):
            for batch in batches:
                idx = torch.tensor(batch, dtype=torch.long, device=self.device)

                states    = torch.tensor(states_arr[batch],    dtype=torch.float, device=self.device)
                pad_masks = torch.tensor(pad_masks_arr[batch],                    device=self.device)
                act_masks = torch.tensor(act_masks_arr[batch],                    device=self.device)
                old_lp    = torch.tensor(old_lp_arr[batch],                       device=self.device)
                actions   = torch.tensor(actions_arr[batch],                       device=self.device)

                dist         = self.actor(states, pad_masks, act_masks)
                critic_value = self.critic(states, pad_masks).squeeze()

                new_lp = dist.log_prob(actions)
                ratio  = (new_lp - old_lp).exp()

                # Masked entropy（只在合法动作上计算）
                p   = torch.softmax(dist.logits, dim=-1)
                lp  = torch.log_softmax(dist.logits, dim=-1)
                ent = -torch.sum(p * lp * act_masks, dim=-1)
                dist_entropy = (ent / (act_masks.sum(dim=-1) + 1e-9)).mean()

                adv    = advantage_t[idx]
                a_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - self.cfg.policy_clip, 1 + self.cfg.policy_clip) * adv,
                ).mean() - entropy_coef * dist_entropy

                returns = adv + values_t[idx]
                c_loss  = ((returns - critic_value) ** 2).mean()

                loss = a_loss + self.cfg.value_loss_coef * c_loss

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),  self.cfg.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                if writer:
                    writer.add_scalar(f"{prefix}/actor_loss",  a_loss.item(),       self.learn_step)
                    writer.add_scalar(f"{prefix}/critic_loss", c_loss.item(),        self.learn_step)
                    writer.add_scalar(f"{prefix}/entropy",     dist_entropy.item(),  self.learn_step)
                self.learn_step += 1

        if self.actor_scheduler:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        self.buffer.clear()
        self.current_update += 1

    def save(self, path):
        torch.save({
            'actor':          self.actor.state_dict(),
            'critic':         self.critic.state_dict(),
            'actor_optim':    self.actor_optim.state_dict(),
            'critic_optim':   self.critic_optim.state_dict(),
            'current_update': self.current_update,
            'learn_step':     self.learn_step,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_optim.load_state_dict(ckpt['actor_optim'])
        self.critic_optim.load_state_dict(ckpt['critic_optim'])
        self.current_update = ckpt.get('current_update', 0)
        self.learn_step     = ckpt.get('learn_step', 0)
