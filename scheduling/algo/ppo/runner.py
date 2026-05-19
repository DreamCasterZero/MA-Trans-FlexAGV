import os
import random
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from .actor_network  import ActorNetwork
from .critic_network import CriticNetwork
from .ppo_agent      import PPOAgent


class Runner:
    def __init__(self, env, train_cfg, log_dir=None):
        self.env      = env
        self.cfg      = train_cfg          # SchedulingConfigAlgo
        self.device   = env.device
        self.log_dir  = log_dir
        self.writer   = None

        net  = train_cfg.network
        fdim = env.feat_dim

        job_actor  = ActorNetwork( fdim, net.embed_size, net.num_heads, net.num_layers, net.dropout_rate)
        job_critic = CriticNetwork(fdim, net.embed_size, net.num_heads, net.num_layers, net.dropout_rate)
        agv_actor  = ActorNetwork( fdim, net.embed_size, net.num_heads, net.num_layers, net.dropout_rate)
        agv_critic = CriticNetwork(fdim, net.embed_size, net.num_heads, net.num_layers, net.dropout_rate)

        self.job_agent = PPOAgent(job_actor,  job_critic, train_cfg, self.device)
        self.agv_agent = PPOAgent(agv_actor,  agv_critic, train_cfg, self.device)

        self.current_episode = 0

    # ─── 主入口 ───────────────────────────────────────────────
    def run(self, val_dataset=None):
        total_eps     = self.cfg.runner.total_episodes
        total_updates = total_eps // self.cfg.runner.batch_episodes
        log_interval  = getattr(self.cfg.runner, 'log_interval', 10)

        self.job_agent.init_schedulers(total_updates)
        self.agv_agent.init_schedulers(total_updates)

        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)

        current_instance  = self._generate_instance()
        ep_buffer_count   = 0
        best_val_makespan = float('inf')

        # rolling window accumulators (reset every log_interval)
        _COMP_KEYS = ('transport', 'agv_wait', 'makespan_r', 'completion', 'balance')
        win_makespan  = 0.0
        win_job_score = 0.0
        win_agv_score = 0.0
        win_ep_time   = 0.0
        win_comps     = {k: 0.0 for k in _COMP_KEYS}

        start_time = time.time()

        print(f"{'─'*56}")
        print(f"  Task           : {self.cfg.runner.experiment_name}")
        print(f"  Total episodes : {total_eps}  |  Updates : {total_updates}")
        print(f"  Batch          : {self.cfg.runner.batch_episodes} ep"
              f"  |  Log every : {log_interval} ep")
        print(f"{'─'*56}")

        for ep in range(self.current_episode, total_eps + 1):
            if ep > 0 and ep % self.cfg.runner.instance_change_freq == 0:
                current_instance = self._generate_instance()

            ep_t0 = time.time()
            job_score, agv_score, comps = self._collect_episode(*current_instance)
            win_ep_time   += time.time() - ep_t0

            win_makespan  += self.env.fjsp.makespan
            win_job_score += job_score
            win_agv_score += agv_score
            for k in _COMP_KEYS:
                win_comps[k] += comps[k]
            ep_buffer_count += 1

            if ep_buffer_count >= self.cfg.runner.batch_episodes:
                self.job_agent.learn(
                    total_updates, self.cfg.runner.batch_episodes, self.writer, "job")
                self.agv_agent.learn(
                    total_updates, self.cfg.runner.batch_episodes, self.writer, "agv")
                ep_buffer_count = 0

            if self.writer:
                self.writer.add_scalar("Train/makespan",   self.env.fjsp.makespan, ep)
                self.writer.add_scalar("Train/job_reward", job_score,              ep)
                self.writer.add_scalar("Train/agv_reward", agv_score,              ep)
                for k in _COMP_KEYS:
                    self.writer.add_scalar(f"Train/{k}", comps[k], ep)

            if ep > 0 and ep % log_interval == 0:
                n          = log_interval
                avg_ms     = win_makespan  / n
                avg_jr     = win_job_score / n
                avg_ar     = win_agv_score / n
                avg_t      = win_ep_time   / n
                avg_comps  = {k: win_comps[k] / n for k in _COMP_KEYS}

                elapsed    = time.time() - start_time
                speed      = max(ep - self.current_episode, 1) / elapsed
                eta_secs   = int((total_eps - ep) / speed)
                cur_update = self.job_agent.current_update

                print(f"[Ep {ep:>7}/{total_eps} | Update {cur_update:>5}/{total_updates}]")
                print(f"  ETA      : {self._fmt_time(eta_secs)}  (~{eta_secs} s)")
                print(f"  Makespan : {avg_ms:>8.2f}    Time/ep : {avg_t:.4f} s")
                print(f"  {'─'*48}")
                print(f"  {'':18s}  {'transport':>10}  {'agv_wait':>8}  "
                      f"{'makespan':>8}  {'complet':>7}  {'balance':>7}  {'total':>9}")
                print(f"  {'Job  (rs2+ms+rf1)':18s}  {'—':>10}  "
                      f"{avg_comps['agv_wait']:>8.4f}  "
                      f"{avg_comps['makespan_r']:>8.4f}  "
                      f"{avg_comps['completion']:>7.4f}  {'—':>7}  {avg_jr:>9.4f}")
                print(f"  {'AGV  (all)':18s}  "
                      f"{avg_comps['transport']:>10.4f}  "
                      f"{avg_comps['agv_wait']:>8.4f}  "
                      f"{avg_comps['makespan_r']:>8.4f}  "
                      f"{avg_comps['completion']:>7.4f}  "
                      f"{avg_comps['balance']:>7.4f}  {avg_ar:>9.4f}")
                print()

                win_makespan  = 0.0
                win_job_score = 0.0
                win_agv_score = 0.0
                win_ep_time   = 0.0
                win_comps     = {k: 0.0 for k in _COMP_KEYS}

            if val_dataset and ep > 0 and ep % 100 == 0:
                avg_makespan = self._evaluate(val_dataset)
                if self.writer:
                    self.writer.add_scalar("Eval/avg_makespan", avg_makespan, ep)
                if avg_makespan < best_val_makespan:
                    best_val_makespan = avg_makespan
                    self.save(os.path.join(self.log_dir, "model_best.pt"))
                    print(f"  >>> New best val makespan: {avg_makespan:.2f} (Ep {ep})")
                    print()

            if ep > 0 and ep % self.cfg.runner.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{ep}.pt"))

        self.current_episode = total_eps
        total_time = time.time() - start_time
        print(f"{'─'*56}")
        print(f"  Training done. Total time: {self._fmt_time(total_time)}")
        print(f"{'─'*56}")
        if self.writer:
            self.writer.close()

    @staticmethod
    def _fmt_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ─── 单局 rollout ─────────────────────────────────────────
    def _collect_episode(self, job_data, agv_num):
        state, pad_mask, job_mask, agv_mask = self.env.reset(job_data, agv_num)
        done      = False
        job_score = 0.0
        agv_score = 0.0
        comps = {'transport': 0.0, 'agv_wait': 0.0,
                 'makespan_r': 0.0, 'completion': 0.0, 'balance': 0.0}

        while not done:
            job_act, job_lp, job_val = self.job_agent.choose_action(
                state, pad_mask, job_mask)

            mid_state, pad_mask, agv_mask, mach_act = self.env.job_step(job_act)

            agv_act, agv_lp, agv_val = self.agv_agent.choose_action(
                mid_state, pad_mask, agv_mask)

            next_state, next_pad, next_jmask, next_amask, reward, done, info = \
                self.env.step(job_act, mach_act, agv_act)

            jr = self._job_reward(info)
            ar = self._agv_reward(info)

            self.job_agent.push(state,     pad_mask, job_mask,  job_act, job_lp, job_val, jr, done)
            self.agv_agent.push(mid_state, pad_mask, agv_mask,  agv_act, agv_lp, agv_val, ar, done)

            state, pad_mask, job_mask, agv_mask = next_state, next_pad, next_jmask, next_amask
            job_score += jr
            agv_score += ar
            comps['transport']  += info['transport']
            comps['agv_wait']   += info['agv_wait']
            comps['makespan_r'] += info['makespan']
            comps['completion'] += info['completion']
            comps['balance']    += info['balance']

        return job_score, agv_score, comps

    # ─── 评估 ─────────────────────────────────────────────────
    def _evaluate(self, val_dataset):
        for agent in (self.job_agent, self.agv_agent):
            agent.actor.eval()
            agent.critic.eval()
        total = 0.0
        for case in val_dataset:
            state, pad_mask, job_mask, agv_mask = self.env.reset(
                case['PT'], case['agv_num'])
            done = False
            while not done:
                job_act, _, _ = self.job_agent.choose_action(
                    state, pad_mask, job_mask, greedy=True)
                mid_state, pad_mask, agv_mask, mach_act = self.env.job_step(job_act)
                agv_act, _, _ = self.agv_agent.choose_action(
                    mid_state, pad_mask, agv_mask, greedy=True)
                state, pad_mask, job_mask, agv_mask, _, done, _ = \
                    self.env.step(job_act, mach_act, agv_act)
            total += self.env.fjsp.makespan
        for agent in (self.job_agent, self.agv_agent):
            agent.actor.train()
            agent.critic.train()
        return total / len(val_dataset)

    # ─── 实例生成 ─────────────────────────────────────────────
    def _generate_instance(self):
        prob = self.cfg.problem if hasattr(self.cfg, 'problem') else self.env.cfg.problem
        n_jobs = random.randint(prob.num_jobs_min, prob.num_jobs_max)
        PT = []
        for _ in range(n_jobs):
            num_ops = random.randint(prob.num_ops_min, prob.num_ops_max)
            job_ops = []
            for _ in range(num_ops):
                row     = [0] * self.env.num_machines
                capable = random.sample(range(self.env.num_machines),
                                        random.randint(prob.num_capable_min, prob.num_capable_max))
                for m in capable:
                    row[m] = random.randint(prob.proc_time_min, prob.proc_time_max)
                job_ops.append(row)
            PT.append(job_ops)
        return PT, prob.num_agvs

    # ─── 奖励分配 ─────────────────────────────────────────────
    def _job_reward(self, info):
        scale = self.env.cfg.reward.job_makespan_scale
        return info['agv_wait'] + info['makespan'] * scale + info['completion']

    def _agv_reward(self, info):
        scale = self.env.cfg.reward.agv_makespan_scale
        return (info['transport'] + info['agv_wait']
                + info['makespan'] * scale + info['balance'] + info['completion'])

    # ─── 保存 / 加载 ──────────────────────────────────────────
    def save(self, path):
        torch.save({
            'job_agent':       self._agent_state(self.job_agent),
            'agv_agent':       self._agent_state(self.agv_agent),
            'current_episode': self.current_episode,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self._load_agent(self.job_agent, ckpt['job_agent'])
        self._load_agent(self.agv_agent, ckpt['agv_agent'])
        self.current_episode = ckpt.get('current_episode', 0)

    @staticmethod
    def _agent_state(agent):
        return {
            'actor':          agent.actor.state_dict(),
            'critic':         agent.critic.state_dict(),
            'actor_optim':    agent.actor_optim.state_dict(),
            'critic_optim':   agent.critic_optim.state_dict(),
            'current_update': agent.current_update,
            'learn_step':     agent.learn_step,
        }

    @staticmethod
    def _load_agent(agent, state):
        agent.actor.load_state_dict(state['actor'])
        agent.critic.load_state_dict(state['critic'])
        agent.actor_optim.load_state_dict(state['actor_optim'])
        agent.critic_optim.load_state_dict(state['critic_optim'])
        agent.current_update = state.get('current_update', 0)
        agent.learn_step     = state.get('learn_step', 0)
