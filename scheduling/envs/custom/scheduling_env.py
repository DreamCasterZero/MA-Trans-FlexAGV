import torch
import numpy as np
from ..base.base_task import BaseTask
from ..core.FJSP import FJSP, INFEASIBLE

class SchedulingEnv(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._init_env()
        self._init_buffers()
    
    # ─── Init ────────────────────────────────────────────────
    def _init_env(self):
        # 从 cfg 读取所有维度参数，创建 FJSP 实例
        self.num_machines = self.cfg.env.num_machines
        self.max_seq_len  = self.cfg.env.max_seq_len
        self.feat_dim     = self.cfg.env.feat_dim

        self.agv_trans = self.cfg.transport.agv_trans

        self.norm_time      = self.cfg.obs.norm_time
        self.norm_dist      = self.cfg.obs.norm_dist
        self.norm_count     = self.cfg.obs.norm_count
        self.norm_agv_count = self.cfg.obs.norm_agv_count

        self.fjsp = FJSP(self.num_machines)

        self.num_jobs     = 0
        self.num_agvs     = 0
        self.last_makespan = 0

    def _init_buffers(self):
        # 预先构建 job_type/agv_type tensor
        self.job_type = torch.tensor(
            self.cfg.obs.job_type, dtype=torch.float, device=self.device)
        self.agv_type = torch.tensor(
            self.cfg.obs.agv_type, dtype=torch.float, device=self.device)

        self.state           = torch.zeros(self.max_seq_len, self.feat_dim, device=self.device)
        self.padding_mask    = torch.ones(self.max_seq_len, dtype=torch.bool, device=self.device)
        self.job_action_mask = torch.zeros(self.max_seq_len, dtype=torch.bool, device=self.device)
        self.agv_action_mask = torch.zeros(self.max_seq_len, dtype=torch.bool, device=self.device)

        self._prepare_reward_functions()

    # ─── BaseTask 接口 ────────────────────────────────────────
    def reset(self, new_job_data=None, new_agv_num=None):
        self.fjsp.reset(new_job_data, new_agv_num)
        self.num_jobs  = len(self.fjsp.jobs)
        self.num_agvs  = len(self.fjsp.agvs)

        self._trans1             = 0.0
        self._r2                 = 0.0
        self._makespan_increment = 0.0
        self._done               = False

        return self._get_obs()

    def step(self, job_action, machine_action, agv_action):
        job     = self.fjsp.jobs[job_action]
        machine = self.fjsp.machines[machine_action - 1]   # machine_action 是 1-based
        agv     = self.fjsp.agvs[agv_action - len(self.fjsp.jobs)]

        # 1. AGV 运输（仅工件不在目标机器时）
        if job.cur_pos != machine.pos:
            trans1          = self.agv_trans[agv.cur_pos][job.cur_pos]   # 空载
            trans2          = self.agv_trans[job.cur_pos][machine.pos]   # 负载
            # pre-arrival：允许 AGV 提前出发，使到达时刻恰好 >= job.end
            transport_start = max(agv.end, job.end - trans1)
            actual_arrival  = transport_start + trans1
            self._trans1    = trans1
            self._r2        = max(0.0, actual_arrival - job.end)         # 工件等 AGV 的时间
            agv.update(transport_start, trans1, trans2, job.cur_pos, machine.pos, job.idx)
            machine_start   = max(machine.end, actual_arrival + trans2)
        else:
            self._trans1  = 0.0
            self._r2      = 0.0
            machine_start = max(machine.end, job.end)

        # 2. 机器 & 工件更新
        process_time = job.get_process_time(machine_action)
        machine.update(machine_start, process_time, job.idx)
        job_end = machine_start + process_time
        job.update(job_end, machine.pos)

        # 3. Makespan 增量（奖励用）
        old_makespan             = self.fjsp.makespan
        self.fjsp.makespan       = max(old_makespan, job_end)
        self._makespan_increment = self.fjsp.makespan - old_makespan

        # 4. Done & 奖励
        self._done   = self.is_done()
        reward, info = self._compute_reward()

        # 5. 新观测
        next_state, padding_mask, job_mask, agv_mask = self._get_obs()
        return next_state, padding_mask, job_mask, agv_mask, reward, self._done, info

    def get_observations(self):
        return self.state

    def is_done(self):
        return all(j.done for j in self.fjsp.jobs)

    # ─── 两阶段决策（FJSP 特有）──────────────────────────────
    def job_step(self, job_action):
        job        = self.fjsp.jobs[job_action]
        target_pos = job.cur_pos          # AGV 要去取货的位置
        num_jobs   = len(self.fjsp.jobs)

        mid_state = self.state.clone()

        # 1. 被选工件的 flag 位置：[job_type(2), flag(1), ...]，flag 在索引 2
        mid_state[job_action][len(self.job_type)] = 1.0

        # 2. 填充每台 AGV token 中的 3 个预留槽（dist / arrival / is_same_loc）
        agv_feat_start = len(self.agv_type) + 2   # 跳过 cur_pos 和 end，= 4
        for i, agv in enumerate(self.fjsp.agvs):
            agv_row       = num_jobs + i
            dist          = self.agv_trans[agv.cur_pos][target_pos]
            agv_avail     = max(agv.end, self.fjsp.makespan)
            arrival_time  = agv_avail + dist
            is_same_loc   = 1.0 if agv.cur_pos == target_pos else 0.0
            mid_state[agv_row][agv_feat_start]     = dist         / self.norm_dist
            mid_state[agv_row][agv_feat_start + 1] = arrival_time / self.norm_time
            mid_state[agv_row][agv_feat_start + 2] = is_same_loc

        # 3. 贪心机器推荐（作为 machine_action 传给 step）
        machine_end_times = [m.end for m in self.fjsp.machines]
        machine_action, _, _, _ = self._compute_greedy_info(job, machine_end_times)

        return mid_state, self.padding_mask, self.agv_action_mask, machine_action

    # ─── 观测构建 ─────────────────────────────────────────────
    def _get_obs(self):
        machine_end_times   = [m.end for m in self.fjsp.machines]
        current_makespan    = max(self.fjsp.makespan, 1.0)
        job_tokens = [self._build_job_token(job, machine_end_times, current_makespan)
                        for job in self.fjsp.jobs]
        agv_tokens = [self._build_agv_token(agv) for agv in self.fjsp.agvs]
        all_tokens = torch.stack(job_tokens + agv_tokens)
        real_len   = all_tokens.shape[0]
        if real_len < self.max_seq_len:
            pad        = torch.zeros(self.max_seq_len - real_len, self.feat_dim, device=self.device)
            self.state = torch.cat([all_tokens, pad], dim=0)
        else:
            self.state = all_tokens[:self.max_seq_len]
        self._build_masks(len(self.fjsp.jobs), len(self.fjsp.agvs), real_len)
        return self.state, self.padding_mask, self.job_action_mask, self.agv_action_mask

    def _build_job_token(self, job, machine_end_times, current_makespan):
        if job.done:
            raw = torch.cat([self.job_type, torch.tensor([0.0], device=self.device)], dim=0)
            return self._pad_token(raw)

        mach_choice, ok_time, by_time, process_time = self._compute_greedy_info(job, machine_end_times)
        remain_time, remain_ops = job.get_remaining_stats()
        # get_remaining_stats includes the current op, so subtract current op's contribution
        # to get "future ops only" remaining time; add it back via process_time below
        total_remain = remain_time / self.norm_time

        is_continuous = 1.0 if mach_choice == job.last_pos else 0.0

        if mach_choice > 0:
            feat_mach_wait = max(0.0, self.fjsp.machines[mach_choice - 1].end - job.end)
        else:
            feat_mach_wait = 0.0

        p_idx = job.cur_process - 1
        valid_times = [t for t in job.PT[p_idx] if 0 < t < INFEASIBLE] if p_idx < len(job.PT) else []
        if valid_times:
            feat_flexibility = len(valid_times) / self.num_machines
            feat_avg_time    = sum(valid_times) / len(valid_times)
            feat_advantage   = process_time / (feat_avg_time + 1e-5)
        else:
            feat_flexibility = 0.0
            feat_avg_time    = 0.0
            feat_advantage   = 1.0

        min_agv_cost   = float('inf')
        count_free_agv = 0.0
        for agv in self.fjsp.agvs:
            cost = self.agv_trans[agv.cur_pos][job.cur_pos] + max(0.0, agv.end - job.end)
            if cost < min_agv_cost:
                min_agv_cost = cost
            if agv.end <= job.end:
                count_free_agv += 1.0
        if min_agv_cost == float('inf'):
            min_agv_cost = 0.0

        feat_urgency = (job.end + remain_time) / current_makespan

        raw = torch.cat([
            self.job_type,
            torch.tensor([
                0.0,                                         # selected flag (set in job_step)
                job.cur_process / job.all_process,           # progress ratio
                remain_ops     / self.norm_count,            # remaining ops
                mach_choice    / self.num_machines,          # greedy machine choice
                process_time   / self.norm_time,             # greedy process time
                by_time        / self.norm_dist,             # greedy transport time
                ok_time        / self.norm_time,             # greedy finish time
                total_remain,                                # total remaining time
                job.end        / self.norm_time,             # last completion time
                job.last_time  / self.norm_time,             # previous completion time
                is_continuous,                               # same location flag
                feat_mach_wait / self.norm_time,             # machine congestion
                feat_flexibility,                            # flexibility ratio
                feat_avg_time  / self.norm_time,             # average process time
                feat_advantage,                              # greedy advantage
                min_agv_cost   / self.norm_dist,             # nearest AGV cost
                count_free_agv / self.norm_agv_count,        # free AGV ratio
                feat_urgency,                                # urgency
            ], device=self.device),
        ], dim=0)
        return self._pad_token(raw)

    def _build_agv_token(self, agv):
        ref_time = max(self.fjsp.makespan, 1.0)
        raw = torch.cat([
            self.agv_type,
            torch.tensor([
                agv.cur_pos / self.num_machines,
                max(0.0, agv.end - ref_time) / self.norm_time,
                0.0,   # dist_to_target (filled in job_step)
                0.0,   # arrival_time   (filled in job_step)
                0.0,   # is_same_loc    (filled in job_step)
            ], device=self.device),
        ], dim=0)
        return self._pad_token(raw)

    def _build_masks(self, num_jobs, num_agvs, real_len):
        self.padding_mask[:]    = True
        self.padding_mask[:real_len] = False

        self.job_action_mask[:] = False
        for i, job in enumerate(self.fjsp.jobs):
            if not job.done:
                self.job_action_mask[i] = True

        self.agv_action_mask[:] = False
        for i, agv in enumerate(self.fjsp.agvs):
            if agv.enabled:
                self.agv_action_mask[num_jobs + i] = True

    def _pad_token(self, token):
        pad_len = self.feat_dim - token.shape[0]
        if pad_len > 0:
            return torch.cat([token, torch.zeros(pad_len, device=self.device)], dim=0)
        return token[:self.feat_dim]

    # ─── 观测特征辅助 ─────────────────────────────────────────
    def _compute_greedy_info(self, job, machine_end_times):
        if job.done:
            return 0, 0, 0, 0
        p_idx = job.cur_process - 1
        if p_idx >= len(job.PT):
            return 0, 0, 0, 0

        best_machine     = -1
        min_finish_time  = float('inf')
        best_process_time = 0
        best_by_time     = 0
        for i in range(self.num_machines):
            m_id      = i + 1
            proc_time = job.PT[p_idx][i]
            if 0 < proc_time < INFEASIBLE:
                path_time    = self.agv_trans[job.cur_pos][m_id]
                arrival_time = job.end + path_time
                finish_time  = max(arrival_time, machine_end_times[i]) + proc_time
                if finish_time < min_finish_time:
                    min_finish_time   = finish_time
                    best_machine      = m_id
                    best_process_time = proc_time
                    best_by_time      = path_time

        if best_machine == -1:
            return 0, 0, 0, 0
        return best_machine, min_finish_time, best_by_time, best_process_time

    # ─── 奖励 ────────────────────────────────────────────────
    def _prepare_reward_functions(self):
        self.reward_functions = []
        self.reward_names = []
        for name in dir(self):
            if name.startswith('_reward_'):
                self.reward_functions.append(getattr(self, name))
                self.reward_names.append(name[len('_reward_'):])

    def _compute_reward(self):
        total = 0.0
        info = {}
        for func, name in zip(self.reward_functions, self.reward_names):
            rew = func()
            info[name] = rew
            total += rew
        info['abs_makespan'] = self.fjsp.makespan
        return total, info

    def _reward_transport(self):
        return -(self._trans1 * self.cfg.reward.transport_coef)

    def _reward_agv_wait(self):
        return -(self._r2 * self.cfg.reward.wait_coef)

    def _reward_makespan(self):
        return -(self._makespan_increment * self.cfg.reward.makespan_coef)

    def _reward_completion(self):
        if not self._done:
            return 0.0
        return self.cfg.reward.completion_bonus

    def _reward_balance(self):
        if not self._done:
            return 0.0
        agv_counts = [len([x for x in a.on if x is not None])
                        for a in self.fjsp.agvs]
        return -(np.std(agv_counts) * self.cfg.reward.balance_coef)



