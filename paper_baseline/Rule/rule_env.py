import gym
from gym import spaces
import numpy as np
import config
import torch
from FJSP import FJSP

class EnvWorkShop(gym.Env):
    def __init__(self, machine_num=8):
        self.num_machines = machine_num
        self.FJSP = FJSP(self.num_machines)
        self.max_seq_len = 50
        self.feat_dim = 24  # 根据下面 Token 的实际长度调整
        self.observation_space = spaces.Box(low=-1, high=1000,
                                            shape=(self.max_seq_len, self.feat_dim),
                                            dtype=np.float32)
        # 动作空间: [选哪个工件, 选哪个机器, 选哪个AGV]
        # 注意: 这里上限设得比较大以适应最大规模，实际 Mask 会限制有效动作
        self.action_space = spaces.MultiDiscrete([self.max_seq_len, self.num_machines + 1, self.max_seq_len])
        self.job_state = []
        self.agv_state = []
        # === 缓存变量 (用于 job_step 复用) ===
        self.state = None
        self.padding_mask = None
        self.job_action_mask = None
        self.agv_action_mask = None
        self.last_makespan = 0
        # === 🟢 定义归一化常数 ===
        self.NORM_TIME = 200.0
        self.NORM_DIST = 100.0
        self.NORM_COUNT = 20.0

    def pad_sequence_dim(self, raw_state):
        """
        [新增辅助函数] 对 Token 序列的行数进行填充
        Input: (Current_N, 16)
        Output: (MAX_SEQ_LEN, 16)
        """
        curr_len = raw_state.shape[0]
        if curr_len >= self.max_seq_len:
            return raw_state[:self.max_seq_len, :]
        pad_len = self.max_seq_len - curr_len
        # 生成全 0 的 padding
        padding = torch.zeros((pad_len, self.feat_dim), dtype=raw_state.dtype)
        return torch.cat([raw_state, padding], dim=0)

    def calculate_greedy_info(self, job, machine_end_times):
        """
        基于当前 Job 对象内部的 PT 矩阵实时计算贪心推荐。
        """
        # 1. 获取当前工序的索引 (因为 job.cur_process 是 1-based，列表是 0-based)
        p_idx = job.cur_process - 1
        # 安全检查：防止工件已做完导致的索引越界
        if p_idx >= len(job.PT):
            return 0, 0, 0, 0

        # 初始化最小值
        best_machine = -1  # 记录最佳机器ID (1-based)
        min_finish_time = 999999  # 记录最小完工时间
        best_process_time = 0  # 记录最佳机器的加工时间
        best_by_time = 0  # 记录最佳路径的搬运时间
        # 2. 遍历所有机器 (假设有 8 台，索引 0~7)
        for i in range(self.num_machines):
            m_id_1based = i + 1  # 机器的物理ID (1~8)

            # --- 关键修改点：直接从 job 对象里拿 PT，不再查全局表 ---
            # 对应你原来的: PT_this_job[i]
            proc_time = job.PT[p_idx][i]

            # 3. 判断机器是否可行 (参考你原来的: if PT_this_job[i] != 0)
            if proc_time > 0 and proc_time < 9999:
                # --- 计算搬运时间 ---
                # 对应你原来的: path_time = agv_trans[job_cur_pos][i+1]
                # 假设 config.agv_trans 是固定的距离矩阵 (8x8 或者更大)
                path_time = config.agv_trans[job.cur_pos][m_id_1based]
                # --- 计算开始和完工时间 ---
                # 对应你原来的: max((job_end_time+path_time), machine_end_time_list[i])
                # 注意：machine_end_times[i] 对应第 i 台机器
                arrival_time = job.end + path_time
                start_process_time = max(arrival_time, machine_end_times[i])
                finish_time = start_process_time + proc_time
                # --- 贪心比较 (找最小完工时间) ---
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_machine = m_id_1based
                    best_process_time = proc_time
                    best_by_time = path_time
        # 如果找不到可行机器 (极端情况)，返回 0
        if best_machine == -1:
            return 0, 0, 0, 0
        return best_machine, min_finish_time, best_by_time, best_process_time

    def calculate_remaining_time(self, job, start_op_idx):
        """
        移植自你原来的 calc_remaining_time。
        计算从 start_op_idx (0-based) 开始到最后一道工序的剩余平均时间。
        """
        job_average_time = []
        # 直接使用 job.PT，不需要 jobid
        for i in range(len(job.PT)):
            # 只有当工序索引 >= 指定开始位置时才计算
            if i >= start_op_idx:
                # 你的原始逻辑：过滤掉 0 (不可行机器)
                filtered_data = [value for value in job.PT[i] if value != 0]

                # 防止整行都是 0 (虽然生成器应该避免这种情况，但为了健壮性)
                if len(filtered_data) > 0:
                    average = np.mean(filtered_data)
                    job_average_time.append(average)
                else:
                    job_average_time.append(0)

        tmp = np.sum(job_average_time)
        return tmp

    def reset(self, new_job_data=None, new_agv_num=None):
        """
        核心: 接收动态数据进行重置
        """
        # 调用 FJSP 的高级 reset
        self.FJSP.reset(new_job_data, new_agv_num)
        # 更新当前环境的数量变量
        self.num_jobs = len(self.FJSP.Jobs)
        self.num_agv = len(self.FJSP.AGVs)
        self.last_makespan = 0
        return self._get_state()

    def _get_state(self):
        machine_end_time_list = [m.end for m in self.FJSP.Machines]
        # 当前最大 Makespan (用于归一化紧迫度)
        current_max_makespan = max(self.FJSP.max, 1.0)
        # 初始工件生成
        job_token_list = []
        for i, job in enumerate(self.FJSP.Jobs):
            if job.done:
                # 完成的工件填 0
                raw_token = torch.cat([
                    config.job_type,
                    torch.tensor([0.0]) # <--- [Selected Flag] 默认为 0
                ], dim=0)
                job_token_temp = config.pad_to_length(raw_token, target_length=self.feat_dim)
            else:
                # 1. 贪心计算当前工序的具体数值
                mach_choice, ok_time, by_time, process_time = self.calculate_greedy_info(job, machine_end_time_list)
                remain_need_time = self.calculate_remaining_time(job, start_op_idx=job.cur_process)
                total_remain_time_feat = remain_need_time + process_time
                remain_ops = job.all_process - job.cur_process + 1
                is_continuous = 1 if mach_choice == job.last_pos else 0
                # A. 机器拥堵度
                if mach_choice > 0:
                    target_m = self.FJSP.Machines[mach_choice - 1]
                    # 机器还要忙多久？(相对于工件当前的 ready time)
                    feat_mach_wait = max(0, target_m.end - job.end)
                else:
                    feat_mach_wait = 0.0
                # B. 统计特征 (柔性/难度/优势)
                # 获取当前工序所有可行的加工时间
                p_idx = job.cur_process - 1
                if p_idx < len(job.PT):
                    valid_times = [t for t in job.PT[p_idx] if t > 0 and t < 9999]
                else:
                    valid_times = []

                if len(valid_times) > 0:
                    feat_flexibility = len(valid_times) / self.num_machines  # 归一化柔性
                    feat_avg_time = sum(valid_times) / len(valid_times)
                    feat_advantage = process_time / (feat_avg_time + 1e-5)
                else:
                    feat_flexibility = 0.0
                    feat_avg_time = 0.0
                    feat_advantage = 1.0
                # C. 最近 AGV 距离 (物流潜力)
                min_agv_cost = 99999.0
                count_free_agv = 0.0
                for agv in self.FJSP.AGVs:
                    # 距离 + AGV剩余忙碌时间
                    d = config.agv_trans[agv.cur_pos][job.cur_pos] + max(0, agv.end - job.end)
                    if d < min_agv_cost:
                        min_agv_cost = d
                    if agv.end <= job.end:  # 简单判断空闲
                        count_free_agv += 1.0
                # D. 紧迫度
                est_finish = job.end + process_time + remain_need_time
                feat_urgency = est_finish / current_max_makespan

                raw_token = torch.cat([
                    config.job_type,  # One-hot Header
                    torch.tensor([0]),# Flag
                    # 状态
                    torch.tensor([job.cur_process / job.all_process]),
                    torch.tensor([remain_ops / self.NORM_COUNT]),
                    # 贪心提案
                    torch.tensor([mach_choice / self.num_machines]),
                    torch.tensor([process_time / self.NORM_TIME]),
                    torch.tensor([by_time / self.NORM_DIST]),
                    torch.tensor([ok_time / self.NORM_TIME]),
                    # 全局视野 (这里替换为你原来的逻辑)
                    torch.tensor([total_remain_time_feat / self.NORM_TIME]),  # <--- 修改了这里
                    torch.tensor([job.end / self.NORM_TIME]),
                    torch.tensor([job.last_time / self.NORM_TIME]),
                    # 标志
                    torch.tensor([is_continuous]),
                    # ✅ 新增特征 (7维)
                    torch.tensor([feat_mach_wait / self.NORM_TIME]),  # 机器等待
                    torch.tensor([feat_flexibility]),  # 柔性
                    torch.tensor([feat_avg_time / self.NORM_TIME]),  # 平均难度
                    torch.tensor([feat_advantage]),  # 贪心优势
                    torch.tensor([min_agv_cost / self.NORM_DIST]),  # 最近AGV代价
                    torch.tensor([count_free_agv / 5.0]),  # 空闲车数 (假设最多5台)
                    torch.tensor([feat_urgency]),  # 紧迫度
                ], dim=0)
                job_token_temp = config.pad_to_length(raw_token, target_length=self.feat_dim)
            job_token_list.append(job_token_temp)
        agv_token_list = []
        # 参考时间 (用于计算相对忙碌时间)
        ref_time = max(self.last_makespan, 1.0)
        for agv in self.FJSP.AGVs:
            feat_time_until_free = max(0, agv.end - ref_time)
            raw_agv_token = torch.cat([
                config.agv_type,  # Header
                # AGV 特征
                torch.tensor([agv.cur_pos / self.num_machines]),
                torch.tensor([agv.end / self.NORM_TIME]),
                # ✅ 预留位置给 Target Features (先填 0)
                # 我们预留 3 个位置: [Dist, Arrival, Same_Loc]
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                torch.tensor([0.0]),
            ], dim=0)
            agv_token_temp = config.pad_to_length(raw_agv_token, target_length=self.feat_dim)
            agv_token_list.append(agv_token_temp)
        # 拼接
        self.job_state = torch.stack(job_token_list)  # [Num_Jobs, 16]
        self.agv_state = torch.stack(agv_token_list)  # [Num_AGVs, 16]
        raw_state = torch.cat([self.job_state, self.agv_state], dim=0) # [Num_Real, 16]
        # 保存真实长度，用于生成 Mask
        real_len = raw_state.shape[0]
        num_jobs = len(self.FJSP.Jobs)
        num_agvs = len(self.FJSP.AGVs)
        # 核心：调用 Padding 函数，补全到 (50, 16)
        self.state = self.pad_sequence_dim(raw_state)
        # True = Padding (垃圾数据), False = Real (有效数据)
        self.padding_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        self.padding_mask[:real_len] = False
        # B. Job Action Mask (给工件智能体输出层用)
        # True = 可选, False = 不可选 (AGV/完成/Padding)
        self.job_action_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        for i, job in enumerate(self.FJSP.Jobs):
            if not job.done:
                self.job_action_mask[i] = True  # 只有未完成的工件是 True
        # C. AGV Action Mask (给 AGV 智能体输出层用)
        # True = 可选, False = 不可选 (工件/Padding)
        self.agv_action_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        # 只有 AGV 所在的那一段索引是 True
        self.agv_action_mask[num_jobs: num_jobs + num_agvs] = True
        # 返回 4 个值
        return self.state, self.padding_mask, self.job_action_mask, self.agv_action_mask

    def job_step(self,job_action):
        real_job_idx = job_action - 1
        job = self.FJSP.Jobs[real_job_idx]
        target_pos = job.cur_pos  # 目标位置
        job_ready_time = job.end
        mid_state = self.state.clone()
        l_head = len(config.job_type)
        flag_index = l_head
        # 将被选中的工件 Flag 设为 1
        mid_state[real_job_idx][flag_index] = 1.0
        # --- 动态更新 AGV Token (让 AGV 看到目标) ---
        num_jobs = len(self.FJSP.Jobs)
        agv_feat_start_idx = len(config.agv_type) + 2  # 跳过 Pos, End

        for i, agv in enumerate(self.FJSP.AGVs):
            agv_row_idx = num_jobs + i

            # A. 距离
            dist_to_target = config.agv_trans[agv.cur_pos][target_pos]

            # B. 预计到达时间
            # AGV 何时能动？ max(agv.end, current_makespan)
            # 这里简单用 job.end 近似 current_time，或者维护一个 env.current_time
            agv_avail = max(agv.end, self.last_makespan)
            arrival_time = agv_avail + dist_to_target

            # C. 是否顺路 (同位置)
            is_same_loc = 1.0 if agv.cur_pos == target_pos else 0.0

            # 写入 Tensor (直接覆盖 _get_state 里预留的 0.0)
            mid_state[agv_row_idx][agv_feat_start_idx] = dist_to_target / self.NORM_DIST
            mid_state[agv_row_idx][agv_feat_start_idx + 1] = arrival_time / self.NORM_TIME
            mid_state[agv_row_idx][agv_feat_start_idx + 2] = is_same_loc

        # 2. 计算机器贪心
        machine_end_time_list = [m.end for m in self.FJSP.Machines]
        machine_action, _, _, _ = self.calculate_greedy_info(job, machine_end_time_list)

        # 3. 返回中间状态 + 之前算好的 Mask
        # AGV 智能体需要: mid_state (看 Flag), padding_mask (做 Attention), agv_action_mask (选动作)
        return mid_state, self.padding_mask, self.agv_action_mask, machine_action


    def step(self,job_action,machine_action,agv_action):
        """
        执行完整一步
        """
        job = self.FJSP.Jobs[job_action-1]
        machine = self.FJSP.Machines[machine_action-1]
        agv = self.FJSP.AGVs[agv_action-1]

        # =========== ✅ 修正后的代码 (rule_env.py) ===========
        # 1. AGV 更新
        if job.cur_pos != machine.pos:
            trans1 = config.agv_trans[agv.cur_pos][job.cur_pos] # 空载时间
            trans2 = config.agv_trans[job.cur_pos][machine.pos] # 负载时间
            
            # 核心修正：倒推出发时间
            # 我们希望 AGV 到达时间 (start + trans1) >= job.end
            # 所以 start >= job.end - trans1
            # 同时 AGV 必须空闲: start >= agv.end
            transport_start = max(agv.end, job.end)
            # transport_start = max(agv.end, job.end - trans1)
            
            # 计算等待时间 (AGV到了，但是货还没好)
            # 实际到达时刻 = transport_start + trans1
            actual_arrival_time = transport_start + trans1
            r2 = actual_arrival_time - job.end 
            
            # 更新 AGV
            agv.update(transport_start, trans1, trans2, job.cur_pos, machine.pos, job_action)
        else:
            trans1, trans2, r2 = 0, 0, 0
            # 如果原地不动，AGV 到达时间就是当前时间（虚拟的）
            actual_arrival_time = max(agv.end, job.end)

        # 2. 机器更新
        if job.cur_pos == machine.pos:
            machine_start_time = max(machine.end, job.end)
            r3 = 0
        else:
            # 机器必须等：1. 机器空闲  2. AGV 把货送到了
            # 送到时刻 = 实际到达时刻 (接货) + trans2 (运货)
            agv_delivery_time = actual_arrival_time + trans2
            machine_start_time = max(machine.end, agv_delivery_time)
            r3 = machine_start_time - agv_delivery_time

        process_time = job.get_process_time(machine_action)
        machine.update(machine_start_time, process_time, job_action)
        # 3. 工件更新
        job_end = machine_start_time + process_time
        job.update(job_end, machine.pos)
        rs1 = - (trans1 * 0.01)
        rs2 = - (r2 * 0.02)
        rs3 = - (r3 * 0.02)
        # Makespan 增量惩罚
        new_makespan = max(self.FJSP.max, job_end)
        makespan_increment = new_makespan - self.last_makespan
        self.FJSP.max = new_makespan
        self.last_makespan = new_makespan

        r_makespan = - (makespan_increment * 1.0)
        # PPO 优化只看这个总分
        reward = rs1 + rs2 + r_makespan
        # === 重新生成状态 (会自动带上 Padding 和 Mask) ===
        next_state, padding_mask, job_mask, agv_mask = self._get_state()
        # === 判断 Done ===
        done = True
        for j in self.FJSP.Jobs:
            if not j.done:
                done = False
                break
        # 终局奖励
        rf1, rf2 = 0, 0
        if done:
            rf1 = 50.0  # 完成奖励
            # 负载均衡奖励
            agv_counts = [len([x for x in a.on if x is not None]) for a in self.FJSP.AGVs]
            AMiss = np.std(agv_counts)  # 用标准差衡量不均衡
            rf2 = - (AMiss * 0.5)
            reward += (rf1 + rf2)
        info = {
            'rs1': rs1,
            'rs2': rs2,
            'r_makespan': r_makespan,
            'rf1': rf1,
            'rf2': rf2,
            'makespan': self.FJSP.max  # 记录一下当前的 Makespan 方便画图
        }

        # 这里返回所有 Mask，PPO 训练循环里自己去解包
        return next_state, padding_mask, job_mask, agv_mask, reward, done, info