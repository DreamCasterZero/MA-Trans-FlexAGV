import random
import copy
import numpy as np
import matplotlib.pyplot as plt

class Machine:
    def __init__(self, idx):
        self.idx = idx  # 机器编号
        self.using_time = []  # 使用记录（[start, end] 时间段）
        self.on = []  # 对应时间段执行的是哪个工序/作业
        self.end = 0  # 当前机器忙到的时间（可看作机器的“可用时间”）

    def update(self,s,pt,_on):
        e = s + pt  # 计算结束时间
        self.using_time.append([s, e])  # 记录时间段
        self.on.append(_on)  # 记录执行的任务是谁
        self.end = e  # 更新机器的最晚可用时间

class Job:
    def __init__(self,idx,PT,MT,L_U):
        self.idx = idx  # 作业编号
        self.PT = PT  # 加工时间列表：每道工序在各机器上的加工时间
        self.MT = MT  # 每道工序可选机器集合
        self.cur_site = L_U  # 当前所处位置（初始为原料区）
        self.L_U = L_U  # 固定的原始起始位置（原料区），用于参考
        self.end = 0  # 当前工序的完成时间（初始化为 0）
        self.cur_op = 1  # 当前是第几道工序（从 0 开始）

    def get_info(self):
        return self.end, self.cur_site, self.PT[self.cur_op-1]

    def update(self,e,m):
        self.end=e
        self.cur_op += 1
        self.cur_site = m

class AGV:
    def __init__(self,idx,L_U):
        self.idx = idx  # AGV编号
        self.cur_site = L_U  # 当前所处位置（初始为原料区）
        self.using_time = []  # 使用记录：每段运输的时间区间 [start, end]
        self.on = []  # 每段运输中运输的任务信息
        self._to = []  # 每段运输的目的地
        self.end = 0  # 当前AGV的可用时间（上次任务结束时间）

    def ST(self,s,t1,t2):
        start=max(s,self.end+t1)
        return start-t1,start+t2

    #当前位置 → J_site（取货） → J_m（送货） → 结束
    def update(self,s,trans1,trans2,J_site,J_m,_on=None):
        self.using_time.append([s,s+trans1])
        self.using_time.append([s + trans1, s+trans1 + trans2])
        self.on.append(None)
        self.on.append(_on)
        self._to.extend([J_site,J_m])
        self.end=s+trans1+trans2
        self.cur_site=J_m

class RJSP:
    def __init__(self,n,m,agv_num,PT,MT,TT,L_U):
        self.n,self.m,self.agv_num=n,m,agv_num
        self.PT=PT
        self.MT=MT
        self.TT=TT
        self.L_U=L_U
        self.C_max = 0  # 当前最优调度总时间（Makespan）

    def reset(self):
        self.Jobs = []
        for i in range(self.n):
            Ji = Job(i+1, self.PT[i], self.MT[i], self.L_U)
            self.Jobs.append(Ji)
        self.Machines = []
        for j in range(self.m+1): #0号是原料区占位
            Mi = Machine(j)
            self.Machines.append(Mi)
        self.AGVs = []
        for k in range(self.agv_num):
            agv = AGV(k+1, self.L_U)
            self.AGVs.append(agv)
        self.C_max = 0

    def VAA_decode(self, Jobi, Machi):
        Ji = self.Jobs[Jobi-1]
        # print(Ji.idx)
        Mi = self.Machines[Machi]
        # print(Mi.idx)
        J_end, J_site, mach_time_list = Ji.get_info()
        mach_time = mach_time_list[Machi - 1]
        # print(mach_time)
        best_agv = None
        min_tf = 99999
        best_s, best_e, t1, t2 = None, None, None, None
        for agv in self.AGVs:
            trans1 = self.TT[agv.cur_site][J_site]
            # print(trans1)
            trans2 = self.TT[J_site][Machi]
            # print(trans2)
            start, end = agv.ST(J_end, trans1, trans2)
            if end < min_tf:
                best_s, best_e, t1, t2 = start, end, trans1, trans2
                best_agv = agv
                min_tf = best_e
        best_agv.update(best_s, t1, t2, J_site, Machi, Ji.idx)
        start = max(best_e, Mi.end)
        Mi.update(start, mach_time, Ji.idx)
        Jend = start + mach_time
        Ji.update(Jend, Machi)
        if Jend > self.C_max:
            self.C_max = Jend
        return self.C_max


class MA:  # 🔴 改名为 MA
    def __init__(self, n, m, agv_num, PT, MT, agv_trans, pop_size=100, gene_size=1000, pc=0.8, pm=0.1, N_elite=10):
        self.N_elite = N_elite
        self.rjsp = RJSP(n, m, agv_num, PT, MT, agv_trans, 0)
        self.Pop_size = pop_size
        self.gene_size = gene_size
        self.pc = pc
        self.pm = pm
        self.Fit = []

        self.job_op_offset = []
        offset = 0
        self.total_ops = 0
        for job_pt in self.rjsp.PT:
            self.job_op_offset.append(offset)
            ops_count = len(job_pt)
            offset += ops_count
            self.total_ops += ops_count

        self.best_cmax = float('inf')
        self.best_os = None
        self.best_ms = None

    def initial_population(self):
        self.OS_Pop = []
        self.MS_Pop = []
        base_os = []
        for job_id, job_pt in enumerate(self.rjsp.PT):
            base_os.extend([job_id + 1] * len(job_pt))

        for _ in range(self.Pop_size):
            os_chrom = copy.deepcopy(base_os)
            random.shuffle(os_chrom)
            self.OS_Pop.append(os_chrom)
            ms_chrom = []
            for job_idx, job_mt in enumerate(self.rjsp.MT):
                for op_mt in job_mt:
                    ms_chrom.append(random.choice(list(op_mt)))
            self.MS_Pop.append(ms_chrom)

    def POX(self, p1, p2):
        jobs = list(set(p1))
        job_subset = random.sample(jobs, len(jobs) // 2)
        c1 = [0] * len(p1);
        c2 = [0] * len(p2)
        for i, job in enumerate(p1):
            if job in job_subset: c1[i] = job
        for i, job in enumerate(p2):
            if job in job_subset: c2[i] = job
        p2_idx = 0
        for i in range(len(c1)):
            if c1[i] == 0:
                while p2[p2_idx] in job_subset: p2_idx += 1
                c1[i] = p2[p2_idx];
                p2_idx += 1
        p1_idx = 0
        for i in range(len(c2)):
            if c2[i] == 0:
                while p1[p1_idx] in job_subset: p1_idx += 1
                c2[i] = p1[p1_idx];
                p1_idx += 1
        return c1, c2

    def UniformCrossover(self, ms1, ms2):
        mask = [random.randint(0, 1) for _ in range(len(ms1))]
        child1 = [];
        child2 = []
        for i, m in enumerate(mask):
            if m == 0:
                child1.append(ms1[i]);
                child2.append(ms2[i])
            else:
                child1.append(ms2[i]);
                child2.append(ms1[i])
        return child1, child2

    def decode(self, os_chrom, ms_chrom):
        self.rjsp.reset()
        job_op_counter = [0] * self.rjsp.n
        for job_id_plus1 in os_chrom:
            job_idx = job_id_plus1 - 1
            op_idx = job_op_counter[job_idx]
            ms_index = self.job_op_offset[job_idx] + op_idx
            machine_id = ms_chrom[ms_index]
            self.rjsp.VAA_decode(job_id_plus1, machine_id)
            job_op_counter[job_idx] += 1
        return self.rjsp.C_max

    def fitness(self):
        self.Fit = []
        for i in range(self.Pop_size):
            cmax = self.decode(self.OS_Pop[i], self.MS_Pop[i])
            self.Fit.append(cmax)

    def Select(self):
        best_idx = np.argmin(self.Fit)
        new_OS, new_MS = [], []
        # 精英保留
        for _ in range(self.N_elite):
            new_OS.append(copy.deepcopy(self.OS_Pop[best_idx]))
            new_MS.append(copy.deepcopy(self.MS_Pop[best_idx]))
        # 锦标赛
        while len(new_OS) < self.Pop_size:
            a, b = random.sample(range(self.Pop_size), 2)
            winner = a if self.Fit[a] < self.Fit[b] else b
            new_OS.append(copy.deepcopy(self.OS_Pop[winner]))
            new_MS.append(copy.deepcopy(self.MS_Pop[winner]))
        self.OS_Pop = new_OS
        self.MS_Pop = new_MS

    def crossover_operator(self):
        indices = list(range(self.Pop_size))
        random.shuffle(indices)
        new_OS, new_MS = [], []
        # 注意：精英通常不参与破坏性交叉，这里为了简单全盘交叉，但保留了Select中的精英副本在下一代
        for i in range(0, self.Pop_size, 2):
            idx1, idx2 = indices[i], indices[i + 1] if i + 1 < self.Pop_size else indices[0]
            os1, os2 = self.OS_Pop[idx1], self.OS_Pop[idx2]
            ms1, ms2 = self.MS_Pop[idx1], self.MS_Pop[idx2]
            if random.random() < self.pc:
                c_os1, c_os2 = self.POX(os1, os2)
                c_ms1, c_ms2 = self.UniformCrossover(ms1, ms2)
                new_OS.extend([c_os1, c_os2])
                new_MS.extend([c_ms1, c_ms2])
            else:
                new_OS.extend([os1, os2])
                new_MS.extend([ms1, ms2])
        self.OS_Pop = new_OS[:self.Pop_size]
        self.MS_Pop = new_MS[:self.Pop_size]

    def mutation_operator(self):
        for i in range(self.Pop_size):
            if random.random() < self.pm:
                idx1, idx2 = random.sample(range(len(self.OS_Pop[i])), 2)
                self.OS_Pop[i][idx1], self.OS_Pop[i][idx2] = self.OS_Pop[i][idx2], self.OS_Pop[i][idx1]
            if random.random() < self.pm:
                m_idx = random.randint(0, len(self.MS_Pop[i]) - 1)
                # 简单反查逻辑
                target_job = -1;
                target_op = -1
                for job_idx, offset in enumerate(self.job_op_offset):
                    next_offset = self.job_op_offset[job_idx + 1] if job_idx + 1 < len(
                        self.job_op_offset) else self.total_ops
                    if offset <= m_idx < next_offset:
                        target_job = job_idx;
                        target_op = m_idx - offset;
                        break
                available_machines = list(self.rjsp.MT[target_job][target_op])
                if len(available_machines) > 1:
                    current_machine = self.MS_Pop[i][m_idx]
                    choices = [m for m in available_machines if m != current_machine]
                    if choices: self.MS_Pop[i][m_idx] = random.choice(choices)

    def local_search(self, elite_ratio=0.1, max_search_steps=10):
        """
        升级版局部搜索：同时尝试 OS 交换 和 MS 变异
        """
        elite_count = int(self.Pop_size * elite_ratio)
        sorted_indices = np.argsort(self.Fit)
        elites = sorted_indices[:elite_count]

        for idx in elites:
            current_os = self.OS_Pop[idx]
            current_ms = self.MS_Pop[idx]
            current_fit = self.Fit[idx]

            for _ in range(max_search_steps):
                # --- 策略 1: 邻域搜索 OS (交换两个工序) ---
                temp_os = copy.deepcopy(current_os)
                temp_ms = copy.deepcopy(current_ms) # 保持 MS 不变
                
                # 50% 概率搜 OS，50% 概率搜 MS
                if random.random() < 0.5:
                    p1, p2 = random.sample(range(len(temp_os)), 2)
                    temp_os[p1], temp_os[p2] = temp_os[p2], temp_os[p1]
                else:
                    # --- 策略 2: 邻域搜索 MS (单点变异机器) ---
                    # 随机选一个位置改变机器
                    m_pos = random.randint(0, len(temp_ms) - 1)
                    # 反查这是哪个 Job 的哪个 Op (复用 mutation 里的逻辑)
                    target_job = -1; target_op = -1
                    for job_idx, offset in enumerate(self.job_op_offset):
                        next_offset = self.job_op_offset[job_idx + 1] if job_idx + 1 < len(self.job_op_offset) else self.total_ops
                        if offset <= m_pos < next_offset:
                            target_job = job_idx; target_op = m_pos - offset; break
                    
                    available_machines = list(self.rjsp.MT[target_job][target_op])
                    if len(available_machines) > 1:
                        current_machine = temp_ms[m_pos]
                        choices = [m for m in available_machines if m != current_machine]
                        if choices: temp_ms[m_pos] = random.choice(choices)

                # 评估
                new_fit = self.decode(temp_os, temp_ms)

                # 贪婪接受
                if new_fit < current_fit:
                    self.OS_Pop[idx] = temp_os
                    self.MS_Pop[idx] = temp_ms
                    self.Fit[idx] = new_fit
                    current_fit = new_fit

    def main(self):
        Fit_best = []
        self.initial_population()
        self.fitness()

        Best_Global_Fit = min(self.Fit)
        best_idx = np.argmin(self.Fit)
        self.best_os = copy.deepcopy(self.OS_Pop[best_idx])
        self.best_ms = copy.deepcopy(self.MS_Pop[best_idx])
        self.best_cmax = Best_Global_Fit
        # print(f"Initial Best: {Best_Global_Fit}")

        for step in range(self.gene_size):
            self.Select()
            self.crossover_operator()
            self.mutation_operator()
            self.fitness()  # 更新 fitness

            # 🔴【核心】插入局部搜索
            # 在交叉变异后，对优秀个体再挖一挖
            self.local_search(elite_ratio=0.1, max_search_steps=10)

            # 更新全局最优
            curr_best = min(self.Fit)
            if curr_best < Best_Global_Fit:
                Best_Global_Fit = curr_best
                best_idx = np.argmin(self.Fit)
                self.best_os = copy.deepcopy(self.OS_Pop[best_idx])
                self.best_ms = copy.deepcopy(self.MS_Pop[best_idx])
                self.best_cmax = Best_Global_Fit

            Fit_best.append(Best_Global_Fit)
            # if step % 50 == 0:
                # print(f"Gen {step}: {Best_Global_Fit}")
        #
        # plt.plot(Fit_best)
        # plt.xlabel("Generation")
        # plt.ylabel("Makespan")
        # plt.title("MA (Memetic Algorithm) Training Curve")
        # plt.show()
        return Best_Global_Fit
