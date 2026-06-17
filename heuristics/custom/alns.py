from config import PT, pt_to_mt
import copy
import random
import math
import matplotlib.pyplot as plt

class Machine:
    def __init__(self, idx):
        self.idx = idx
        self.using_time = []
        self.on = []
        self.end = 0

    def update(self, start, process_time, _on):
        end = start + process_time
        self.using_time.append([start, end])
        self.on.append(_on)
        self.end = end

class Job:
    def __init__(self, idx, PT, MT):
        self.idx = idx
        self.PT = PT
        self.MT = MT
        self.end = 0
        self.current_operation = 1
    
    def update(self, end):
        self.end = end
        self.current_operation += 1

    def get_info(self):
        return self.end, self.PT[self.current_operation - 1]

class FJSP:
    def __init__(self, n, m, PT, MT):
        self.n = n
        self.m = m
        self.PT = PT
        self.MT = MT
        self.makespan = 0

    def reset(self):
        self.Jobs = []
        for i in range(self.n):
            Ji = Job(i+1, self.PT[i], self.MT[i])
            self.Jobs.append(Ji)
        self.Machines = []
        for j in range(self.m):
            Mj = Machine(j+1)
            self.Machines.append(Mj)
        self.makespan = 0

    def decode(self, Jobi, Machinej):
        Ji = self.Jobs[Jobi-1]
        Mj = self.Machines[Machinej-1]
        J_end, mach_time_list = Ji.get_info()
        mach_time = mach_time_list[Machinej-1]
        start = max(J_end, Mj.end)
        end = start + mach_time
        Mj.update(start, mach_time, Ji.idx)
        Ji.update(end)
        if end > self.makespan:
            self.makespan = end
        return self.makespan

class ALNS:
    def __init__(self, n, m, PT, MT, max_iter=2000, T_init=500, T_min=1, alpha=0.97,destroy_ratio=0.3, weight_decay=0.8):
        self.fjsp = FJSP(n, m, PT, MT)
        self.max_iter = max_iter
        self.T_init = T_init        # 初始温度（接受准则借用SA）
        self.T_min = T_min
        self.alpha = alpha          # 冷却系数
        self.destroy_ratio = destroy_ratio  # 每次破坏的工件比例
        self.weight_decay = weight_decay    # 权重衰减系数
        self.job_op_offset = []
        offset = 0
        self.total_ops = 0
        for job_pt in PT:
            self.job_op_offset.append(offset)
            offset += len(job_pt)
        self.total_ops = offset
        # 算子权重初始化（每个算子初始权重相等）
        self.destroy_names = ["random_destroy", "worst_destroy"]
        self.repair_names = ["greedy_repair", "random_repair"]
        self.destroy_weights = [1.0, 1.0]
        self.repair_weights = [1.0, 1.0]
        # 算子得分记录
        self.destroy_scores = [0.0, 0.0]
        self.repair_scores = [0.0, 0.0]
        self.destroy_counts = [0, 0]
        self.repair_counts = [0, 0]

    def generate_initial_solution(self):
        base_os = []
        for job_id, job_pt in enumerate(self.fjsp.PT):
            base_os.extend([job_id + 1] * len(job_pt))
        os_chrom = copy.deepcopy(base_os)
        random.shuffle(os_chrom)
        ms_chrom = []
        for job_idx, job_mt in enumerate(self.fjsp.MT):
            for op_mt in job_mt:
                ms_chrom.append(random.choice(list(op_mt)))
        return os_chrom, ms_chrom

    def decode(self, os_chrom, ms_chrom):
        self.fjsp.reset()
        job_op_counter = [0] * self.fjsp.n
        for job_id_plus1 in os_chrom:
            job_idx = job_id_plus1 - 1
            op_idx = job_op_counter[job_idx]
            ms_index = self.job_op_offset[job_idx] + op_idx
            machine_id = ms_chrom[ms_index]
            self.fjsp.decode(job_id_plus1, machine_id)
            job_op_counter[job_idx] += 1
        return self.fjsp.makespan

    def random_destroy(self, os_chrom, ms_chrom):
        num_destroy = max(1, int(self.fjsp.n * self.destroy_ratio))
        all_jobs = list(range(1, self.fjsp.n + 1))
        removed_jobs = set(random.sample(all_jobs, num_destroy))
        # 从OS中移除被选中工件的工序，保留其余工序的顺序
        remaining_os = [gene for gene in os_chrom if gene not in removed_jobs]
        return remaining_os, removed_jobs

    def worst_destroy(self, os_chrom, ms_chrom):
        """最差破坏：移除完工时间最晚的几个工件（它们最可能是瓶颈）。"""
        num_destroy = max(1, int(self.fjsp.n * self.destroy_ratio))
        # 先解码一次，获取每个工件的完工时间
        self.decode(os_chrom, ms_chrom)
        job_end_times = []
        for job in self.fjsp.Jobs:
            job_end_times.append((job.idx, job.end))
        # 按完工时间降序排列，取最差的几个
        job_end_times.sort(key=lambda x: x[1], reverse=True)
        removed_jobs = set(j for j, _ in job_end_times[:num_destroy])
        remaining_os = [gene for gene in os_chrom if gene not in removed_jobs]
        return remaining_os, removed_jobs

    def greedy_repair(self, remaining_os, removed_jobs, ms_chrom):
        """贪心修复：被移除的工件，逐个尝试插入OS的每个位置，选makespan最小的。"""
        new_os = copy.deepcopy(remaining_os)
        new_ms = copy.deepcopy(ms_chrom)
        # 对每个被移除的工件，按工序数依次插入
        removed_list = list(removed_jobs)
        random.shuffle(removed_list)
        for job_id in removed_list:
            job_idx = job_id - 1
            num_ops = len(self.fjsp.PT[job_idx])
            for op_idx in range(num_ops):
                best_pos = len(new_os)  # 默认插到末尾
                best_machine = None
                best_cost = float('inf')
                # MS中该工序对应位置
                ms_index = self.job_op_offset[job_idx] + op_idx
                eligible = list(self.fjsp.MT[job_idx][op_idx])
                # 尝试每台可选机器
                for machine in eligible:
                    new_ms[ms_index] = machine
                    # 尝试插入OS的不同位置（为了效率，只试几个随机位置+首尾）
                    positions = [0, len(new_os)]
                    positions += random.sample(range(len(new_os) + 1), min(5, len(new_os) + 1))
                    positions = list(set(positions))
                    for pos in positions:
                        test_os = new_os[:pos] + [job_id] + new_os[pos:]
                        cost = self.decode(test_os, new_ms)
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = pos
                            best_machine = machine
                # 用最优位置和机器插入
                new_os.insert(best_pos, job_id)
                new_ms[ms_index] = best_machine
        return new_os, new_ms

    def random_repair(self, remaining_os, removed_jobs, ms_chrom):
        """随机修复：被移除的工件，随机插入OS的位置，随机选机器。"""
        new_os = copy.deepcopy(remaining_os)
        new_ms = copy.deepcopy(ms_chrom)
        removed_list = list(removed_jobs)
        random.shuffle(removed_list)
        for job_id in removed_list:
            job_idx = job_id - 1
            num_ops = len(self.fjsp.PT[job_idx])
            for op_idx in range(num_ops):
                # 随机选插入位置
                pos = random.randint(0, len(new_os))
                new_os.insert(pos, job_id)
                # 随机选机器
                ms_index = self.job_op_offset[job_idx] + op_idx
                eligible = list(self.fjsp.MT[job_idx][op_idx])
                new_ms[ms_index] = random.choice(eligible)
        return new_os, new_ms

    def select_operator(self, weights):
        """按权重做轮盘赌选择，返回选中的算子下标。"""
        total = sum(weights)
        r = random.random() * total
        cumsum = 0
        for i, w in enumerate(weights):
            cumsum += w
            if r <= cumsum:
                return i
        return len(weights) - 1

    def update_weights(self):
        """根据算子得分更新权重，得分高的权重增加。"""
        for i in range(len(self.destroy_weights)):
            if self.destroy_counts[i] > 0:
                avg_score = self.destroy_scores[i] / self.destroy_counts[i]
                self.destroy_weights[i] = self.destroy_weights[i] * self.weight_decay + (1 - self.weight_decay) * avg_score
                self.destroy_weights[i] = max(0.1, self.destroy_weights[i])  # 防止权重降到0
            self.destroy_scores[i] = 0.0
            self.destroy_counts[i] = 0
        for i in range(len(self.repair_weights)):
            if self.repair_counts[i] > 0:
                avg_score = self.repair_scores[i] / self.repair_counts[i]
                self.repair_weights[i] = self.repair_weights[i] * self.weight_decay + (1 - self.weight_decay) * avg_score
                self.repair_weights[i] = max(0.1, self.repair_weights[i])
            self.repair_scores[i] = 0.0
            self.repair_counts[i] = 0

    def run(self):
        # 初始化
        cur_os, cur_ms = self.generate_initial_solution()
        cur_cost = self.decode(cur_os, cur_ms)
        best_os, best_ms, best_cost = copy.deepcopy(cur_os), copy.deepcopy(cur_ms), cur_cost
        T = self.T_init
        history = []
        weight_update_interval = 50  # 每50次迭代更新一次算子权重
        destroy_funcs = [self.random_destroy, self.worst_destroy]
        repair_funcs = [self.greedy_repair, self.random_repair]
        for iteration in range(self.max_iter):
            # 按权重选算子
            d_idx = self.select_operator(self.destroy_weights)
            r_idx = self.select_operator(self.repair_weights)
            # 破坏
            remaining_os, removed_jobs = destroy_funcs[d_idx](cur_os, cur_ms)
            # 修复
            new_os, new_ms = repair_funcs[r_idx](remaining_os, removed_jobs, cur_ms)
            # 评估
            new_cost = self.decode(new_os, new_ms)
            delta = new_cost - cur_cost
            # 算子评分
            if new_cost < best_cost:
                score = 3       # 找到全局最优，高分
            elif delta < 0:
                score = 2       # 比当前解好
            elif random.random() < math.exp(-delta / max(T, 0.01)):
                score = 1       # 更差但被接受
            else:
                score = 0       # 更差且被拒绝
            self.destroy_scores[d_idx] += score
            self.destroy_counts[d_idx] += 1
            self.repair_scores[r_idx] += score
            self.repair_counts[r_idx] += 1
            # 接受准则（和SA一样）
            if delta < 0:
                cur_os, cur_ms, cur_cost = new_os, new_ms, new_cost
            elif score == 1:  # 上面已经判断过概率接受
                cur_os, cur_ms, cur_cost = new_os, new_ms, new_cost
            # 更新全局最优
            if cur_cost < best_cost:
                best_os, best_ms, best_cost = copy.deepcopy(cur_os), copy.deepcopy(cur_ms), cur_cost
            # 定期更新算子权重
            if (iteration + 1) % weight_update_interval == 0:
                self.update_weights()
            history.append(best_cost)
            T = T * self.alpha
            if iteration % 100 == 0:
                print(f"Iter {iteration}: best makespan = {best_cost}, "
                f"D_weights={[round(w, 2) for w in self.destroy_weights]}, "
                f"R_weights={[round(w, 2) for w in self.repair_weights]}")

        print(f"ALNS最终结果: makespan = {best_cost}")
        return best_cost, history

if __name__ == "__main__":
    MT = pt_to_mt(PT)
    n = len(PT)
    m = len(PT[0][0])
    alns = ALNS(n, m, PT, MT)
    best_makespan, history = alns.run()
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Makespan")
    plt.title("ALNS Convergence")
    plt.savefig("alns_convergence.png")
    plt.show()