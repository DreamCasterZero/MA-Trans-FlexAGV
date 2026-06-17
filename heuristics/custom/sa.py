from matplotlib.backend_bases import cursors
from config import PT, pt_to_mt
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import math

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

class SA:
    def __init__(self, n, m, PT, MT, T_init=500, T_min=1, alpha=0.95, L=50):
        self.fjsp = FJSP(n, m, PT, MT)
        self.T_init = T_init
        self.T_min = T_min
        self.alpha = alpha
        self.L = L
        self.job_op_offset = []
        offset = 0
        self.total_ops = 0
        for job_pt in PT:
            self.job_op_offset.append(offset)
            offset += len(job_pt)
        self.total_ops = offset

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

    def neighbor(self, os_chrom, ms_chrom):
        new_os = copy.deepcopy(os_chrom)
        new_ms = copy.deepcopy(ms_chrom)
        idx1, idx2 = random.sample(range(len(new_os)), 2)
        new_os[idx1], new_os[idx2] = new_os[idx2], new_os[idx1]
        # MS扰动：随机改变一个工序的机器
        m_idx = random.randint(0, len(new_ms) - 1)
        target_job = -1
        target_op = -1
        for job_idx, offset in enumerate(self.job_op_offset):
            next_offset = self.job_op_offset[job_idx + 1] if job_idx + 1 < len(self.job_op_offset) else self.total_ops
            if offset <= m_idx < next_offset:
                target_job = job_idx
                target_op = m_idx - offset
                break
        available_machines = list(self.fjsp.MT[target_job][target_op])
        if len(available_machines) > 1:
            current_machine = new_ms[m_idx]
            choices = [m for m in available_machines if m != current_machine]
            if choices:
                new_ms[m_idx] = random.choice(choices)

        return new_os, new_ms

    def run(self):
        cur_os, cur_ms = self.generate_initial_solution()
        cur_cost = self.decode(cur_os, cur_ms)
        best_os, best_ms, best_cost = copy.deepcopy(cur_os), copy.deepcopy(cur_ms), cur_cost
        T = self.T_init
        history = []
        while T > self.T_min:
            for _ in range(self.L):
                new_os, new_ms = self.neighbor(cur_os, cur_ms)
                new_cost = self.decode(new_os, new_ms)
                delta = new_cost - cur_cost
                if delta < 0:
                    cur_os, cur_ms, cur_cost = new_os, new_ms, new_cost
                elif random.random() < math.exp(-delta / T):
                    cur_os, cur_ms, cur_cost = new_os, new_ms, new_cost
                if cur_cost < best_cost:
                    best_os, best_ms, best_cost = copy.deepcopy(cur_os), copy.deepcopy(cur_ms), cur_cost
            history.append(best_cost)
            T = T * self.alpha
        print(f"SA最终结果: makespan = {best_cost}")
        return best_cost, history

if __name__ == "__main__":
    MT = pt_to_mt(PT)
    n = len(PT)
    m = len(PT[0][0])
    sa = SA(n, m, PT, MT)
    best_makespan, history = sa.run()
    plt.plot(history)
    plt.xlabel("Temperature Step")
    plt.ylabel("Makespan")
    plt.title("SA Convergence")
    plt.savefig("sa_convergence.png")
    plt.show()

