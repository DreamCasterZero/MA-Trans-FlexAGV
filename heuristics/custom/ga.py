from config import PT, pt_to_mt
import copy
import random
import numpy as np
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

class GA:
    def __init__(self, n, m, PT, MT, pop_size=100, gene_size=100, pc=0.8, pm=0.1, N_elite=10):
        self.fjsp = FJSP(n, m, PT, MT)
        self.pop_size = pop_size
        self.gene_size = gene_size
        self.pc = pc
        self.pm = pm
        self.N_elite = N_elite
        self.Fit = []
        # 预计算每个工件的工序起止索引，用于MS染色体解码
        self.job_op_offset = []
        offset = 0
        self.total_ops = 0
        for job_pt in PT:
            self.job_op_offset.append(offset)
            offset += len(job_pt)
        self.total_ops = offset
        self.best_makespan = float('inf')
        self.best_os = None
        self.best_ms = None
    #独立编码、解码关联
    def initial_population(self):
        self.OS_Pop = []
        self.MS_Pop = []
        
        base_os = []
        for job_id, job_pt in enumerate(self.fjsp.PT):
            base_os.extend([job_id + 1] * len(job_pt))
        for _ in range(self.pop_size):
            os_chrom = copy.deepcopy(base_os)
            random.shuffle(os_chrom)
            self.OS_Pop.append(os_chrom)
            ms_chrom = []
            for job_idx, job_mt in enumerate(self.fjsp.MT):
                for op_mt in job_mt:
                    ms_chrom.append(random.choice(list(op_mt)))
            self.MS_Pop.append(ms_chrom)

    def decode(self, os_chrom, ms_chrom):
        self.fjsp.reset()
        # 辅助计数器：记录每个作业当前处理到第几道工序了
        job_op_counter = [0] * self.fjsp.n
        for job_id_plus1 in os_chrom:
            job_idx = job_id_plus1 - 1
            op_idx = job_op_counter[job_idx]
            ms_index = self.job_op_offset[job_idx] + op_idx
            machine_id = ms_chrom[ms_index]
            self.fjsp.decode(job_id_plus1, machine_id)
            job_op_counter[job_idx] += 1
        return self.fjsp.makespan
    
    def fitness(self):
        self.Fit = []
        for i in range(self.pop_size):
            cmax = self.decode(self.OS_Pop[i], self.MS_Pop[i])
            self.Fit.append(cmax)
    
    def select(self):
        best_idx = np.argmin(self.Fit)
        new_OS, new_MS = [], []
        for _ in range(self.N_elite):
            new_OS.append(copy.deepcopy(self.OS_Pop[best_idx]))
            new_MS.append(copy.deepcopy(self.MS_Pop[best_idx]))

        while len(new_OS) < self.pop_size:
            a, b = random.sample(range(self.pop_size), 2)
            winner = a if self.Fit[a] < self.Fit[b] else b
            new_OS.append(copy.deepcopy(self.OS_Pop[winner]))
            new_MS.append(copy.deepcopy(self.MS_Pop[winner]))

        self.OS_Pop = new_OS
        self.MS_Pop = new_MS

    def POX(self, p1, p2):
        jobs = list(set(p1))
        job_subset = random.sample(jobs, len(jobs) // 2)
        c1 = [0] * len(p1)
        c2 = [0] * len(p2)
        for i, job in enumerate(p1):
            if job in job_subset: c1[i] = job
        for i, job in enumerate(p2):
            if job in job_subset: c2[i] = job

        p2_idx = 0
        for i in range(len(c1)):
            if c1[i] == 0:
                while p2[p2_idx] in job_subset:
                    p2_idx += 1
                c1[i] = p2[p2_idx]
                p2_idx += 1
        p1_idx = 0
        for i in range(len(c2)):
            if c2[i] == 0:
                while p1[p1_idx] in job_subset:
                    p1_idx += 1
                c2[i] = p1[p1_idx]
                p1_idx += 1
        return c1, c2 

    def UniformCrossover(self, ms1, ms2):
        mask = [random.randint(0, 1) for _ in range(len(ms1))]
        child1 = []
        child2 = []
        for i, m in enumerate(mask):
            if m == 0:
                child1.append(ms1[i])
                child2.append(ms2[i])
            else:
                child1.append(ms2[i])
                child2.append(ms1[i])
        return child1, child2

    def crossover_operator(self):
        indices = list(range(self.pop_size))
        random.shuffle(indices)
        new_OS, new_MS = [], []
        for i in range(0, self.pop_size, 2):
            idx1, idx2 = indices[i], indices[i+1] if i+1 < self.pop_size else indices[0]
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
        self.OS_Pop = new_OS[:self.pop_size]
        self.MS_Pop = new_MS[:self.pop_size]

    def mutation_operator(self):
        for i in range(self.pop_size):
            if random.random() < self.pm:
                idx1, idx2 = random.sample(range(len(self.OS_Pop[i])), 2)
                self.OS_Pop[i][idx1], self.OS_Pop[i][idx2] = self.OS_Pop[i][idx2], self.OS_Pop[i][idx1]
            if random.random() < self.pm:
                m_idx = random.randint(0, len(self.MS_Pop[i]) - 1)
                target_job = -1
                target_op = -1
                for job_idx, offset in enumerate(self.job_op_offset):
                    next_offset = self.job_op_offset[job_idx+1] if job_idx+1 < len(self.job_op_offset) else self.total_ops
                    if offset <= m_idx < next_offset:
                        target_job = job_idx
                        target_op = m_idx - offset
                        break
                available_machines = list(self.fjsp.MT[target_job][target_op])
                if len(available_machines) > 1:
                    current_machine = self.MS_Pop[i][m_idx]
                    choices = [m for m in available_machines if m != current_machine]
                    if choices:
                        self.MS_Pop[i][m_idx] = random.choice(choices)
                
    def run(self):
        self.initial_population()
        self.fitness()
        best_idx = np.argmin(self.Fit)
        self.best_makespan = self.Fit[best_idx]
        self.best_os = copy.deepcopy(self.OS_Pop[best_idx])
        self.best_ms = copy.deepcopy(self.MS_Pop[best_idx])
        history = []
        for gen in range(self.gene_size):
            self.select()
            self.crossover_operator()
            self.mutation_operator()
            self.fitness()
            curr_best_idx = np.argmin(self.Fit)
            curr_best = self.Fit[curr_best_idx]
            if curr_best < self.best_makespan:
                self.best_makespan = curr_best
                self.best_os = copy.deepcopy(self.OS_Pop[curr_best_idx])
                self.best_ms = copy.deepcopy(self.MS_Pop[curr_best_idx])
            history.append(self.best_makespan)
            if gen % 10 == 0:
                print(f"Gen {gen}: best makespan = {self.best_makespan}")
        print(f"最终结果: makespan = {self.best_makespan}")
        return self.best_makespan, history

if __name__ == "__main__":
    MT = pt_to_mt(PT)
    n = len(PT)
    m = len(PT[0][0])
    ga = GA(n, m, PT, MT)
    best_makespan, history = ga.run()
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Makespan")
    plt.title("GA Convergence")
    plt.savefig("convergence.png")
    plt.show()
