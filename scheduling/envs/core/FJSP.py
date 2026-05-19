# sentinel value for infeasible machine assignments; must exceed any real processing time
INFEASIBLE = 9999

class Job:
    def __init__(self, idx, ops_times):
        self.idx = idx
        self.PT = ops_times       # [[op1_m1, op1_m2, ...], [op2_m1, ...], ...]
        self.cur_pos = 0          # 当前位置，0 = 仓库/卸货区
        self.cur_process = 1      # 当前工序编号，1-based
        self.all_process = len(self.PT)
        self.end = 0              # 上一道工序完工时间
        self.last_time = 0
        self.last_pos = 0
        self.done = False

    def reset(self):
        self.cur_pos = 0
        self.cur_process = 1
        self.end = 0
        self.last_time = 0
        self.last_pos = 0
        self.done = False

    def update(self, end, machine_pos):
        self.last_time = self.end
        self.last_pos = self.cur_pos
        self.end = end
        self.cur_pos = machine_pos
        self.cur_process += 1
        if self.cur_process > self.all_process:
            self.done = True

    def get_process_time(self, mach_id):
        if self.done:
            return INFEASIBLE
        if mach_id < 1 or mach_id > len(self.PT[self.cur_process - 1]):
            return INFEASIBLE
        return self.PT[self.cur_process - 1][mach_id - 1]

    def get_remaining_stats(self):
        if self.done:
            return 0.0, 0
        total = 0.0
        for i in range(self.cur_process - 1, self.all_process):
            valid = [t for t in self.PT[i] if 0 < t < INFEASIBLE]
            if valid:
                total += sum(valid) / len(valid)
        return total, self.all_process - self.cur_process + 1

class Machine:
    def __init__(self, idx):
        self.idx = idx
        self.pos = self.idx + 1   # 物理位置编号，1-based，与 AGV 位置系统对齐
        self.enabled = True
        self.end = 0              # 机器空闲时刻
        self.using_time = []      # [[start, end], ...]，用于甘特图
        self.on = []              # 记录加工的工件序号，用于甘特图

    def reset(self):
        self.enabled = True
        self.end = 0
        self.using_time = []
        self.on = []

    def update(self, start, process_time, job_idx):
        end = start + process_time
        self.using_time.append([start, end])
        self.on.append(job_idx)
        self.end = end

class AGV:
    def __init__(self, idx, init_pos=0):
        self.idx = idx
        self.enabled = True
        self.cur_pos = init_pos
        self.end = 0              # AGV 空闲时刻
        self.using_time = []      # [[start, end], ...]，用于甘特图
        self.on = []              # 搬运记录（None=空载段，job_idx=负载段），用于甘特图
        self.to = []              # 路径记录，用于甘特图

    def reset(self):
        self.enabled = True
        self.cur_pos = 0
        self.end = 0
        self.using_time = []
        self.on = []
        self.to = []

    def update(self, start, empty_time, loaded_time, pickup_pos, delivery_pos, job_idx=None):
        self.using_time.append([start, start + empty_time])
        self.using_time.append([start + empty_time, start + empty_time + loaded_time])
        self.on.append(None)
        self.on.append(job_idx)
        self.to.extend([pickup_pos, delivery_pos])
        self.end = start + empty_time + loaded_time
        self.cur_pos = delivery_pos

    def is_free(self, current_time):
        return self.end <= current_time

class FJSP:
    def __init__(self, machine_num):
        self.machine_num = machine_num
        self.machines = [Machine(i) for i in range(machine_num)]
        self.jobs = []
        self.agvs = []
        self.makespan = 0

    def reset(self, new_job_data=None, new_agv_num=None):
        for m in self.machines:
            m.reset()

        if new_agv_num is not None:
            self.agvs = [AGV(i) for i in range(new_agv_num)]
        else:
            for agv in self.agvs:
                agv.reset()

        if new_job_data is not None:
            self.jobs = [Job(i, ops_times) for i, ops_times in enumerate(new_job_data)]
        else:
            for job in self.jobs:
                job.reset()

        self.makespan = 0
