
class Job:
    def __init__(self, idx, ops_times): # ops_times (该工件的工序时间矩阵)
        self.idx = idx  # 工件在当前批次中的编号 (0, 1, 2...)
        self.PT = ops_times # 直接持有自己的加工时间数据 [[Op1_M1, Op1_M2...], [Op2_M1...]]
        self.cur_pos = 0 # 当前位置 (0代表在卸货区/仓库)
        self.cur_process = 1 # 当前是第几道工序 (从1开始)
        self.all_process = len(self.PT) # 总工序数
        self.end = 0 # 上一道工序完工时间
        self.last_time = 0 # 上一次操作结束时间
        self.last_pos = 0 # 上一次所在位置
        self.done = False # 是否全部完成
        self.choose = -1 # 当前工序选择了哪台机器

    # 获取当前工序在指定机器上的加工时间
    def get_process_time(self,mach_id):
        if mach_id < 1 or mach_id > len(self.PT[self.cur_process - 1]):
            return 9999  # 防止越界，返回一个大数代表不可行
        time = self.PT[self.cur_process-1][mach_id-1]
        return time

    def update(self, end, M):
        self.last_time = self.end
        self.last_pos = self.cur_pos
        self.end = end
        self.cur_process += 1
        self.cur_pos = M
        if self.cur_process > self.all_process:
            self.done = True

    def get_remaining_stats(self):
        """
        [新增辅助方法] 返回：(剩余工序的平均总耗时, 剩余工序数)
        """
        if self.done:
            return 0, 0

        total_remain_time = 0
        # 从当前工序开始往后算
        for i in range(self.cur_process - 1, self.all_process):
            # 拿到这一道工序所有可行机器的时间
            valid_times = [t for t in self.PT[i] if t > 0 and t < 9999]
            if valid_times:
                total_remain_time += sum(valid_times) / len(valid_times)

        return total_remain_time, (self.all_process - self.cur_process + 1)

class Machine:
    def __init__(self,idx):
        self.idx = idx  # 0~7
        self.pos = self.idx + 1 # 机器的物理位置编号 (假设机器1在位置1)
        self.enable = True
        self.end = 0 # 机器什么时候空闲
        self.using_time = [] # 记录使用历史 [[start, end], ...]
        self.on = [] # 记录加工了哪些工件

    def update(self, start, process_time, on):
        end = start + process_time
        self.using_time.append([start,end])
        self.on.append(on)
        self.end = end

class AGV:
    def __init__(self,idx,cur_pos=0):
        self.idx = idx
        self.cur_pos = cur_pos
        self.using_time = []
        self.on = [] # 搬运的工件
        self.to = [] # 路径记录
        self.end = 0 # AGV 什么时候空闲
        self.last = 0
        self.cur_start = 0

    def update(self, start, trans1, trans2, J_site, J_m, on_job_idx=None):
        self.using_time.append([start, start + trans1])
        self.using_time.append([start + trans1, start + trans1 + trans2])
        self.on.append(None) # 空载段
        self.on.append(on_job_idx) # 负载段
        self.to.extend([J_site,J_m])
        self.last = self.end
        self.cur_start = start
        self.end = start + trans1 + trans2
        self.cur_pos = J_m

    def is_free(self, current_time):
        return self.end <= current_time

class FJSP:
    def __init__(self, machine_num):
        # --- 初始化时只定死机器和AGV数量，不传具体的工件 ---
        self.machine_num = machine_num
        self.Machines = [Machine(i) for i in range(self.machine_num)]
        # Job 和 AGV 是动态的，初始化为空
        self.Jobs = []
        self.AGVs = []
        self.max = 0 # Makespan

    def reset(self, new_job_data=None, new_agv_num=None):
        """
        :param new_job_data: 动态生成的工件 PT 矩阵列表
        :param new_agv_num: 本轮随激生成的 AGV 数量 (int)
        """
        # 1. 重置机器 (机器是不变的，只是清空状态)
        for m in self.Machines:
            m.end = 0
            m.using_time = []
            m.on = []
            m.enable = True
        # 2. 重建/重置 AGV (数量会变)
        if new_agv_num is not None:
            self.AGVs = [AGV(i) for i in range(new_agv_num)]
        else:
            # 如果没传新数量，就只重置现有 AGV
            for agv in self.AGVs:
                agv.end = 0
                agv.cur_pos = 0
                agv.using_time = []
                agv.on = []
                agv.to = []
                agv.last = 0
                agv.cur_start = 0
        if new_job_data is not None:
            self.Jobs = []
            for i, ops_times in enumerate(new_job_data):
                self.Jobs.append(Job(i, ops_times))
        else:
            # 仅重置状态 (用于测试集固定案例)
            for job in self.Jobs:
                job.cur_pos = 0
                job.cur_process = 1
                job.end = 0
                job.last_time = 0
                job.last_pos = 0
                job.done = False
                job.choose = -1
        self.max = 0


