import numpy as np
import config

class RuleSolver:
    def __init__(self, env):
        self.env = env
        self.FJSP = env.FJSP
        self.num_machines = env.num_machines

    # 辅助函数：统一的保守型 AGV 选择逻辑
    def _find_best_agv_conservative(self, job, machine):
        """
        在'旧环境'逻辑下，寻找能最早把 job 送到 machine 的 AGV。
        约束：AGV 必须等 job.end 之后才能出发 (transport_start >= job.end)
        """
        best_agv_idx = -1
        min_arrival_at_machine = float('inf')
        agvs = self.FJSP.AGVs
        
        for a_i, agv in enumerate(agvs):
            # 1. 物理距离
            time_agv_to_job = config.agv_trans[agv.cur_pos][job.cur_pos]
            time_job_to_mach = config.agv_trans[job.cur_pos][machine.pos]
            
            # 2. 保守逻辑核心：出发时间限制
            # AGV 必须等到 (AGV空闲 AND 工件完工) 两个条件都满足，且逻辑上
            # 模拟的是"工件完工信号发出后，AGV才出发"
            start_move_time = max(agv.end, job.end)
            
            # 3. 计算时间轴
            arrive_job_site = start_move_time + time_agv_to_job
            # 离开工件位置 (到了就能装，因为 job.end 肯定过了)
            leave_job_site = arrive_job_site 
            
            arrive_mach = leave_job_site + time_job_to_mach
            
            if arrive_mach < min_arrival_at_machine:
                min_arrival_at_machine = arrive_mach
                best_agv_idx = a_i + 1
        
        return best_agv_idx, min_arrival_at_machine

    # MWKR (Most Work Remaining)
    # 策略：Job选剩余总时间最长的 -> Machine选最早完工(EST) -> AGV选保守EAT
    def solve_mwkr(self):
        jobs = self.FJSP.Jobs
        machines = self.FJSP.Machines
        
        candidates = [j for j in jobs if not j.done]
        if not candidates: return None, None, None

        # 1. 第一步：筛选出剩余工作量最大的那个工件 (MWKR核心)
        best_job = None
        max_remain_work = -1

        for job in candidates:
            # 手动计算剩余工作量 (Sum of average processing times of remaining ops)
            remain_work = 0
            # 从当前工序算起
            for i in range(job.cur_process - 1, len(job.PT)):
                valid_times = [t for t in job.PT[i] if 0 < t < 9999]
                if valid_times:
                    remain_work += sum(valid_times) / len(valid_times)
            
            if remain_work > max_remain_work:
                max_remain_work = remain_work
                best_job = job
        
        if best_job is None: return self.solve_random_greedy()

        # 2. 第二步：给这个"大哥"工件选一个能最早完工的机器 (Standard Greedy)
        best_mach_id = -1
        best_agv_id = -1
        min_finish_time = float('inf')

        p_idx = best_job.cur_process - 1
        op_times = best_job.PT[p_idx]

        for m_idx, proc_time in enumerate(op_times):
            if proc_time <= 0 or proc_time >= 9999: continue
            
            target_machine = machines[m_idx]
            
            # 使用保守逻辑找 AGV
            agv_id, arrive_mach_time = self._find_best_agv_conservative(best_job, target_machine)
            if agv_id == -1: continue

            # 计算完工时间
            start_process = max(arrive_mach_time, target_machine.end)
            finish_time = start_process + proc_time

            if finish_time < min_finish_time:
                min_finish_time = finish_time
                best_mach_id = m_idx + 1
                best_agv_id = agv_id

        if best_mach_id == -1: return self.solve_random_greedy()

        return self.FJSP.Jobs.index(best_job) + 1, best_mach_id, best_agv_id
    # Global Greedy (G-EST)

    def solve_global_greedy(self):
        jobs = self.FJSP.Jobs
        agvs = self.FJSP.AGVs
        machines = self.FJSP.Machines

        candidates = [j for j in jobs if not j.done]
        if not candidates:
            return None, None, None

        best_job = None
        best_mach_id = -1
        best_agv_id = -1
        
        global_min_finish_time = float('inf')

        for job in candidates:
            p_idx = job.cur_process - 1
            if p_idx >= len(job.PT): continue
            
            op_times = job.PT[p_idx]
            
            for m_idx, proc_time in enumerate(op_times):
                if proc_time <= 0 or proc_time >= 9999: continue
                
                target_machine = machines[m_idx]
                time_job_to_mach = config.agv_trans[job.cur_pos][target_machine.pos]
                
                min_arrival_at_machine = float('inf')
                best_agv_idx_for_this_pair = -1
                for a_i, agv in enumerate(agvs):
                    # AGV 何时能动
                    agv_available_time = agv.end 
                    
                    # 2. AGV 跑过来接货要多久？
                    time_agv_to_job = config.agv_trans[agv.cur_pos][job.cur_pos]
                    
                    # === 🔴 降级修改：模拟旧环境的“傻等”逻辑 ===
                    # 旧逻辑：AGV 必须等到 job.end 之后，才能从它当前的位置出发
                    # 这意味着：start_move_time = max(agv.end, job.end)
                    start_move_time = max(agv_available_time, job.end)
                    
                    # 3. AGV 到达工件位置的时间点
                    # 出发时间 + 路程
                    agv_arrive_job_time = start_move_time + time_agv_to_job
                    
                    # 4. 离开工件位置的时间
                    # 既然是等到 job.end 才出发的，到了肯定已经好了，直接走
                    leave_job_site_time = agv_arrive_job_time
                    
                    # === 修改结束 ===

                    # 5. 到达目标机器的时间点
                    arrive_mach_time = leave_job_site_time + time_job_to_mach
                
                # for a_i, agv in enumerate(agvs):
                #     # 🔥🔥🔥 核心修改在这里 🔥🔥🔥
                #     # 只要 AGV 空闲了就能动，不用管全厂最晚的时间
                #     agv_available_time = agv.end 
                    
                #     # 2. AGV 跑过来接货要多久？ (Empty Travel)
                #     time_agv_to_job = config.agv_trans[agv.cur_pos][job.cur_pos]
                    
                #     # 3. AGV 到达工件位置的时间点
                #     agv_arrive_job_time = agv_available_time + time_agv_to_job
                    
                #     # 4. 真正的出发时间：必须等 AGV 到，且工件也做完了
                #     leave_job_site_time = max(agv_arrive_job_time, job.end)
                    
                #     # 5. 到达目标机器的时间点
                #     arrive_mach_time = leave_job_site_time + time_job_to_mach
                    
                    if arrive_mach_time < min_arrival_at_machine:
                        min_arrival_at_machine = arrive_mach_time
                        best_agv_idx_for_this_pair = a_i + 1 
                
                if best_agv_idx_for_this_pair == -1: continue

                start_time = max(min_arrival_at_machine, target_machine.end)
                finish_time = start_time + proc_time
                
                if finish_time < global_min_finish_time:
                    global_min_finish_time = finish_time
                    best_job = job
                    best_mach_id = m_idx + 1
                    best_agv_id = best_agv_idx_for_this_pair

        if best_job is None:
            return self.solve_random_greedy()

        job_action = self.FJSP.Jobs.index(best_job) + 1
        return job_action, best_mach_id, best_agv_id
