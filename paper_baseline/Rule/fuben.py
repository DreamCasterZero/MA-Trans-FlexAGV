import numpy as np
import config
import random

class RuleSolver:
    def __init__(self, env):
        self.env = env
        self.FJSP = env.FJSP
        self.num_machines = env.num_machines

    # ==================================================
    # ✅ 修正版: Global Greedy (G-EST) - 彻底修复串行Bug
    # ==================================================
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

    # ==================================================
    # 🔴 修正版规则: Global Greedy (G-EST)
    # 逻辑: 遍历所有未完成工件，看"谁"能最快完成当前工序
    # ==================================================
    # ==================================================
    # ✅ 修正版: Global Greedy (G-EST) - 考虑 AGV 接货耗时
    # ==================================================
    # ==================================================
    # ✅ 修正版: Global Greedy (G-EST) - 彻底修复串行Bug
    # ==================================================
    # def solve_global_greedy(self):
    #     jobs = self.FJSP.Jobs
    #     agvs = self.FJSP.AGVs
    #     machines = self.FJSP.Machines

    #     candidates = [j for j in jobs if not j.done]
    #     if not candidates:
    #         return None, None, None

    #     best_job = None
    #     best_mach_id = -1
    #     best_agv_id = -1
        
    #     global_min_finish_time = float('inf')

    #     for job in candidates:
    #         p_idx = job.cur_process - 1
    #         if p_idx >= len(job.PT): continue
            
    #         op_times = job.PT[p_idx]
            
    #         for m_idx, proc_time in enumerate(op_times):
    #             if proc_time <= 0 or proc_time >= 9999: continue
                
    #             target_machine = machines[m_idx]
    #             time_job_to_mach = config.agv_trans[job.cur_pos][target_machine.pos]
                
    #             min_arrival_at_machine = float('inf')
    #             best_agv_idx_for_this_pair = -1
                
    #             for a_i, agv in enumerate(agvs):
    #                 # 🔥🔥🔥 核心修改在这里 🔥🔥🔥
    #                 # 只要 AGV 空闲了就能动，不用管全厂最晚的时间
    #                 agv_available_time = agv.end 
                    
    #                 # 2. AGV 跑过来接货要多久？ (Empty Travel)
    #                 time_agv_to_job = config.agv_trans[agv.cur_pos][job.cur_pos]
                    
    #                 # 3. AGV 到达工件位置的时间点
    #                 agv_arrive_job_time = agv_available_time + time_agv_to_job
                    
    #                 # 4. 真正的出发时间：必须等 AGV 到，且工件也做完了
    #                 # leave_job_site_time = max(agv_arrive_job_time, job.end)
    #                 # AGV 必须等 job.end 之后才出发
    #                 agv_start_move_time = max(agv.end, job.end)
    #                 arrive_job_site_time = agv_start_move_time + time_agv_to_job
    #                 # 此时工件肯定早就好了，直接装车
    #                 leave_job_site_time = arrive_job_site_time
    #                 # 5. 到达目标机器的时间点
    #                 arrive_mach_time = leave_job_site_time + time_job_to_mach
                    
    #                 if arrive_mach_time < min_arrival_at_machine:
    #                     min_arrival_at_machine = arrive_mach_time
    #                     best_agv_idx_for_this_pair = a_i + 1 
                
    #             if best_agv_idx_for_this_pair == -1: continue

    #             start_time = max(min_arrival_at_machine, target_machine.end)
    #             finish_time = start_time + proc_time
                
    #             if finish_time < global_min_finish_time:
    #                 global_min_finish_time = finish_time
    #                 best_job = job
    #                 best_mach_id = m_idx + 1
    #                 best_agv_id = best_agv_idx_for_this_pair

    #     if best_job is None:
    #         return self.solve_random_greedy()

    #     job_action = self.FJSP.Jobs.index(best_job) + 1
    #     return job_action, best_mach_id, best_agv_id
    # # ==================================================
    # 🧪 测试专用: Random + Greedy + Greedy
    # ==================================================
    def solve_random_greedy(self):
        """
        Job: Random (随机选一个没做完的)
        Machine: Greedy (最小完工时间)
        AGV: EAT (最早到达时间)
        """
        jobs = self.FJSP.Jobs
        agvs = self.FJSP.AGVs

        # 1. 筛选未完成的工件
        candidates = [j for j in jobs if not j.done]
        if not candidates:
            return None, None, None

        # --- 🎲 随机选 Job ---
        best_job = random.choice(candidates)

        # 2. Machine: 贪心 (最小完工时间)
        mach_end_times = [m.end for m in self.FJSP.Machines]
        mach_action, _, _, _ = self.env.calculate_greedy_info(best_job, mach_end_times)

        # 3. AGV: EAT (最早到达时间)
        best_agv_idx = -1
        min_arrival_time = float('inf')
        target_pos = best_job.cur_pos

        for i, agv in enumerate(agvs):
            # AGV 何时能动
            agv_ready = max(agv.end, self.env.last_makespan)
            dist = config.agv_trans[agv.cur_pos][target_pos]
            arrival = agv_ready + dist

            if arrival < min_arrival_time:
                min_arrival_time = arrival
                best_agv_idx = i + 1

        # 返回 Job 的 1-based ID
        job_action = self.FJSP.Jobs.index(best_job) + 1

        return job_action, mach_action, best_agv_idx
    # ==================================================
    # 规则 1: SRPT + EAT
    # ==================================================
    def solve_srpt_eat(self):
        """
        Job: SRPT (最短剩余加工时间)
        Machine: Greedy (最小完工时间 - 复用环境逻辑)
        AGV: EAT (最早到达时间 - 包含等待)
        """
        jobs = self.FJSP.Jobs
        agvs = self.FJSP.AGVs

        # 1. 筛选未完成的工件
        candidates = [j for j in jobs if not j.done]
        if not candidates:
            return None, None, None

        # 2. SRPT: 计算剩余时间并找最小
        # 环境里有 calculate_remaining_time(job, start_op_idx)
        best_job = None
        min_remain_time = float('inf')

        for job in candidates:
            # 算出剩余时间
            remain = self.env.calculate_remaining_time(job, job.cur_process)
            # 加上当前工序的预估耗时 (简单用平均值或最小加工时间)
            # 这里为了简单，只比较后续剩余时间，或者 "后续+当前平均"
            # 按照 SRPT 标准定义，通常指由当前工序开始的所有工序时间之和
            if remain < min_remain_time:
                min_remain_time = remain
                best_job = job

        # 3. Machine: 贪心 (最小完工时间)
        # 直接调用环境的 helper，它返回的是 1-based ID
        mach_end_times = [m.end for m in self.FJSP.Machines]
        mach_action, _, _, _ = self.env.calculate_greedy_info(best_job, mach_end_times)

        # 4. AGV: EAT (最早到达时间)
        best_agv_idx = -1  # 1-based
        min_arrival_time = float('inf')

        target_pos = best_job.cur_pos

        for i, agv in enumerate(agvs):
            # AGV 何时能动
            agv_ready = max(agv.end, self.env.last_makespan)
            dist = config.agv_trans[agv.cur_pos][target_pos]
            arrival = agv_ready + dist

            if arrival < min_arrival_time:
                min_arrival_time = arrival
                best_agv_idx = i + 1

        # 返回 Job 的 1-based ID
        # 假设 jobs 列表顺序就是 ID 顺序 0~N-1
        job_action = self.FJSP.Jobs.index(best_job) + 1

        return job_action, mach_action, best_agv_idx

    # ==================================================
    # 规则 2: EST + ND
    # ==================================================
    def solve_est_nd(self):
        """
        修正后的 EST 策略：
        遍历所有工件，选择那个能 "最早开始加工 (Earliest Start)" 的工件。
        这通常能极大地减少机器的空转时间。
        """
        jobs = self.FJSP.Jobs
        agvs = self.FJSP.AGVs
        machines = self.FJSP.Machines
        
        candidates = [j for j in jobs if not j.done]
        if not candidates: return None, None, None

        best_job = None
        best_mach_id = -1
        best_agv_id = -1
        min_start_time = float('inf') # 比较的是开始时间，不是完工时间

        # 逻辑同上，但是比较指标变了
        for job in candidates:
            p_idx = job.cur_process - 1
            op_times = job.PT[p_idx]
            
            for m_idx, proc_time in enumerate(op_times):
                if proc_time <= 0 or proc_time >= 9999: continue
                machine = machines[m_idx]
                dist_j2m = config.agv_trans[job.cur_pos][machine.pos]
                
                # 找最好的 AGV
                min_arrival = float('inf')
                curr_agv = -1
                for a_i, agv in enumerate(agvs):
                    agv_ready = max(agv.end, self.env.last_makespan)
                    dist_a2j = config.agv_trans[agv.cur_pos][job.cur_pos]
                    meet = max(agv_ready + dist_a2j, job.end)
                    arr = meet + dist_j2m
                    if arr < min_arrival:
                        min_arrival = arr
                        curr_agv = a_i + 1
                
                # 计算开始时间
                start_time = max(min_arrival, machine.end)
                
                # === 区别在这里：我们选开始最早的，而不是结束最早的 ===
                # 这有助于填补时间缝隙
                if start_time < min_start_time:
                    min_start_time = start_time
                    best_job = job
                    best_mach_id = m_idx + 1
                    best_agv_id = curr_agv

        if best_job is None: return self.solve_random_greedy()
        return self.FJSP.Jobs.index(best_job) + 1, best_mach_id, best_agv_id