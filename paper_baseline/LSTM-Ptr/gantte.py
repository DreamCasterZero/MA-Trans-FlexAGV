import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# ================= ⚙️ 全局字体配置 =================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

def get_color_map():
    cmap = plt.get_cmap('tab20')
    colors = [mcolors.rgb2hex(cmap(i)) for i in range(20)]
    return colors

def plot_gantt_on_axis(ax, machines, agvs, title=None, show_job_id=False):
    """
    绘制单张甘特图
    """
    colors = get_color_map()
    
    y_labels = []
    y_ticks = []
    
    # --- 1. 绘制 AGV (下半部分) ---
    for k, agv in enumerate(agvs):
        y_pos = k
        # 修改点1: 去掉空格，写成 AGV1, AGV2...
        y_labels.append(f'AGV{k+1}') 
        y_ticks.append(y_pos)
        
        for m in range(len(agv.using_time)):
            start, end = agv.using_time[m]
            duration = end - start
            if duration <= 1e-3: continue # 忽略过短的片段
            
            # 判断负载
            is_loaded = (agv.on[m] is not None) and (agv.on[m] > 0)
            
            if is_loaded:
                job_id = agv.on[m]
                color = colors[(job_id - 1) % len(colors)]
                # 实心条块
                ax.barh(y_pos, width=duration, left=start, height=0.7,
                        color=color, edgecolor='black', linewidth=0.5, alpha=0.9)
                if show_job_id and duration > 5:
                    ax.text(start + duration/2, y_pos, f'J{job_id}', 
                            ha='center', va='center', fontsize=8, color='white')
            else:
                # 空载条块 (白色 + 灰色斜线)
                ax.barh(y_pos, width=duration, left=start, height=0.4,
                        color='white', edgecolor='gray', linewidth=0.5, hatch='////')

    # --- 2. 分割线 ---
    separator_y = len(agvs) - 0.5
    ax.axhline(y=separator_y, color='black', linewidth=1.5)
    
    # --- 3. 绘制 Machine (上半部分) ---
    base_y_machine = len(agvs)
    for i, machine in enumerate(machines):
        y_pos = base_y_machine + i
        # 修改点2: 去掉空格，写成 M1, M2...
        y_labels.append(f'M{i+1}')
        y_ticks.append(y_pos)
        
        for j in range(len(machine.using_time)):
            start, end = machine.using_time[j]
            duration = end - start
            if duration <= 1e-3: continue
            
            job_id = machine.on[j]
            if job_id <= 0: continue
            
            color = colors[(job_id - 1) % len(colors)]
            ax.barh(y_pos, width=duration, left=start, height=0.7,
                    color=color, edgecolor='black', linewidth=0.5, alpha=0.9)
            
            if show_job_id and duration > 5:
                ax.text(start + duration/2, y_pos, f'J{job_id}', 
                        ha='center', va='center', fontsize=9, color='white')

    # --- 4. 坐标轴美化 ---
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11) # 字号稍大一点
    
    ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax.set_xlim(left=0)
    ax.set_ylim(-0.5, base_y_machine + len(machines) - 0.5)
    
    # 竖向网格线 (虚线)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # 去除上右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- 修改点3: 关于标题 ---
    # 学术论文建议不要在图里写标题，而是用 LaTeX 的 \caption
    # 如果你坚持要写，可以把 y 参数调大一点 (比如 -0.15)
    # 这里我建议直接注释掉，或者保留参数但默认不显示
    if title:
        # y=-0.15 会比 -0.25 更靠近图片
        # 但我强烈建议：不要在 Python 里加 title，在论文 Caption 里写
        # ax.set_title(title, y=-0.15, fontsize=12, fontweight='bold') 
        pass