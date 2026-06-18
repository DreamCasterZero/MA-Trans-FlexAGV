import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
BASE_DIR = "./20-5-3"

FILES = {
    "MA-Trans": f"{BASE_DIR}/transformer/eval.csv",
    "ResGAT": f"{BASE_DIR}/gnn/eval.csv",
    "LSTM-Ptr": f"{BASE_DIR}/lstm/eval.csv",
    "SA-PPO": f"{BASE_DIR}/sa/eval.csv",
}

# 🎨 配色方案 (Key 必须严格对应)
COLORS = {
    "MA-Trans": "#D62728",  # 红色 (主角)
    "ResGAT":   "#1F77B4",  # 蓝色
    "LSTM-Ptr": "#2CA02C",  # 绿色
    "SA-PPO":   "#9467BD"   # 紫色
}

# 线型设计
LINE_STYLES = {
    "MA-Trans": "-",   # 实线
    "ResGAT":   "--",  # 虚线
    "LSTM-Ptr": "-.",  # 点划线
    "SA-PPO":   ":"    # 点线
}
SMOOTH_FACTOR = 0.90  # 平滑系数
MAX_STEPS = 100000  # X轴截断

# ================= 2. 数据处理函数 =================
def smooth_curve(scalars, weight):
    """ 指数加权平均平滑 """
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def clean_data_logic(file_path):
    """ 清洗数据：处理断点续训导致的 Step 重置问题 """
    try:
        df = pd.read_csv(file_path)
        steps = df['Step'].values
        values = df['Value'].values

        # 检测断点
        reset_indices = np.where(np.diff(steps) < 0)[0]

        if len(reset_indices) > 0:
            print(f"⚠️ 检测到断点续训: {file_path}，正在拼接数据...")
            split_idx = reset_indices[0]
            resume_step = steps[split_idx + 1]

            # 保留第一段中 < resume_step 的数据
            valid_mask_1 = steps[:split_idx + 1] < resume_step
            steps_1 = steps[:split_idx + 1][valid_mask_1]
            values_1 = values[:split_idx + 1][valid_mask_1]

            # 拼接第二段
            steps_2 = steps[split_idx + 1:]
            values_2 = values[split_idx + 1:]

            final_steps = np.concatenate([steps_1, steps_2])
            final_values = np.concatenate([values_1, values_2])

            # 重新排序
            sort_idx = np.argsort(final_steps)
            return final_steps[sort_idx], final_values[sort_idx]

        return steps, values
    except Exception as e:
        print(f"数据读取错误 {file_path}: {e}")
        return np.array([]), np.array([])


# ================= 3. 绘图主逻辑 =================
# 全局字体设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# 🟢 修改1：设置画布大小为 8x5 英寸
# 配合 DPI=300，最终像素为 (8*300)x(5*300) = 2400x1500
fig, ax = plt.subplots(figsize=(8, 5))

min_final_score = 9999

for name, path in FILES.items():
    steps, values = clean_data_logic(path)

    if len(steps) == 0: continue

    # 截断数据
    mask = steps <= MAX_STEPS
    steps = steps[mask]
    values = values[mask]

    # 平滑
    smoothed = smooth_curve(values, SMOOTH_FACTOR)

    # 记录最低分用于自动调整 Y 轴下限
    if len(smoothed) > 0:
        min_final_score = min(min_final_score, np.min(smoothed[-10:]))

    # 设置图层顺序和线宽
    zorder = 2
    lw = 2.0

    if name == "MA-Trans":
        zorder = 10
        lw = 3.0
    elif name == "ResGAT":
        zorder = 5
        lw = 2.5

    # === 绘制曲线 ===
    # 1. 阴影 (Raw Data)
    ax.plot(steps, values, color=COLORS[name], alpha=0.1, linewidth=0.8, zorder=1)

    # 2. 实线 (Smoothed)
    ax.plot(steps, smoothed, color=COLORS[name], label=name,
            linewidth=lw, linestyle=LINE_STYLES[name], zorder=zorder)

# ================= 4. 细节调整 =================

ax.set_xlabel("Training Episodes", fontsize=16, fontweight='bold')
ax.set_ylabel("Average Makespan", fontsize=16, fontweight='bold')
ax.tick_params(labelsize=14)

ax.set_xlim(0, MAX_STEPS)
ax.set_ylim(max(0, min_final_score - 20), 480)

ax.grid(True, which='major', linestyle='--', alpha=0.4)

ax.legend(fontsize=13, loc='upper right', frameon=True, edgecolor='black', fancybox=False)

# ================= 5. 保存 =================
# 🟢 修改2：使用 tight_layout 自动调整间距，防止文字被切，但不改变图片物理尺寸
plt.tight_layout()

# 🟢 修改3：去掉 bbox_inches='tight'，保留 dpi=300
# 这样保存出来的图片就是严格的 2400x1500
plt.savefig("Compare_SOTA.pdf", dpi=300)
plt.savefig("Compare_SOTA.png", dpi=300)

print("✅ 绘图完成！已保存为 Compare_SOTA.png (2400x1500)")
plt.show()