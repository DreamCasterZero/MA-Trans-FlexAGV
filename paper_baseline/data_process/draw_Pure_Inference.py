import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# ================= ⚙️ 配置区域 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14

# 定义所有数据源 (把4个规模的文件全读进来)
ALL_DIRS = [
    "./10-5-2",
    "./15-10-4",
    "./20-5-3",
    "./20-10-5"
]

METHODS = ["MA-Trans", "ResGAT", "LSTM-Ptr"]
LABELS_MAP = {
    "MA-Trans": "MA-Trans (Ours)",
    "ResGAT": "ResGAT",
    "LSTM-Ptr": "LSTM-Ptr"
}
# 颜色配置 (红、蓝、绿)
COLORS = ["#D62728", "#1F77B4", "#2CA02C"]
HATCHES = ["//", "..", "xx"]

# ================= 📥 数据聚合 =================
aggregated_data = {m: [] for m in METHODS}

print("🚀 开始读取所有规模的数据...")

for base_dir in ALL_DIRS:
    # 映射文件夹名到实际路径 (根据您之前的代码习惯)
    # 假设结构是 ./10-5-2/transformer/time.csv
    # 如果您的文件夹命名不同，请在这里微调
    paths = {
        "MA-Trans": os.path.join(base_dir, "transformer/time.csv"),
        "ResGAT":   os.path.join(base_dir, "gnn/time.csv"),
        "LSTM-Ptr": os.path.join(base_dir, "lstm/time.csv")
    }
    
    for method in METHODS:
        full_path = paths[method]
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path)
                vals = df.iloc[:, 0].values.tolist()
                aggregated_data[method].extend(vals) # 把数据追加进去
            except:
                print(f"⚠️ 读取失败: {full_path}")
        else:
            # print(f"❌ 文件不存在: {full_path}")
            pass

# 计算最终的均值和标准差
final_means = []
final_stds = []

print("\n📊 最终统计结果:")
for method in METHODS:
    data = np.array(aggregated_data[method])
    if len(data) > 0:
        mean_val = np.mean(data)
        std_val = np.std(data)
    else:
        mean_val = 0
        std_val = 0
    
    final_means.append(mean_val)
    final_stds.append(std_val)
    print(f"  {method}: {mean_val:.4f} ms ± {std_val:.4f}")

# ================= 🎨 绘图 =================
fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(METHODS))
width = 0.5 # 单柱子可以宽一点

# 绘制柱子
bars = ax.bar(x, final_means, width, yerr=final_stds, 
              color=COLORS, edgecolor='black', 
              alpha=0.9, capsize=6, hatch=HATCHES)

# 为每个柱子设置不同的纹理 (matplotlib bar返回的是列表，需要手动设置)
for bar, hatch in zip(bars, HATCHES):
    bar.set_hatch(hatch)

# ================= 🎨 绘图 (修正版) =================
fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(METHODS))
width = 0.5 

# 绘制柱子
bars = ax.bar(x, final_means, width, yerr=final_stds, 
              color=COLORS, edgecolor='black', 
              alpha=0.9, capsize=6, hatch=HATCHES)

# 设置纹理
for bar, hatch in zip(bars, HATCHES):
    bar.set_hatch(hatch)

# ----------------- 🔴 重点修改区域：数值标注 -----------------
# 我们使用 enumerate 同时获取 index (i) 和柱子对象 (bar)
# 这样就可以从 final_stds 列表中拿到对应的标准差
for i, bar in enumerate(bars):
    mean_val = final_means[i]
    std_val = final_stds[i]
    
    # 计算标注的 Y 轴位置：均值 + 标准差 (即误差棒的顶端)
    y_pos = mean_val + std_val
    
    ax.annotate(f'{mean_val:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, y_pos), # 基准点设在误差棒顶端
                xytext=(0, 5),  # 再往上偏移 5 个点，保证不压线
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')
# -----------------------------------------------------------

# ================= 🖼️ 装饰 =================
ax.set_ylabel('Average Inference Time (ms)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([LABELS_MAP[m] for m in METHODS])

# Y轴网格
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
ax.set_axisbelow(True)

# 🔴 自动调整 Y 轴上限
# 因为数字跑到了误差棒上面，所以要把天花板再顶高一点，防止数字被切掉
# 取 (均值+标准差) 的最大值，再乘以 1.15 倍留空
max_height = max([m + s for m, s in zip(final_means, final_stds)])
ax.set_ylim(0, max_height * 1.15)

plt.tight_layout()
plt.savefig('fig_inference_final.pdf', format='pdf', bbox_inches='tight')
plt.savefig('fig_inference_final.png', format='png', dpi=300, bbox_inches='tight')
plt.show()