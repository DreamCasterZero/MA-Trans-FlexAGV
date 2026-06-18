import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14

# ================= 1. 真实数据录入 (来自您的 Table 6) =================
# 场景标签
labels = ['10×5\n(Small)', '15×10\n(Medium)', '20×10\n(Bottleneck)', '25×10\n(Large)', '30×10\n(Limit)']

# 均值数据 (Mean Makespan)
# 注意：这里直接填入您表格里的 Mean 值，程序会自动算百分比，绝对准确
means = {
    "MWKR":   [253.6, 419.5, 551.1, 507.1, 534.7],
    "ResGAT": [244.3, 373.7, 474.6, 398.4, 440.7],
    "LSTM":   [236.8, 376.5, 473.5, 403.3, 434.3],
    "MA":     [235.6, 363.1, 468.6, 384.1, 424.6]
}

# ================= 2. 自动计算提升率 (Improvement over Rule) =================
# 公式: (Rule - Method) / Rule * 100
def calc_imp(method_vals, rule_vals):
    return [((r - m) / r) * 100 for m, r in zip(method_vals, rule_vals)]

ma_imp = calc_imp(means["MA"], means["MWKR"])
res_imp = calc_imp(means["ResGAT"], means["MWKR"])
lstm_imp = calc_imp(means["LSTM"], means["MWKR"])

# 打印一下核对数据 (您可以在控制台看到修正后的数值)
print("Corrected Improvements (%):")
print(f"MA-Trans: {np.round(ma_imp, 1)}")
print(f"ResGAT:   {np.round(res_imp, 1)}")
print(f"LSTM-Ptr: {np.round(lstm_imp, 1)}")
# 预期：在 20x10 (第3个) 场景下，大家应该都是 14%-15% 左右，非常接近

# ================= 3. 绘图 =================
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

# 颜色：学术红、蓝、绿
rects1 = ax.bar(x - width, ma_imp, width, label='MA-Trans (Ours)', 
                color='#D62728', alpha=0.9, hatch='//', edgecolor='black')

rects2 = ax.bar(x, res_imp, width, label='ResGAT', 
                color='#1F77B4', alpha=0.8, hatch='..', edgecolor='black')

rects3 = ax.bar(x + width, lstm_imp, width, label='LSTM-Ptr', 
                color='#2CA02C', alpha=0.8, hatch='xx', edgecolor='black')

# ================= 4. 装饰 =================
ax.set_ylabel('Improvement over MWKR-EAT Rule (%)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)

# Y轴网格
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
ax.set_axisbelow(True)

# 图例
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

# 标注数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('fig_generalization_corrected.pdf', format='pdf', bbox_inches='tight')
plt.savefig('fig_generalization_corrected.png', format='png', dpi=300, bbox_inches='tight')
plt.show()