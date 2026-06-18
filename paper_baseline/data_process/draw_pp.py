import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 配置区域 =================
# 请根据实际情况修改路径
ROOT = "./10-5-2"

FILES = {
    "MA-Trans": f"{ROOT}/transformer/test_1005.csv",
    "ResGAT": f"{ROOT}/gnn/test_1005.csv",
    "LSTM-Ptr": f"{ROOT}/lstm/test_1005.csv",
    "SA-PPO": f"{ROOT}/sa/test_1005.csv",
    "MA": f"{ROOT}/ma/test_1005.csv"
}

# 🎨 配色 (保持统一)
COLORS = {
    "MA-Trans": "#D62728",  # 红
    "ResGAT": "#1F77B4",  # 蓝
    "LSTM-Ptr": "#2CA02C",  # 绿
    "SA-PPO": "#9467BD",  # 紫
    "MA": "#FF7F0E"  # 橙
}

# 线型设计
LINE_STYLES = {
    "MA-Trans": "-",
    "ResGAT": "--",
    "LSTM-Ptr": "-.",
    "SA-PPO": ":",
    "MA": (0, (3, 1, 1, 1))
}


# ================= 2. 数据处理 =================
def load_and_calc_rpd():
    combined_data = {}

    for method_name, file_path in FILES.items():
        try:
            # 尝试读取，兼容有无 Header 的情况
            df = pd.read_csv(file_path, header=None)

            if isinstance(df.iloc[0, 0], str):
                df = pd.read_csv(file_path)

            # 确保数据都是数值型 (float)，取前100个
            values = pd.to_numeric(df.values.flatten(), errors='coerce')[:100]

            combined_data[method_name] = values
            print(f"✅ Loaded {method_name}: {len(values)} rows")

        except Exception as e:
            print(f"❌ Error loading {method_name}: {e}")
            return None

    df_all = pd.DataFrame(combined_data)

    # 计算每个算例的 Best Known Solution (每行的最小值)
    best_values = df_all.min(axis=1)

    # 计算 Gap (RPD)
    rpd_df = pd.DataFrame()
    for col in df_all.columns:
        rpd_df[col] = (df_all[col] - best_values) / best_values

    return rpd_df


# ================= 3. 绘制 Performance Profile =================
def draw_performance_profile():
    rpd_df = load_and_calc_rpd()
    if rpd_df is None: return

    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.size'] = 12

    # 🟢 修改1：设置画布大小为 8x5 英寸
    # 配合 DPI=300，最终像素为 2400x1500
    fig, ax = plt.subplots(figsize=(8, 5))

    # X 轴范围：展示 Gap 从 0 (最优) 到 0.5 (差50%)
    x_grid = np.linspace(0, 0.5, 1000)

    for method in FILES.keys():  # 保证顺序一致
        if method not in rpd_df.columns: continue

        gaps = rpd_df[method].values

        # 计算 ECDF
        y_values = []
        for x in x_grid:
            ratio = np.sum(gaps <= x + 1e-6) / len(gaps)
            y_values.append(ratio)

        # 绘制曲线
        is_ours = (method == "MA-Trans")
        lw = 3.0 if is_ours else 1.8
        alpha = 1.0 if is_ours else 0.8
        zorder = 10 if is_ours else 5

        ax.plot(x_grid * 100, np.array(y_values) * 100,
                label=method,
                color=COLORS[method],
                linestyle=LINE_STYLES[method],
                linewidth=lw,
                alpha=alpha,
                zorder=zorder)

    # ================= 4. 美化细节 =================
    ax.set_xlabel(r"Gap Tolerance $\tau$ (%)", fontsize=16, fontweight='bold')
    ax.set_ylabel(r"Fraction of Solved Instances (%)", fontsize=16, fontweight='bold')

    # X轴看前 30%
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 105)

    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.4)

    # 图例
    ax.legend(fontsize=13, loc='lower right', frameon=True, edgecolor='black', fancybox=False)

    # 🟢 修改2：使用 tight_layout 调整布局
    plt.tight_layout()

    # 🟢 修改3：去掉 bbox_inches='tight'，保留 dpi=300
    save_name = "Performance_Profile"
    plt.savefig(f"{save_name}.pdf", dpi=300)
    plt.savefig(f"{save_name}.png", dpi=300)

    print(f"🎉 绘图完成！已保存为 {save_name}.png (2400x1500)")
    plt.show()


if __name__ == "__main__":
    draw_performance_profile()