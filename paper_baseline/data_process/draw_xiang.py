import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
BASE_DIR = "./20-10-5"

FILES = {
    "MA-Trans": f"{BASE_DIR}/transformer/test.csv",
    "ResGAT": f"{BASE_DIR}/gnn/test.csv",
    "LSTM-Ptr": f"{BASE_DIR}/lstm/test.csv",
    "SA-PPO": f"{BASE_DIR}/sa/test.csv",
    "MA": f"{BASE_DIR}/ma/test.csv"
}

# 🎨 配色
COLORS = {
    "MA-Trans": "#D62728",
    "ResGAT": "#1F77B4",
    "LSTM-Ptr": "#2CA02C",
    "SA-PPO": "#9467BD",
    "MA": "#FF7F0E"
}

ORDER = ["MA-Trans", "ResGAT", "LSTM-Ptr", "SA-PPO", "MA"]


# ================= 2. 数据处理与 RPD 计算 =================
def load_and_calculate_rpd():
    # 1. 先把所有数据读到一个大的 DataFrame 里 (100行 x 5列)
    combined_df = pd.DataFrame()

    for name, path in FILES.items():
        try:
            # 读取数据
            df = pd.read_csv(path, header=None)
            if isinstance(df.iloc[0, 0], str):  # 跳过 header
                df = pd.read_csv(path)

            # 确保只有一列数据，且取前100个
            values = df.values.flatten()[:100]
            combined_df[name] = values

        except Exception as e:
            print(f"❌ 读取 {name} 失败: {e}")
            return None

    print("📊 原始数据加载完毕，开始计算 RPD (Gap)...")
    np.savetxt("all.csv", combined_df)

    # 2. 计算每一行的最小值 (Best Known Solution per Instance)
    # axis=1 表示按行求最小
    best_values = combined_df.min(axis=1)

    # 3. 计算 RPD: (当前值 - 最小值) / 最小值 * 100
    rpd_df = pd.DataFrame()
    for col in combined_df.columns:
        # 这里的计算是向量化的，每一行都会减去对应行的 best_value
        rpd = ((combined_df[col] - best_values) / best_values) * 100
        rpd_df[col] = rpd

        avg_gap = rpd.mean()
        print(f"   -> {col} 平均 Gap: {avg_gap:.2f}%")

    # 4. 转换为 Seaborn 需要的长格式 (Melt)
    # 结果包含两列: 'Method' 和 'RPD'
    melted_data = rpd_df.melt(var_name="Method", value_name="RPD")

    return melted_data


# ================= 3. 绘图主逻辑 =================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def draw_rpd_box():
    data = load_and_calculate_rpd()
    if data is None: return

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(x="Method", y="RPD",
                hue="Method",  # <--- 新增：让 hue 也等于 "Method"
                legend=False,  # <--- 新增：关闭多余的图例
                data=data, order=ORDER,
                palette=COLORS, width=0.5, linewidth=2.0,
                showfliers=True, ax=ax,
                flierprops={"marker": "d", "markerfacecolor": "gray", "markersize": 3})
    # ================= 4. 细节美化 =================
    ax.set_xlabel("", fontsize=14)

    # 🟢 关键修改：Y轴现在是 Gap (%)
    ax.set_ylabel("RPD to Best Known Solution (%)", fontsize=16, fontweight='bold')

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # 加虚线网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)

    # 可以在 y=0 处加一条参考线，表示"完美基准"
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.2)

    plt.tight_layout()
    plt.savefig("Boxplot_RPD_Comparison.pdf", dpi=300)
    plt.savefig("Boxplot_RPD_Comparison.png", dpi=300)
    print("🎉 RPD 箱线图绘制完成！")
    plt.show()


if __name__ == "__main__":
    draw_rpd_box()