import pandas as pd
import numpy as np
import os

# ================= ⚙️ 配置区域 =================
# 这里对应你的文件名
FILES = {
    "MA-Trans": "./test_trans.csv",  # 你的方法
    "ResGAT":   "./test_resgat.csv", # 对比方法1
    "LSTM-Ptr": "./test_lstm.csv",   # 对比方法2 (重点对比对象)
    "SA-PPO":   "./test_sa.csv",     # 对比方法3
}

# 你想主要打败谁？
# 选项 A: 'Min_Baseline' (推荐) -> 比所有基准里最好的那个还要好，这才叫完胜。
# 选项 B: 'LSTM-Ptr' -> 专门针对 LSTM 找差距最大的（为了配合你正文的论述）。
TARGET_BASELINE = 'Min_Baseline' 
# TARGET_BASELINE = 'LSTM-Ptr' 

# ================= 🚀 处理脚本 =================
def find_best_case():
    print("🚀 开始寻找最佳可视化案例...")
    
    # 1. 读取数据
    data_dict = {}
    try:
        for method, path in FILES.items():
            if os.path.exists(path):
                # 假设 CSV 没有表头或者第一列就是数据，这里读取第一列
                # 如果你的 CSV 有表头 'Value'，pandas 会自动识别
                df = pd.read_csv(path)
                # 强制取第一列作为 makespan 数据
                data_dict[method] = df.iloc[:, 0].values
                print(f"✅ 已读取 {method}: {len(df)} 条数据")
            else:
                print(f"❌ 文件未找到: {path}")
                return
    except Exception as e:
        print(f"⚠️ 读取出错: {e}")
        return

    # 转换为 DataFrame 方便计算
    df_all = pd.DataFrame(data_dict)
    
    # 确保列对齐
    if df_all.isnull().values.any():
        print("⚠️ 警告：数据中存在缺失值，请检查 CSV 文件长度是否一致。")
        df_all = df_all.dropna()

    # 2. 计算差距 (Gap)
    # 找出所有基准方法列
    baseline_cols = [col for col in df_all.columns if col != "MA-Trans"]
    
    if TARGET_BASELINE == 'Min_Baseline':
        # 找出每行中，表现最好的那个基准（Makespan 最小）
        df_all['Best_Baseline_Val'] = df_all[baseline_cols].min(axis=1)
        df_all['Baseline_Name'] = df_all[baseline_cols].idxmin(axis=1) # 记录是谁
    else:
        # 指定对比 LSTM
        df_all['Best_Baseline_Val'] = df_all[TARGET_BASELINE]
        df_all['Baseline_Name'] = TARGET_BASELINE

    # 计算优势：基准 - 你的方法 (值越大，代表你缩短的时间越多)
    df_all['Gap'] = df_all['Best_Baseline_Val'] - df_all['MA-Trans']
    
    # 计算提升百分比 (Gap / Baseline)
    df_all['Improve_Pct'] = (df_all['Gap'] / df_all['Best_Baseline_Val']) * 100

    # 3. 排序找最大差距
    # 按照 Gap 降序排列
    top_cases = df_all.sort_values(by='Gap', ascending=False).head(5)

    # ================= 📊 输出结果 =================
    print("\n🏆 推荐用于画甘特图的 Top 5 案例 (差距最大):")
    print("-" * 60)
    print(f"{'Case ID':<8} | {'MA-Trans':<10} | {'Baseline':<10} ({'Method':<8}) | {'Gap':<8} | {'Improve':<8}")
    print("-" * 60)
    
    for idx, row in top_cases.iterrows():
        # idx 就是原始 CSV 里的行号 (0-based)
        print(f"{idx:<8} | {row['MA-Trans']:<10.1f} | {row['Best_Baseline_Val']:<10.1f} ({row['Baseline_Name']:<8}) | {row['Gap']:<8.1f} | {row['Improve_Pct']:.1f}%")
    
    print("-" * 60)
    
    best_id = top_cases.index[0]
    print(f"\n💡 建议选择 Case ID: [{best_id}]")
    print(f"   在这个案例中，你的方法比表现最好的基准快了 {top_cases.iloc[0]['Gap']:.1f} 秒。")
    print(f"   (注：Case ID 对应 CSV 中的行号，也就是第 {best_id+1} 个测试数据)")

if __name__ == "__main__":
    find_best_case()