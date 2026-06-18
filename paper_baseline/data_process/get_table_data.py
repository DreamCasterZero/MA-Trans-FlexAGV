import pandas as pd
import numpy as np
from scipy import stats
import os
BASE_DIR = "./20-10-5"

FILES = {
    "MA-Trans": f"{BASE_DIR}/transformer/test.csv",
    "ResGAT": f"{BASE_DIR}/gnn/test.csv",
    "LSTM-Ptr": f"{BASE_DIR}/lstm/test.csv",
    "SA-PPO": f"{BASE_DIR}/sa/test.csv",
    "MA": f"{BASE_DIR}/ma/test.csv",
    "G-EST": f"{BASE_DIR}/G-EST/test.csv",
    "MWKR-EAT": f"{BASE_DIR}/MWKR/test.csv",
}

OUR_METHOD_NAME = "MA-Trans"

def load_data(files):
    data_dict = {}
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"⚠️ 警告: 文件不存在，跳过 - {path}")
            continue
        
        try:
            # 读取 CSV，假设只有一列数据或者表头是 'Makespan'
            df = pd.read_csv(path)
            # 如果有多列，取第一列数值；或者指定列名 'Makespan'
            if 'Makespan' in df.columns:
                values = df['Makespan'].values
            else:
                values = df.iloc[:, 0].values
            
            data_dict[name] = values
        except Exception as e:
            print(f"❌ 读取错误 {name}: {e}")
    return data_dict

def calculate_metrics(data_dict, baseline_name):
    if baseline_name not in data_dict:
        raise ValueError(f"❌ 基准方法 {baseline_name} 的数据未找到！无法计算 Gap 和 p-value。")

    baseline_data = data_dict[baseline_name]
    baseline_mean = np.mean(baseline_data)
    
    results = []
    
    # 按照 FILES 定义的顺序或者自定义顺序输出
    # 这里我们按 Mean 降序排列 (除了 Baseline 放最后)
    sorted_keys = list(data_dict.keys())
    # 如果想把 Ours 放在最后，可以解除下面这行的注释
    if baseline_name in sorted_keys:
        sorted_keys.remove(baseline_name)
        sorted_keys.append(baseline_name)

    print(f"\n{'='*65}")
    print(f"{'Method':<15} | {'Mean ± Std':<20} | {'Gap (%)':<10} | {'p-value':<10}")
    print(f"{'-'*65}")

    for name in sorted_keys:
        values = data_dict[name]
        
        # 1. 计算 Mean 和 Std
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # 2. 计算 Gap (%)
        # 公式: (Current - Baseline) / Baseline * 100
        # 如果是 Makespan 问题，正值代表比 Baseline 差，负值代表比 Baseline 好
        gap = ((mean_val - baseline_mean) / baseline_mean) * 100
        
        # 3. 计算 p-value (配对 T 检验)
        if name == baseline_name:
            p_value_str = "-"
            gap_str = "-"
        else:
            # ttest_rel 用于配对样本 (即 100 个算例是一一对应的)
            t_stat, p_val = stats.ttest_rel(values, baseline_data)
            
            # 格式化 p-value
            if p_val < 0.001:
                p_value_str = "< 0.001"
                # p_value_str = f"{p_val:.1e}"
            else:
                p_value_str = f"{p_val:.3f}"
            
            gap_str = f"+{gap:.1f}" if gap > 0 else f"{gap:.1f}"

        # 格式化 Mean ± Std
        mean_std_str = f"{mean_val:.1f} ± {std_val:.1f}"
        
        # 打印行
        print(f"{name:<15} | {mean_std_str:<20} | {gap_str:<10} | {p_value_str:<10}")
        
        results.append({
            "Method": name,
            "Mean": mean_val,
            "Std": std_val,
            "Gap(%)": gap if name != baseline_name else 0,
            "p-value": p_val if name != baseline_name else 1.0
        })
    
    print(f"{'='*65}\n")
    return results

if __name__ == "__main__":
    # 1. 加载数据
    print("正在加载数据...")
    all_data = load_data(FILES)
    
    # 2. 计算并打印表格
    if all_data:
        calculate_metrics(all_data, OUR_METHOD_NAME)
    else:
        print("未加载到任何数据，请检查路径。")