import pandas as pd
import numpy as np
from scipy import stats
import os

# ================= 配置区域 =================
# 你只需要修改这里的路径，每次运行对应一个规模
BASE_DIR = "./newout/3010"  

FILES = {
    "MA-Trans": f"{BASE_DIR}/MA-Trans/test.csv", # 基准 (Gap=0, p-val=-)
    "ResGAT": f"{BASE_DIR}/ResGAT/test.csv",
    "LSTM-Ptr": f"{BASE_DIR}/LSTM/test.csv",
    "MWKR-EAT": f"{BASE_DIR}/MWKR/test.csv",
}

# 想要在表格中显示的顺序
ORDER = ["MWKR-EAT", "ResGAT", "LSTM-Ptr", "MA-Trans"]

# ================= 计算逻辑 =================
def load_and_calc():
    data = {}
    print(f"📊 Processing directory: {BASE_DIR}")
    
    # 1. 读取数据
    for name, path in FILES.items():
        if not os.path.exists(path):
            print(f"   ❌ File not found: {path}")
            return
        try:
            # 兼容带header和不带header的情况
            df = pd.read_csv(path, header=None)
            if isinstance(df.iloc[0,0], str): 
                df = pd.read_csv(path)
            # 确保取前100个浮点数
            data[name] = df.values.flatten().astype(float)[:100]
        except Exception as e:
            print(f"   ❌ Error reading {name}: {e}")
            return

    # 2. 确定基准 (MA-Trans)
    if "MA-Trans" not in data:
        print("   ❌ MA-Trans data missing, cannot calc Gap/P-value.")
        return

    base_vals = data["MA-Trans"]
    base_mean = np.mean(base_vals)

    print("-" * 75)
    print(f"{'Method':<12} | {'Mean ± Std':<18} | {'Gap (%)':<10} | {'p-value':<10}")
    print("-" * 75)
    
    latex_rows = []

    for method in ORDER:
        if method not in data: continue
        
        vals = data[method]
        mean = np.mean(vals)
        std = np.std(vals)
        
        # --- Gap 计算 (统一逻辑: (Method - Ours) / Ours) ---
        # 结果为正数，表示该方法比 MA-Trans 慢了多少
        if method == "MA-Trans":
            gap_str = "--"
        else:
            gap = ((mean - base_mean) / base_mean) * 100
            gap_str = f"+{gap:.1f}" # 强制加号，保持队形

        # --- P-value 计算 (配对 t 检验) ---
        if method == "MA-Trans":
            p_str = "--"
        else:
            try:
                # 配对检验要求长度一致
                t_stat, p_val = stats.ttest_rel(vals, base_vals)
                if p_val < 0.001:
                    p_str = "< 0.001"
                elif p_val < 0.05:
                    p_str = f"{p_val:.3f}"
                else:
                    p_str = f"{p_val:.3f}" # 不显著也显示数值
            except:
                p_str = "NaN"

        # 打印控制台表格
        print(f"{method:<12} | {mean:.1f} ± {std:.1f}      | {gap_str:<10} | {p_str:<10}")
        
        # 生成 LaTeX 字符串
        # 格式: Name & Mean \pm Std & Gap & P-val \\
        latex_row = f"& {method} & {mean:.1f} $\\pm$ {std:.1f} & {gap_str} & {p_str} \\\\"
        latex_rows.append(latex_row)

    print("-" * 75)
    print("📋 LaTeX Code Snippet (Copy to your table):")
    for row in latex_rows:
        print(row)
    print("\n")

if __name__ == "__main__":
    load_and_calc()