import pandas as pd
import numpy as np
import os

# === 配置路径 ===
BASE_DIR = "./20-10-5"  
FILES = {
    "MA-Trans": f"{BASE_DIR}/transformer/gpu.csv", 
    "ResGAT":   f"{BASE_DIR}/gnn/gpu.csv",
    "LSTM-Ptr": f"{BASE_DIR}/lstm/gpu.csv"
}

# === 针对不同模型的分割阈值 (基于你之前发的数据规律) ===
# 逻辑：大于阈值判定为训练，小于阈值判定为推理
THRESHOLDS = {
    "MA-Trans": 50.0,  # Trans: 推理~20%, 训练~84%. 阈值50很安全
    "ResGAT":   28.0,  # GNN: 推理~19%, 训练30-60%. 阈值28能分开
    "LSTM-Ptr": 80.0   # LSTM: 推理~67%, 训练~90%. 阈值80能分开
}

def analyze_file(model_name, file_path):
    print(f"--- Analyzing {model_name} ---")
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    try:
        # 1. 读取数据 (处理 % 符号)
        df = pd.read_csv(file_path)
        # 假设第一列是数据，列名包含 utilization.gpu
        col_name = df.columns[0]
        # 去掉 % 并转为 float
        data = df[col_name].astype(str).str.replace(' %', '').astype(float)
        
        # 2. 数据分离
        threshold = THRESHOLDS[model_name]
        
        # 推理数据 (Inference)
        inf_data = data[data < threshold]
        
        # 训练数据 (Training)
        train_data = data[data >= threshold]
        
        # 3. 计算统计量
        inf_avg = inf_data.mean() if len(inf_data) > 0 else 0
        train_avg = train_data.mean() if len(train_data) > 0 else 0
        train_peak = train_data.max() if len(train_data) > 0 else 0
        train_std = train_data.std() if len(train_data) > 0 else 0
        
        print(f"样本总数: {len(data)}")
        print(f"推理阶段 (Samples: {len(inf_data)}): Avg = {inf_avg:.2f}%")
        print(f"训练阶段 (Samples: {len(train_data)}): Avg = {train_avg:.2f}%, Peak = {train_peak:.2f}%, Std = {train_std:.2f}")
        
        return {
            "Model": model_name,
            "Inference Avg (%)": f"{inf_avg:.1f}",
            "Training Avg (%)": f"{train_avg:.1f}",
            "Training Peak (%)": f"{train_peak:.1f}",
            "Stability (Std)": f"{train_std:.1f}"
        }

    except Exception as e:
        print(f"❌ 处理出错: {e}")
        return None

# === 主程序 ===
results = []
for model, path in FILES.items():
    res = analyze_file(model, path)
    if res:
        results.append(res)

# === 生成最终表格数据 ===
print("\n" + "="*50)
print("✅ 最终表格数据 (可以直接填入论文)")
print("="*50)
res_df = pd.DataFrame(results)
print(res_df)
# 导出为 Excel 或 CSV 方便复制
# res_df.to_csv("final_gpu_stats.csv", index=False)