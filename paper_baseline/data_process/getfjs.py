import os
import shutil
import re

# ================= ⚙️ 配置区域 =================
# 1. 你存放那 100 个案例文件的文件夹路径
SOURCE_FOLDER = "./1510"  # 请修改为你实际的路径

# 2. 你想把提取出来的文件存到哪里？
TARGET_FOLDER = "./best_case_visualization"

# 3. 刚才那个脚本跑出来的最佳 ID (Case ID)
TARGET_ID = 30  # 举例：如果刚才算出是 42，就填 42

# 4. 文件扩展名过滤 (防止读到 .DS_Store 等垃圾文件)
FILE_EXT = ".fjs"  # 或者是 .pt, .txt, .json，根据你的实际情况修改

# ================= 🚀 执行逻辑 =================
def extract_file():
    # 1. 确保目标文件夹存在
    if not os.path.exists(TARGET_FOLDER):
        os.makedirs(TARGET_FOLDER)
        print(f"📁 已创建目标文件夹: {TARGET_FOLDER}")

    # 2. 获取源文件夹所有文件
    if not os.path.exists(SOURCE_FOLDER):
        print(f"❌ 错误：源文件夹不存在 -> {SOURCE_FOLDER}")
        return

    # 获取所有符合后缀的文件
    all_files = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith(FILE_EXT)]
    
    # 3. 🔥 关键：必须排序！
    # 计算机默认排序可能是 1, 10, 100, 2... 而不是 1, 2, 3...
    # 我们尝试提取文件名中的数字进行自然排序
    def sort_key(fname):
        # 提取文件名里的数字，如果没数字就按字母排
        nums = re.findall(r'\d+', fname)
        if nums:
            return int(nums[0])
        return fname

    sorted_files = sorted(all_files, key=sort_key)

    print(f"👀 扫描到 {len(sorted_files)} 个文件。")
    
    # 检查 ID 是否越界
    if TARGET_ID < 0 or TARGET_ID >= len(sorted_files):
        print(f"❌ ID {TARGET_ID} 超出范围 (0 - {len(sorted_files)-1})")
        return

    # 4. 锁定目标文件
    target_filename = sorted_files[TARGET_ID]
    src_path = os.path.join(SOURCE_FOLDER, target_filename)
    dst_path = os.path.join(TARGET_FOLDER, f"best_case_id{TARGET_ID}_{target_filename}")

    # 5. 复制文件
    try:
        shutil.copy2(src_path, dst_path)
        print("\n" + "="*40)
        print(f"✅ 提取成功！")
        print(f"📄 原始文件: {target_filename}")
        print(f"🎯 对应 CSV 索引: {TARGET_ID}")
        print(f"💾 已保存到: {dst_path}")
        print("="*40 + "\n")
        print("👉 现在你可以直接读取这个文件来画甘特图了。")
    except Exception as e:
        print(f"❌ 复制失败: {e}")

if __name__ == "__main__":
    extract_file()