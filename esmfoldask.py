import os
import json
from biolmai import BioLM

# 设置 token
os.environ["BIOLMAI_TOKEN"] = "1228aef5249a765a350e6f4ad3d05798961ffeec711bdb890f108bdb20a35c4b"

# davis.txt 文件路径
input_file = "/home/sjh/DTI/MIF-DTI/DataSets/Davis.txt"
output_dir = "/home/sjh/DTI/MIF-DTI/DataSets/Preprocessed/Davis_target_pdb"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取蛋白质ID和序列，去重
proteins_dict = {}
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        protein_id = parts[1]           # 第二列蛋白质ID
        sequence = parts[-2]            # 倒数第二列蛋白质序列

        # 如果有重复ID，保持第一次出现的序列
        if protein_id not in proteins_dict:
            proteins_dict[protein_id] = sequence

# 转成列表方便循环
proteins = list(proteins_dict.items())

# 总蛋白质数量
n = len(proteins)
print(f"Total unique proteins: {n}")

# ----------------- [添加以下代码] -----------------

# 计算并输出最大序列长度
if proteins: # 确保列表不为空
    # (protein_id, sequence)
    max_sequence_length = max(len(seq) for pid, seq in proteins)
    print(f"Max sequence length: {max_sequence_length}")

# ---------------------------------------------------

# 循环批量推理
# for idx, (protein_id, sequence) in enumerate(proteins, start=1):
#     print(sequence)
#     print(f"[{idx}/{n}] Predicting {protein_id} (Length: {len(sequence)}) ...")
#     try:
#         result = BioLM(entity="esmfold", action="predict", items=[{"sequence": sequence}])
#         pdb_text = result['results'][0]['pdb']
#         pdb_text="1"
#         # 保存为 pdb 文件
#         pdb_file = os.path.join(output_dir, f"{protein_id}.pdb")
#         with open(pdb_file, "w") as f:
#             f.write(pdb_text)
#         print(f"Saved {pdb_file}")

#     except Exception as e:
#         print(f"Error predicting {protein_id}: {e}")
