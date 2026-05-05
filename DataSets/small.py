import random
from collections import defaultdict
#sk-4d8f5ac9ab2e404d80012d0347eec185
input_file = "/home/sjh/DTI/MIF-DTI/DataSets/Davis_row.txt"
output_file = "/home/sjh/DTI/MIF-DTI/DataSets/davis_small.txt"
num_proteins = 10  # 随机选择的蛋白质种类数

# 读取文件
with open(input_file, "r") as f:
    lines = f.readlines()

# 如果文件有表头，去掉表头
header = None
if lines[0].strip().split()[0] in ["CID", "156422"]:  # 根据实际文件判断
    header = lines[0]
    lines = lines[1:]

# 按蛋白质 ID 分类
protein_dict = defaultdict(list)
for line in lines:
    parts = line.strip().split()
    if len(parts) < 4:
        continue
    cid, protein_id, smiles, seq, kd = parts[0], parts[1], parts[2], parts[3], parts[-1]
    protein_dict[protein_id].append(line)

# 随机选择 num_proteins 种蛋白质
if len(protein_dict) < num_proteins:
    raise ValueError(f"Available proteins ({len(protein_dict)}) less than requested ({num_proteins})")

selected_proteins = random.sample(list(protein_dict.keys()), num_proteins)

# 写入新文件
with open(output_file, "w") as f:
    if header:
        f.write(header)
    for protein_id in selected_proteins:
        f.writelines(protein_dict[protein_id])

print(f"Selected {num_proteins} proteins and saved to {output_file}")
