# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

# ===== 字典定义 =====
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

# ===== 序列编码函数 =====
def label_smiles(line, smi_ch_ind=CHARISOSMISET, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        if ch in smi_ch_ind:
            X[i] = smi_ch_ind[ch]
    return X

def label_sequence(line, smi_ch_ind=CHARPROTSET, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        if ch in smi_ch_ind:
            X[i] = smi_ch_ind[ch]
    return X


# ===== 数据集路径 =====
dataset_path = '/home/sjh/DTI/MIF-DTI/DataSets/BIOSNAP.txt'

# ===== 读取药物和蛋白序列 =====
drugs, prots = [], []
with open(dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        drug = parts[2]
        prot = parts[3]
        drugs.append(drug)
        prots.append(prot)

# 去重
drugs = list(set(drugs))
prots = list(set(prots))
print(f"药物数量（去重后）: {len(drugs)}")
print(f"蛋白数量（去重后）: {len(prots)}")


# ===== 计算字符长度 =====
drug_lens = [len(d) for d in tqdm(drugs, desc="Drug length")]
prot_lens = [len(p) for p in tqdm(prots, desc="Protein length")]


# ===== 输出统计结果 =====
def show_stats(name, lens):
    arr = np.array(lens)
    print(f"\n{name} 长度统计（按字符）:")
    print(f"  平均长度: {arr.mean():.2f}")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  {p}% 百分位: {np.percentile(arr, p):.0f}")

show_stats("Drug", drug_lens)
show_stats("Protein", prot_lens)
