import torch
from torch.utils.data import Dataset
import numpy as np
from subword_nmt.apply_bpe import BPE
import codecs
import pandas as pd
from subword_nmt.apply_bpe import BPE
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


def label_smiles(line, smi_ch_ind=CHARISOSMISET, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, smi_ch_ind=CHARPROTSET, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


import codecs
import pandas as pd
from subword_nmt.apply_bpe import BPE

# ===== Protein BPE =====
vocab_path_p = './ESPF/protein_codes_uniprot.txt'
bpe_codes_protein = codecs.open(vocab_path_p)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')

sub_csv_p = pd.read_csv('./ESPF/subword_units_map_uniprot.csv')
idx2word_p = sub_csv_p['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

# ===== Drug BPE =====
vocab_path_d = './ESPF/drug_codes_chembl.txt'
bpe_codes_drug = codecs.open(vocab_path_d)
dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

sub_csv_d = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
idx2word_d = sub_csv_d['index'].values
words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

def label_smiles_bpe(smiles, max_len=50):
    """将 SMILES 转为 BPE 子词索引序列"""
    tokens = dbpe.process_line(smiles).split()  # 用 drug BPE 切分
    try:
        ids = np.asarray([words2idx_d[tok] for tok in tokens])
    except KeyError:
        ids = np.array([0])  # 未知token用0代替

    # padding / 截断
    if len(ids) < max_len:
        ids = np.pad(ids, (0, max_len - len(ids)), constant_values=0)
    else:
        ids = ids[:max_len]
    return ids


def label_sequence_bpe(seq, max_len=600):
    """将蛋白序列转为 BPE 子词索引序列"""
    tokens = pbpe.process_line(seq).split()  # 用 protein BPE 切分
    try:
        ids = np.asarray([words2idx_p[tok] for tok in tokens])
    except KeyError:
        ids = np.array([0])

    # padding / 截断
    if len(ids) < max_len:
        ids = np.pad(ids, (0, max_len - len(ids)), constant_values=0)
    else:
        ids = ids[:max_len]
    return ids

vocab_size_drug = len(words2idx_d)
vocab_size_prot = len(words2idx_p)
print("""vocab_size_drug: """,vocab_size_drug)
print("""vocab_size_prot: """,vocab_size_prot)