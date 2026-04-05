# -*- coding: utf-8 -*-
"""
protein_init.py
ESM2 + ESMFold sliding-window contact stitching implementation.
"""

import os
import math
import gc
import numpy as np
import torch
import esm
from tqdm import tqdm
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from torch_geometric.utils import to_undirected, add_self_loops, coalesce

# -------------------------
# 你的氨基酸辅助表与工具函数（保持不变或可改）
# -------------------------
def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in list(dic.keys()):
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14, 'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13, 'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36, 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21, 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17, 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13, 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00, 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00, 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59, 'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65, 'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100, 'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7, 'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99, 'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5, 'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0,
                     1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                     res_pkx_table[residue], res_pl_table[residue],
                     res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f'input {x} not in allowable set{allowable_set}:')
    return list(map(lambda s: x == s, allowable_set))


def seq_feature(pro_seq):
    if 'U' in pro_seq or 'B' in pro_seq:
        print('U or B in Sequence')
    pro_seq = pro_seq.replace('U', 'X').replace('B', 'X')
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i, ] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i, ] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


# -------------------------
#  配置（可调）
# -------------------------
MAX_ESMFOLD_LEN = 1200        # > 这长度就尽量不要用全局 ESMFold；分块也可能更稳
FOLD_WINDOW = 600             # ESMFold 分块大小（推荐 400~800）
FOLD_OVERLAP = 150            # 分块之间重叠长度（用以平滑拼接）
ESMFOLD_CHUNK_TRIES = [64, 32, 16]  # 每块尝试的 chunk_size，从大到小尝试以节省显存
ESM2_CONTACT_THRESHOLD = 0.5
SEQ_WINDOW = 8                # ultimate fallback: sequence window adjacency
TEMP_PDB_DIR = "./temp_pdbs_for_fold"

# -------------------------
#  工具：从 PDB 生成 contact map（CA）
# -------------------------
pdb_parser = PDBParser(QUIET=True)


def coords_to_contact_map_from_pdb(pdb_file, seq_offset, seq_len, distance_threshold=8.0):
    """
    读取 pdb_file，返回 patch 的 contact_map（numpy array），以及该 patch 对应的 global index offset 和 patch length。
    seq_offset: 该 pdb 对应到全序列的起始 index（包含）
    seq_len: 全序列长度（用于构造全图）
    """
    try:
        structure = pdb_parser.get_structure('p', pdb_file)
    except Exception:
        return None, seq_offset, 0

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    coords.append(residue['CA'].get_coord())
    coords = np.array(coords)
    if coords.shape[0] == 0:
        return None, seq_offset, 0

    # patch length
    patch_len = coords.shape[0]
    dist_matrix = cdist(coords, coords)
    contact = (dist_matrix < distance_threshold).astype(np.float32)
    return contact, seq_offset, patch_len


# -------------------------
#  ESM2 embedding + (optional) contact sliding extraction
# -------------------------
def esmfold_sliding_contacts(esmfold_model, seq, window=FOLD_WINDOW, overlap=FOLD_OVERLAP,
                             chunk_trials=ESMFOLD_CHUNK_TRIES, tmpdir=TEMP_PDB_DIR):
    """
    ESMFold sliding window folding with debug prints.
    Returns contact_map, contact_count (both np arrays).
    """
    os.makedirs(tmpdir, exist_ok=True)
    L = len(seq)
    contact_map = np.zeros((L, L), dtype=np.float32)
    contact_count = np.zeros((L, L), dtype=np.int32)

    step = window - overlap
    i = 0
    block_id = 0
    successful_blocks = 0
    while i < L:
        start = i
        end = min(i + window, L)
        sub_seq = seq[start:end]
        success = False
        plen = 0
        for try_chunk in chunk_trials:
            try:
                esmfold_model.set_chunk_size(try_chunk)
                with torch.no_grad():
                    pdb_out = esmfold_model.infer_pdb(sub_seq)
                pdb_path = os.path.join(tmpdir, f"tmp_block_{block_id}.pdb")
                with open(pdb_path, "w") as f:
                    f.write(pdb_out)
                sub_contact, offset, plen = coords_to_contact_map_from_pdb(pdb_path, start, L)
                # cleanup
                try:
                    os.remove(pdb_path)
                except Exception:
                    pass
                if sub_contact is not None and plen > 0:
                    contact_map[start:start + plen, start:start + plen] += sub_contact
                    contact_count[start:start + plen, start:start + plen] += 1
                    success = True
                    successful_blocks += 1
                    print(f"[ESMFOLD][BLOCK {block_id}] chunk={try_chunk} SUCCESS start={start} end={end} plen={plen} contact_sum={sub_contact.sum():.4f}", flush=True)
                else:
                    print(f"[ESMFOLD][BLOCK {block_id}] chunk={try_chunk} produced empty contact or plen=0", flush=True)
                break
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cuda out of memory" in msg:
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"[ESMFOLD][BLOCK {block_id}] chunk={try_chunk} OOM -> try smaller chunk", flush=True)
                    continue
                else:
                    print(f"[ESMFOLD][BLOCK {block_id}] chunk={try_chunk} RUNTIME ERROR (re-raise): {str(e)[:200]}", flush=True)
                    raise e
        if not success:
            print(f"[ESMFOLD][BLOCK {block_id}] ALL chunk_trials FAILED or produced no contact -> leaving zeros for this block (start={start}, end={end})", flush=True)

        if end == L:
            break
        i += step
        block_id += 1

    mask = contact_count > 0
    contact_map[mask] = contact_map[mask] / contact_count[mask]
    print(f"[ESMFOLD][SUMMARY] L={L}, blocks={block_id+1}, successful_blocks={successful_blocks}, total_contact_sum={contact_map.sum():.4f}, covered_cells={mask.sum()}", flush=True)
    return contact_map, contact_count



# -------------------------
#  将 contact_map 转为 edge_index/edge_weight（PyG）
# -------------------------
def contactmap_to_edge(contact_map_np, threshold=ESM2_CONTACT_THRESHOLD, device=torch.device('cpu')):
    L = contact_map_np.shape[0]
    if L == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.float32, device=device)
    prot_contact_adj = (contact_map_np >= threshold).astype(np.int64)
    idxs = np.where(prot_contact_adj == 1)
    if idxs[0].size == 0:
        # fallback create window adjacency
        return build_sequence_window_adj(L, window=SEQ_WINDOW, device=device)
    edge_index = torch.from_numpy(np.vstack(idxs)).long().to(device)
    row = idxs[0]; col = idxs[1]
    edge_weight = torch.from_numpy(contact_map_np[row, col]).float().to(device)
    # normalize using coalesce / to_undirected / add_self_loops
    edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes=L, reduce='max')
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, num_nodes=L, reduce='max')
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1.0, num_nodes=L)
    return edge_index, edge_weight


def build_sequence_window_adj(seq_len, window=SEQ_WINDOW, device=torch.device('cpu')):
    if seq_len == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.float32, device=device)
    rows = []
    cols = []
    weights = []
    for i in range(seq_len):
        rows.append(i); cols.append(i); weights.append(1.0)
        start = max(0, i - window)
        end = min(seq_len - 1, i + window)
        for j in range(start, end + 1):
            if j == i:
                continue
            rows.append(i); cols.append(j); weights.append(1.0 / (abs(i - j) + 1e-6))
    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
    edge_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes=seq_len, reduce='max')
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, num_nodes=seq_len, reduce='max')
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1.0, num_nodes=seq_len)
    return edge_index, edge_weight


# -------------------------
#  主函数：protein_init
# -------------------------
def protein_init(seqs):
    """
    输入: seqs (list of strings)
    输出: result_dict: {
        seq: {
            'seq',
            'seq_feat',             # float32 Tensor, device=device
            'token_representation', # float32 Tensor, device=device
            'num_nodes',
            'num_pos',              # Tensor, device=device
            'edge_index',           # Tensor, device=device
            'edge_weight'           # Tensor, device=device
        }
    }
    """
    print("Loading ESM-2 model (esm2_t33_650M_UR50D)...")
    esm2_model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    print("Loading ESMFold model (v1)...")
    esmfold_model = esm.pretrained.esmfold_v1()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm2_model.eval().to(device)
    esmfold_model.eval().to(device)

    batch_converter = alphabet.get_batch_converter()
    
    os.makedirs(TEMP_PDB_DIR, exist_ok=True)

    result_dict = {}
    for i, seq in enumerate(tqdm(seqs, desc="Processing Proteins")):
        seq_len = len(seq)
        # 1) seq_feat -> float32 Tensor on device
        seq_feat_np = seq_feature(seq)
        seq_feat_t = torch.from_numpy(seq_feat_np).float().to(device)

        # 2) ESM2 token embeddings
        try:
            token_repr_np, esm2_contact_np = esm2_sliding_extract(
                esm2_model, batch_converter, seq, layer=33,
                window=350, overlap=50, device=device
            )
        except Exception:
            torch.cuda.empty_cache()
            gc.collect()
            token_repr_np = np.zeros((seq_len, 1280), dtype=np.float32)
            esm2_contact_np = np.zeros((seq_len, seq_len), dtype=np.float32)

        token_repr_t = torch.from_numpy(token_repr_np).float().to(device)

        # 3) ESMFold sliding contacts (unchanged calling code)
        use_fold = seq_len <= (MAX_ESMFOLD_LEN * 2)
        final_contact = None
        fold_count = None
        fold_contact_np = None
        if use_fold:
            try:
                fold_contact_np, fold_count = esmfold_sliding_contacts(
                    esmfold_model, seq,
                    window=FOLD_WINDOW, overlap=FOLD_OVERLAP,
                    chunk_trials=ESMFOLD_CHUNK_TRIES, tmpdir=TEMP_PDB_DIR
                )
            except Exception:
                torch.cuda.empty_cache()
                gc.collect()
                fold_contact_np = None
                fold_count = None

        # 4) diagnostic & fallback decision
        def _density(cm):
            if cm is None:
                return 0.0
            Lc = cm.shape[0]
            if Lc == 0:
                return 0.0
            return cm.sum() / (Lc*Lc)

        def _jaccard(a,b,th_a=0.5,th_b=0.5):
            if a is None or b is None:
                return 0.0
            ba = (a >= th_a)
            bb = (b >= th_b)
            inter = np.logical_and(ba, bb).sum()
            union = np.logical_or(ba, bb).sum()
            return inter/union if union>0 else 0.0

        esm2_density = _density(esm2_contact_np)
        fold_density = _density(fold_contact_np)
        jacc = _jaccard(esm2_contact_np, fold_contact_np)
        fold_cov_frac = 0.0
        if fold_count is not None:
            diag_counts = np.diag(fold_count)
            fold_cov_frac = float(np.sum(diag_counts > 0)) / float(len(diag_counts)) if len(diag_counts)>0 else 0.0

        print(f"[DIAG][SEQ {i}] len={seq_len} esm2_density={esm2_density:.6f} fold_density={fold_density:.6f} jaccard={jacc:.4f} fold_coverage_frac={fold_cov_frac:.4f}", flush=True)

        # Decide final_contact: prefer fold only if it has meaningful coverage, else esm2
        if fold_contact_np is not None and (fold_cov_frac > 0.1) and (fold_density > 0):
            final_contact = fold_contact_np
            print(f"[DIAG][SEQ {i}] Using ESMFold contact (coverage_frac={fold_cov_frac:.3f})", flush=True)
        else:
            if esm2_contact_np is not None and np.sum(esm2_contact_np) > 0:
                final_contact = esm2_contact_np
                print(f"[DIAG][SEQ {i}] Using ESM2 contact (fold not used or insufficient coverage)", flush=True)
            else:
                # sequence-window fallback
                final_contact = np.zeros((seq_len, seq_len), dtype=np.float32)
                for a in range(seq_len):
                    for b in range(max(0, a - SEQ_WINDOW), min(seq_len, a + SEQ_WINDOW + 1)):
                        final_contact[a, b] = 1.0 if a == b else 1.0 / (abs(a - b) + 1e-6)
                print(f"[DIAG][SEQ {i}] Using SEQ-WINDOW fallback (no esm2/fold contacts)", flush=True)


        # 5) contact -> edge_index, edge_weight on device
        edge_index, edge_weight = contactmap_to_edge(final_contact, threshold=ESM2_CONTACT_THRESHOLD, device=device)

        # 6) pack result
        result_dict[seq] = {
            'seq': seq,
            'seq_feat': seq_feat_t,
            'token_representation': token_repr_t,
            'num_nodes': seq_len,
            'num_pos': torch.arange(seq_len, device=device).reshape(-1, 1),
            'edge_index': edge_index
            'edge_weight': edge_weight,
        }

    return result_dict