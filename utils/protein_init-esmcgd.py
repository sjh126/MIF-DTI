import numpy as np
import pandas as pd
import sys
import torch.nn.functional as F
# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import torch
from torch_geometric.utils import degree, add_self_loops, subgraph, to_undirected, remove_self_loops, coalesce
import math
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig




# normalize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def seq_feature(pro_seq):    
    if 'U' in pro_seq or 'B' in pro_seq:
        print('U or B in Sequence')
    pro_seq = pro_seq.replace('U','X').replace('B','X')
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def contact_map(contact_prob_map, threshold=0.5):
    device = contact_prob_map.device
    prot_contact_adj = (contact_prob_map >= threshold).long()
    edge_index = prot_contact_adj.nonzero(as_tuple=False).t().contiguous()
    row, col = edge_index
    edge_weight = contact_prob_map[row, col].float()

    # 1-step/2-step 序列连接
    seq_idx = torch.arange(contact_prob_map.size(0), device=device)
    seq_edge_head1 = torch.stack([seq_idx[:-1], (seq_idx+1)[:-1]], dim=0)
    seq_edge_tail1 = torch.stack([seq_idx[1:], (seq_idx-1)[1:]], dim=0)
    seq_edge_weight1 = torch.ones(seq_edge_head1.size(1) + seq_edge_tail1.size(1), device=device) * threshold
    edge_index = torch.cat([edge_index, seq_edge_head1, seq_edge_tail1], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight1], dim=-1)

    seq_edge_head2 = torch.stack([seq_idx[:-2], (seq_idx+2)[:-2]], dim=0)
    seq_edge_tail2 = torch.stack([seq_idx[2:], (seq_idx-2)[2:]], dim=0)
    seq_edge_weight2 = torch.ones(seq_edge_head2.size(1) + seq_edge_tail2.size(1), device=device) * threshold
    edge_index = torch.cat([edge_index, seq_edge_head2, seq_edge_tail2], dim=-1)
    edge_weight = torch.cat([edge_weight, seq_edge_weight2], dim=-1)

    edge_index, edge_weight = coalesce(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = to_undirected(edge_index, edge_weight, reduce='max')
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1)
    return edge_index, edge_weight
# ==========================================
# esm_extract (保留原名)
# ==========================================
def esm_extract(model, seq, dim=1152, approach='last', window_size=350, overlap=50,
                contact_threshold=0.75, top_k=10):
    """
    ESMC版本的 esm_extract，生成 O(L) sparse graph。
    token_representation: [L, D]
    logits_tensor: [L, D] （可用于其他任务）
    edge_index: [2, E]
    edge_weight: [E]
    """
    import torch
    import torch.nn.functional as F
    from esm.sdk.api import ESMProtein, LogitsConfig
    from torch_geometric.utils import coalesce, to_undirected, remove_self_loops, add_self_loops
    import math

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = len(seq)

    # token representation 初始化
    token_representation = torch.zeros((seq_len, dim), device=device)
    count_matrix = torch.zeros((seq_len, dim), device=device)

    step = window_size - overlap
    num_windows = math.ceil(seq_len / step)

    # sparse graph edge list
    edge_index_list = []
    edge_weight_list = []

    for i in range(num_windows):
        start = i * step
        end = min(start + window_size, seq_len)
        sub_seq = seq[start:end]

        # ESMC encode
        protein = ESMProtein(sequence=sub_seq)
        protein_tensor = model.encode(protein).to(device)
        with torch.no_grad():
            logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            embeddings = logits_output.embeddings[:, 1:-1, :][0]  # [L_window, D]

        # token embeddings
        if approach == 'last':
            sub_repr = embeddings
        elif approach == 'sum':
            sub_repr = embeddings.sum(dim=0, keepdim=True).expand(end-start, dim)
        elif approach == 'mean':
            sub_repr = embeddings.mean(dim=0, keepdim=True).expand(end-start, dim)
        else:
            raise ValueError("approach must be one of ['last','sum','mean']")
        token_representation[start:end] += sub_repr
        count_matrix[start:end] += 1

        # 局部 top-k neighbors
        emb_norm = F.normalize(embeddings, dim=1)
        Lw = end - start
        for idx in range(Lw):
            # 局部窗口 top-k
            win_start = max(0, idx - 5)
            win_end = min(Lw, idx + 6)
            local_emb = emb_norm[win_start:win_end]  # [win_len, D]
            scores = (local_emb @ emb_norm[idx].T + 1.0)/2.0  # [win_len]
            # top-k
            k_val = min(top_k, scores.size(0))
            top_val, top_idx = torch.topk(scores, k=k_val)
            for j, w in zip(top_idx.tolist(), top_val.tolist()):
                neighbor_idx = win_start + j
                if neighbor_idx != idx:
                    edge_index_list.append([start+idx, start+neighbor_idx])
                    edge_weight_list.append(w)

    # token 平均
    token_representation /= count_matrix.clamp(min=1e-6)
    logits_tensor = token_representation.clone()

    # 添加 backbone 1-step / 2-step
    seq_idx = torch.arange(seq_len, device=device)
    for step in [1, 2]:
        head = torch.stack([seq_idx[:-step], seq_idx[step:]], dim=0)
        tail = torch.stack([seq_idx[step:], seq_idx[:-step]], dim=0)
        edge_index_list.extend(head.t().tolist())
        edge_index_list.extend(tail.t().tolist())
        edge_weight_list.extend([1.0]*(head.size(1)+tail.size(1)))

    # PyG 后处理
    if len(edge_index_list) == 0:
        edge_index = torch.zeros((2,0), dtype=torch.long, device=device)
        edge_weight = torch.zeros((0,), dtype=torch.float32, device=device)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long, device=device).t()
        edge_weight = torch.tensor(edge_weight_list, dtype=torch.float32, device=device)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes=seq_len, reduce='max')
        edge_index, edge_weight = to_undirected(edge_index, edge_weight, num_nodes=seq_len, reduce='max')
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1.0)

    return token_representation, logits_tensor, edge_index, edge_weight

def protein_init(seqs):
    """
    初始化蛋白质数据，返回 token_representation + sparse graph
    """
    result_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESMC.from_pretrained("esmc_600m").to(device)

    for seq in tqdm(seqs, desc="Processing proteins"):
        seq_feat = seq_feature(seq)  # 你原来的 seq_feature
        token_repr, logits_tensor, edge_index, edge_weight = esm_extract(
            model, seq, dim=1152, approach='last'
        )

        result_dict[seq] = {
            'seq': seq,
            'seq_feat': torch.from_numpy(seq_feat),
            'token_representation': token_repr.half(),
            'num_nodes': len(seq),
            'num_pos': torch.arange(len(seq), device=device).reshape(-1,1),
            'edge_index': edge_index,
            'edge_weight': edge_weight,
        }

    return result_dict

def generate_ESM_structure(model, filename, sequence):
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on chunk_size', chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None


from Bio.PDB import PDBParser
biopython_parser = PDBParser()

one_to_three = {"A" : "ALA",
              "C" : "CYS",
              "D" : "ASP",
              "E" : "GLU",
              "F" : "PHE",
              "G" : "GLY",
              "H" : "HIS",
              "I" : "ILE",
              "K" : "LYS",
              "L" : "LEU",
              "M" : "MET",
              "N" : "ASN",
              "P" : "PRO",
              "Q" : "GLN",
              "R" : "ARG",
              "S" : "SER",
              "T" : "THR",
              "V" : "VAL",
              "W" : "TRP",
              "Y" : "TYR",
              "B" : "ASX",
              "Z" : "GLX",
              "X" : "UNK",
              "*" : " * "}

three_to_one = {}
for _key, _value in one_to_three.items():
    three_to_one[_value] = _key
three_to_one["SEC"] = "C"
three_to_one["MSE"] = "M"


def extract_pdb_seq(protein_path):

    structure = biopython_parser.get_structure('random_id', protein_path)[0]
    seq = ''
    chain_str = ''
    for i, chain in enumerate(structure):
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not
                try:
                    seq += three_to_one[residue.get_resname()]
                    chain_str += str(chain.id)
                except Exception as e:
                    seq += 'X'
                    chain_str += str(chain.id)
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex. Replacing it with a dash X.')
        
    return seq, chain_str