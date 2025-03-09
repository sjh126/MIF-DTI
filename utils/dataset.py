import torch.utils.data
from torch_geometric.data import Dataset
import torch
from torch_geometric.data import Data
import pickle
import torch.utils.data
import numpy as np
from utils.DataSetsFunction import label_sequence, label_smiles


def get_m2p_edge(mol_x, prot_x, mol_node_level=None):
    """ Construct edges from atoms to amino acids """

    x1 = np.arange(0, mol_x.shape[0])
    x1 = x1[mol_node_level==1] if mol_node_level is not None else x1
    x2 = np.arange(0, prot_x.shape[0])
    
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    edge_list = torch.LongTensor(np.vstack([grid_x1.ravel(), grid_x2.ravel()]))
    
    return edge_list

class ProteinMoleculeDataset(Dataset):
    def __init__(self, sequence_data, mol_obj, prot_obj, device='cpu'):
        super(ProteinMoleculeDataset, self).__init__()

        if isinstance(sequence_data,list):
            self.pairs = sequence_data
        else:
            raise Exception("provide list object")
        
        ## MOLECULES
        if isinstance(mol_obj, dict):
            self.mols = mol_obj
        elif isinstance(mol_obj, str):
            with open(mol_obj, 'rb') as f:
                self.mols = pickle.load(f)
        else:
            raise Exception("provide dict mol object or pickle path")


        ## PROTEINS
        if isinstance(prot_obj, dict):
            self.prots = prot_obj
        elif isinstance(prot_obj, str):
            self.prots = torch.load(prot_obj)
        else:
            raise Exception("provide dict mol object or pickle path")

        self.device = device

        for _, v in self.mols.items():
            v['atom_idx'] = v['atom_idx'].long().view(-1, 1)
            v['atom_feature'] = v['atom_feature'].float()
            adj = v['bond_feature'].long()
            mol_edge_index =  adj.nonzero(as_tuple=False).t().contiguous()
            v['atom_edge_index'] = mol_edge_index
            v['atom_edge_attr'] = adj[mol_edge_index[0], mol_edge_index[1]].long()
            v['atom_num_nodes'] = v['atom_idx'].shape[0]
            v['smiles_x'] = torch.tensor(label_smiles(v['smiles'], MAX_SMI_LEN=200)).reshape(1, -1)


        for _, v in self.prots.items():
            v['seq_feat'] = v['seq_feat'].float()
            v['token_representation'] = v['token_representation'].float()
            v['num_nodes'] = len(v['seq'])
            v['node_pos'] = torch.arange(len(v['seq'])).reshape(-1,1)
            v['edge_weight'] = v['edge_weight'].float()
            v['seq_x'] = torch.tensor(label_sequence(v['seq'], MAX_SEQ_LEN=1500)).reshape(1, -1)

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return self.__len__()
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Extract data
        items = self.pairs[idx].split(' ')
        mol_key, prot_key = items[-3], items[-2]
        cls_y = torch.tensor(int(items[-1])).long()
            
        mol = self.mols[mol_key]
        prot = self.prots[prot_key]
        
        ## atom
        mol_x = mol['atom_idx']
        mol_x_feat = mol['atom_feature']
        mol_edge_index  = mol['atom_edge_index']
        mol_edge_attr = mol['atom_edge_attr']
        mol_num_nodes = mol['atom_num_nodes']
        mol_smiles_x = mol['smiles_x']

        ## Prot
        prot_seq = prot['seq']
        prot_seq_x = prot['seq_x']
        prot_node_aa = prot['seq_feat']
        prot_node_evo = prot['token_representation']
        prot_num_nodes = prot['num_nodes']
        prot_node_pos = prot['node_pos']
        prot_edge_index = prot['edge_index']
        prot_edge_weight = prot['edge_weight']

        out = MultiGraphData(
                ## MOLECULE
                mol_x=mol_x, mol_smiles_x=mol_smiles_x, mol_x_feat=mol_x_feat, mol_edge_index=mol_edge_index,
                mol_edge_attr=mol_edge_attr, mol_num_nodes= mol_num_nodes, mol_node_levels=mol['node_levels'],
                ## PROTEIN
                prot_node_aa=prot_node_aa, prot_node_evo=prot_node_evo, prot_seq_x=prot_seq_x,
                prot_node_pos=prot_node_pos, prot_seq=prot_seq,
                prot_edge_index=prot_edge_index, prot_edge_weight=prot_edge_weight,
                prot_num_nodes=prot_num_nodes,
                ## Y output
                cls_y=cls_y,
                ## keys
                mol_key = mol_key, prot_key = prot_key,
                # BI GRAPH
                m2p_edge_index = get_m2p_edge(mol_x, prot_node_aa, mol['node_levels'])
        )

        return out

def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else int(num_nodes)

def get_self_loop_attr(edge_index, edge_attr, num_nodes):
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    .. note::
        This operation is analogous to getting the diagonal elements of the
        dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = torch.tensor([0.2, 0.3, 0.5])
        >>> get_self_loop_attr(edge_index, edge_weight)
        tensor([0.5000, 0.0000])

        >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
        tensor([0.5000, 0.0000, 0.0000, 0.0000])
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = torch.ones_like(loop_index, dtype=torch.float)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = loop_attr.new_zeros((num_nodes, ) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr



class MultiGraphData(Data):
    def __inc__(self, key, item, *args):
        if key == 'mol_edge_index':
            return self.mol_x.size(0)
        elif key == 'clique_edge_index':
            return self.clique_x.size(0)
        elif key == 'atom2clique_index':
            return torch.tensor([[self.mol_x.size(0)], [self.clique_x.size(0)]])
        elif key == 'prot_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'prot_struc_edge_index':
            return self.prot_node_aa.size(0)
        elif key == 'm2p_edge_index':
            return torch.tensor([[self.mol_x.size(0)], [self.prot_node_aa.size(0)]])
        else:
            return super(MultiGraphData, self).__inc__(key, item, *args)

