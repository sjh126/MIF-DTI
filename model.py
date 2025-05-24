# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from layers import *
from torch_geometric.nn import (
                                GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool
                                )
from config import hyperparameter

class MIF_conv_block(nn.Module):
    def __init__(self, in_channels=200, out_channels=200, num_heads=4, dropout=0.3):
        super(MIF_conv_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout

        self.conv = GATConv(self.in_channels, self.out_channels//self.num_heads, self.num_heads, dropout=self.dropout)
        self.norm = LayerNorm(self.in_channels)
        self.readout = SAGPooling(self.out_channels, min_score=-1)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = F.elu(self.norm(x, batch))
        x = self.conv(x, edge_index, edge_attr)
        x, _, _, x_batch, _, _ = self.readout(x, edge_index, edge_attr=edge_attr, batch=batch)
        global_graph_emb = global_add_pool(x, x_batch)
        return x, global_graph_emb


class MIFBlock(nn.Module):
    def __init__(self, in_channels=200, out_channels=200, num_heads=5, dropout=0.4):
        super(MIFBlock, self).__init__()
        
        self.hidden_channels = out_channels // (num_heads*2)
        self.drug_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=0.1)
        self.prot_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=0.3)
        self.inter_conv = GATConv((in_channels, in_channels), self.hidden_channels, num_heads, dropout=dropout)
        self.drug_norm = LayerNorm(out_channels)
        self.prot_norm = LayerNorm(out_channels)
        self.drug_pool = GATConv(out_channels, out_channels//num_heads, num_heads)
        self.prot_pool = SAGPooling(out_channels, min_score=-1)
        # self.prot_pool = GATConv(out_channels, out_channels//num_heads, num_heads)

    def forward(self, atom_x, atom_edge_index, bond_x, atom_batch, \
                aa_x, aa_edge_index, aa_edge_attr, aa_batch, m2p_edge_index):
        
        atom_x_res = atom_x
        aa_x_res = aa_x

        atom_intra_x = self.drug_conv(atom_x, atom_edge_index, bond_x)
        atom_inter_x = self.inter_conv((aa_x, atom_x), m2p_edge_index[[1,0]])
        atom_x_tmp = torch.cat([atom_intra_x, atom_inter_x], -1)
        atom_x = F.elu(self.drug_norm(atom_x_tmp, atom_batch))

        aa_intra_x = self.prot_conv(aa_x, aa_edge_index, aa_edge_attr)
        aa_inter_x = self.inter_conv((atom_x, aa_x), m2p_edge_index)
        aa_x_tmp = torch.cat([aa_intra_x, aa_inter_x], -1)
        aa_x = F.elu(self.prot_norm(aa_x_tmp, aa_batch))

        atom_x = self.drug_pool(atom_x, atom_edge_index, bond_x)
        aa_x, _, _, aa_batch, _, _ = self.prot_pool(aa_x, aa_edge_index, edge_attr=aa_edge_attr, batch=aa_batch)
        # aa_x, aa_edge_index, aa_edge_attr, aa_batch, _, _ = self.prot_pool(aa_x, aa_edge_index, edge_attr=aa_edge_attr, batch=aa_batch)
        # aa_x = self.prot_pool(aa_x, aa_edge_index, aa_edge_attr)
        atom_x = F.dropout(atom_x_res+F.elu(atom_x), 0.1, self.training)
        aa_x = F.dropout(aa_x_res+F.elu(aa_x), 0.1, self.training)
        drug_global_repr = global_add_pool(atom_x, atom_batch)
        prot_global_repr = global_add_pool(aa_x, aa_batch)

        return atom_x, aa_x, drug_global_repr, prot_global_repr

class MIFBlock_1D(nn.Module):
    def __init__(self, input_dim=200, conv=50, drug_kernel=[4, 6, 8], prot_kernel=[4, 8, 12]):
        super(MIFBlock_1D, self).__init__()
        self.attention_dim = conv * 4
        self.mix_attention_head = 5

        self.Drug_CNNs = get_CNNs(input_dim, conv, drug_kernel)
        self.Protein_CNNs = get_CNNs(input_dim, conv, prot_kernel)

        self.mix_attention_layer = nn.MultiheadAttention(self.attention_dim, self.mix_attention_head, batch_first=True, dropout=0.3)

    def forward(self, drugembed, proteinembed):

        # [batch_size, seq_len, embed_dim] -> [batch_size, embed_dim, seq_len] 
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        # [batch_size, embed_dim, seq_len] -> [batch_size, seq_len, embed_dim]
        drugConv = drugConv.permute(0, 2, 1)
        proteinConv = proteinConv.permute(0, 2, 1)

        # cross Attention
        drug_att, _ = self.mix_attention_layer(drugConv, proteinConv, proteinConv)
        protein_att, _ = self.mix_attention_layer(proteinConv, drugConv, drugConv)

        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5

        drugPool, _ = torch.max(drugConv, dim=1)
        proteinPool, _ = torch.max(proteinConv, dim=1)

        return drugConv, proteinConv, drugPool, proteinPool


class MIFDTI(nn.Module):
    def __init__(self, depth=3, device='cuda:0'):
        super(MIFDTI, self).__init__()

        self.drug_in_channels = 43
        self.prot_in_channels = 33
        self.prot_evo_in_channels = 1280
        self.hidden_channels = 200
        self.depth = depth
        self.device = device

        # MOLECULE IN FEAT
        self.atom_type_encoder = Embedding(20, self.hidden_channels)
        self.atom_feat_encoder = MLP([self.drug_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 
        self.bond_encoder = Embedding(10, self.hidden_channels)

        # PROTEIN IN FEAT
        self.prot_evo = MLP([self.prot_evo_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 
        self.prot_aa = MLP([self.prot_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 

        # ENCODER
        self.blocks = nn.ModuleList([MIFBlock() for _ in range(depth)])

        self.drug_seq_emb = nn.Embedding(65, self.hidden_channels, padding_idx=0)
        self.prot_seq_emb = nn.Embedding(26, self.hidden_channels, padding_idx=0)
        self.blocks_1D = nn.ModuleList([MIFBlock_1D() for _ in range(depth)])

        self.attn = RESCAL(self.hidden_channels, self.depth*2)
        # self.attn = PoolAttention(self.hidden_channels)

        self.to(device)

    def forward(self,data):

        # Molecule
        atom_x, atom_x_feat, smiles_x, atom_edge_index, bond_x, mol_node_levels = \
            data.mol_x, data.mol_x_feat, data.mol_smiles_x, data.mol_edge_index, data.mol_edge_attr, data.mol_node_levels
        # Protein (amino acid)
        aa_x, aa_evo_x, seq_x, aa_edge_index, aa_edge_weight = \
            data.prot_node_aa, data.prot_node_evo, data.prot_seq_x, data.prot_edge_index, data.prot_edge_weight, \
        # Batch
        atom_batch, aa_batch = data.mol_x_batch, data.prot_node_aa_batch
        # Bi Graph
        m2p_edge_index = data.m2p_edge_index

        # MOLECULE Featurize
        atom_x = self.atom_type_encoder(atom_x.squeeze()) + self.atom_feat_encoder(atom_x_feat)
        bond_x = self.bond_encoder(bond_x)
                
        # PROTEIN Featurize
        aa_x = self.prot_aa(aa_x) + self.prot_evo(aa_evo_x)
        aa_edge_attr = rbf(aa_edge_weight, D_max=1.0, D_count=self.hidden_channels, device=self.device)

        # Encoding
        drug_repr = []
        prot_repr = []
        for i in range(self.depth):
            out = self.blocks[i](atom_x, atom_edge_index, bond_x, atom_batch, \
                                 aa_x, aa_edge_index, aa_edge_attr, aa_batch, \
                                 m2p_edge_index)
            atom_x, aa_x, drug_global_repr, prot_global_repr = out
            drug_global_repr = atom_x[mol_node_levels==2]
            drug_repr.append(drug_global_repr)
            prot_repr.append(prot_global_repr)

        atom_x_seq = self.drug_seq_emb(smiles_x)
        aa_x_seq = self.prot_seq_emb(seq_x)
        for i in range(self.depth):
            out_seq = self.blocks_1D[i](atom_x_seq, aa_x_seq)
            atom_x_seq, aa_x_seq, drug_seq_pool, prot_seq_pool = out_seq
            drug_repr.append(drug_seq_pool)
            prot_repr.append(prot_seq_pool)

        drug_repr = torch.stack(drug_repr, dim=-2)
        prot_repr = torch.stack(prot_repr, dim=-2)

        # Co-attn
        scores = self.attn(drug_repr, prot_repr)

        return scores

def get_m2p_edge_from_batch(atom_batch, aa_batch, node_level=None):

    mask = atom_batch.unsqueeze(1) == aa_batch.unsqueeze(0)  # (num_a_nodes, num_b_nodes) 的bool矩阵
    if node_level is not None:
        mask = mask * (node_level==1).unsqueeze(1)
    a_idx, b_idx = torch.nonzero(mask, as_tuple=True)
    edge_list = torch.stack([a_idx, b_idx], dim=0)
    return edge_list