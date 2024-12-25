# -*- coding:utf-8 -*-
'''
Author: MrZQAQ
Date: 2022-03-26 19:34
LastEditTime: 2022-11-23 16:34
LastEditors: MrZQAQ
Description: DeepLearing Model
FilePath: /MCANet/model.py
'''

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


class MCANet(nn.Module):
    def __init__(self, hp,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100):
        super(MCANet, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGTH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGTH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = 65
        self.protein_vocab_size = 26
        self.attention_dim = hp.conv * 4
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
            self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - \
            self.protein_kernel[0] - self.protein_kernel[1] - \
            self.protein_kernel[2] + 3
        self.drug_attention_head = 5
        self.protein_attention_head = 7
        self.mix_attention_head = 5

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )

        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)

        self.mix_attention_layer = nn.MultiheadAttention(
            self.attention_dim, self.mix_attention_head)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, protein):
        # [B, F_O] -> [B, F_O, D_E]
        # [B, T_O] -> [B, T_O, D_E]
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)
        # [B, F_O, D_E] -> [B, D_E, F_O]
        # [B, T_O, D_E] -> [B, D_E, T_O]
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # [B, D_E, F_O] -> [B, D_C, F_C]
        # [B, D_E, T_O] -> [B, D_C, T_C]
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        # [B, D_C, F_C] -> [F_C, B, D_C]
        # [B, D_C, T_C] -> [T_C, B, D_C]
        drug_QKV = drugConv.permute(2, 0, 1)
        protein_QKV = proteinConv.permute(2, 0, 1)

        # cross Attention
        # [F_C, B, D_C] -> [F_C, B, D_C]
        # [T_C, B, D_C] -> [T_C, B, D_C]
        drug_att, _ = self.mix_attention_layer(drug_QKV, protein_QKV, protein_QKV)
        protein_att, _ = self.mix_attention_layer(protein_QKV, drug_QKV, drug_QKV)

        # [F_C, B, D_C] -> [B, D_C, F_C]
        # [T_C, B, D_C] -> [B, D_C, T_C]
        drug_att = drug_att.permute(1, 2, 0)
        protein_att = protein_att.permute(1, 2, 0)

        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict

class HDN_conv_block(nn.Module):
    def __init__(self, in_channels=200, out_channels=200, num_heads=4, dropout=0.3):
        super(HDN_conv_block, self).__init__()
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


class HDNBlock(nn.Module):
    def __init__(self, in_channels=200, out_channels=200, num_heads=2, dropout=0.3):
        super(HDNBlock, self).__init__()
        
        self.hidden_channels = out_channels // (num_heads*2)
        self.drug_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=0.1)
        self.prot_conv = GATConv(in_channels, self.hidden_channels, num_heads, dropout=0.1)
        self.inter_conv = GATConv((in_channels, in_channels), self.hidden_channels, num_heads, dropout=0.3)
        self.drug_norm = LayerNorm(out_channels)
        self.prot_norm = LayerNorm(out_channels)
        self.drug_pool = SAGPooling(out_channels, min_score=-1)
        self.prot_pool = SAGPooling(out_channels, min_score=-1)

    def forward(self, atom_x, atom_edge_index, atom_batch, \
                aa_x, aa_edge_index, aa_edge_attr, aa_batch, m2p_edge_index):
        
        atom_x = F.elu(atom_x)
        aa_x = F.elu(aa_x)

        atom_intra_x = self.drug_conv(atom_x, atom_edge_index)
        atom_inter_x = self.inter_conv((aa_x, atom_x), m2p_edge_index[[1,0]])
        atom_x_tmp = torch.cat([atom_intra_x, atom_inter_x], -1)
        atom_x = F.elu(self.drug_norm(atom_x_tmp, atom_batch))

        aa_intra_x = self.prot_conv(aa_x, aa_edge_index, aa_edge_attr)
        aa_inter_x = self.inter_conv((atom_x, aa_x), m2p_edge_index)
        aa_x_tmp = torch.cat([aa_intra_x, aa_inter_x], -1)
        aa_x = F.elu(self.prot_norm(aa_x_tmp, aa_batch))

        atom_x, _, _, atom_batch, _, _ = self.drug_pool(atom_x, atom_edge_index, batch=atom_batch)
        aa_x, _, _, aa_batch, _, _ = self.prot_pool(aa_x, aa_edge_index, edge_attr=aa_edge_attr, batch=aa_batch)
        drug_global_repr = global_add_pool(atom_x, atom_batch)
        prot_global_repr = global_add_pool(aa_x, aa_batch)

        return atom_x, aa_x, drug_global_repr, prot_global_repr


class HDNDTI(nn.Module):
    def __init__(self, depth=6):
        super(HDNDTI, self).__init__()

        self.drug_in_channels = 43
        self.prot_in_channels = 33
        self.prot_evo_in_channels = 1280
        self.hidden_channels = 200
        self.depth = depth

        # MOLECULE IN FEAT
        self.atom_type_encoder = Embedding(20, self.hidden_channels)
        self.atom_feat_encoder = MLP([self.drug_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 

        # PROTEIN IN FEAT
        self.prot_evo = MLP([self.prot_evo_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 
        self.prot_aa = MLP([self.prot_in_channels, self.hidden_channels * 2, self.hidden_channels], out_norm=True) 

        # ENCODER
        # self.drug_convs = nn.ModuleList([HDN_conv_block() for _ in range(depth)])
        # self.prot_convs = nn.ModuleList([HDN_conv_block() for _ in range(depth)])
        self.blocks = nn.ModuleList([HDNBlock() for _ in range(depth)])

        self.attn = RESCAL(self.hidden_channels)
        # self.attn = AttentionLayer(self.hidden_channels)

    def forward(self,
                # Molecule
                atom_x, atom_x_feat, atom_edge_index, 
                # Protein (amino acid)
                aa_x, aa_evo_x, aa_edge_index, aa_edge_attr,
                # Batch
                atom_batch, aa_batch,
                # Bi Graph
                m2p_edge_index):

        # MOLECULE Featurize
        atom_x = self.atom_type_encoder(atom_x.squeeze()) + self.atom_feat_encoder(atom_x_feat)
                
        # PROTEIN Featurize
        aa_x = self.prot_aa(aa_x) + self.prot_evo(aa_evo_x)

        # Encoding
        drug_repr = []
        prot_repr = []
        for i in range(self.depth):
            # atom_x, drug_global_repr = self.drug_convs[i](atom_x, atom_edge_index, drug_batch)
            # aa_x, prot_global_repr = self.prot_convs[i](aa_x, aa_edge_index, prot_batch, aa_edge_weight)
            out = self.blocks[i](atom_x, atom_edge_index, atom_batch, \
                                 aa_x, aa_edge_index, aa_edge_attr, aa_batch, \
                                 m2p_edge_index)
            atom_x, aa_x, drug_global_repr, prot_global_repr = out
            drug_repr.append(drug_global_repr)
            prot_repr.append(prot_global_repr)
        drug_repr = torch.stack(drug_repr, dim=-2)
        prot_repr = torch.stack(prot_repr, dim=-2)

        # Co-attn
        scores = self.attn(drug_repr, prot_repr)

        return scores

class onlyPolyLoss(nn.Module):
    def __init__(self, hp,
                 protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100):
        super(onlyPolyLoss, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = 65
        self.protein_vocab_size = 26
        self.attention_dim = hp.conv * 4
        self.durg_dim_afterCNNs = self.drug_MAX_LENGH - \
            self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGH - \
            self.protein_kernel[0] - self.protein_kernel[1] - \
            self.protein_kernel[2] + 3

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.durg_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        
        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, protein):
        # [B, F_O] -> [B, F_O, D_E]
        # [B, T_O] -> [B, T_O, D_E]
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)

        # [B, F_O, D_E] -> [B, D_E, F_O]
        # [B, T_O, D_E] -> [B, D_E, T_O]
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        # [B, D_E, F_O] -> [B, D_C, F_C]
        # [B, D_E, T_O] -> [B, D_C, T_C]
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict
