# -*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from layers import *
from torch_geometric.nn import (
                                GATConv,
                                GCNConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool,
                                global_mean_pool,
                                global_max_pool
                                )
from config import hyperparameter
from utils.DataSetsFunction import vocab_size_drug,vocab_size_prot


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms + self.eps)
        return x_norm * self.weight


class DepthAttentionResidual(nn.Module):
    """
    Full AttnRes over depth:
    h_l = sum_i alpha_{i->l} * v_i, alpha_{i->l} = softmax(w_l^T RMSNorm(k_i)).
    """

    def __init__(self, num_layers, hidden_dim, dropout=0.0, causal=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.scale = 1.0 / math.sqrt(hidden_dim)

        self.pseudo_queries = nn.Parameter(torch.empty(num_layers, hidden_dim))
        nn.init.xavier_uniform_(self.pseudo_queries)

        self.key_norm = RMSNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, reprs):
        if reprs.dim() != 3:
            raise ValueError(f"Expected 3D tensor [batch, layers, dim], got {reprs.shape}")
        if reprs.size(1) != self.num_layers:
            raise ValueError(
                f"Layer mismatch: reprs has {reprs.size(1)} layers, expected {self.num_layers}."
            )

        keys = self.key_norm(reprs)
        outputs = []
        for layer_idx in range(self.num_layers):
            if self.causal:
                key_slice = keys[:, :layer_idx + 1, :]
                value_slice = reprs[:, :layer_idx + 1, :]
            else:
                key_slice = keys
                value_slice = reprs

            scores = torch.matmul(key_slice, self.pseudo_queries[layer_idx]) * self.scale
            scores = scores - scores.max(dim=-1, keepdim=True).values
            weights = torch.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            layer_out = torch.sum(weights.unsqueeze(-1) * value_slice, dim=1)
            outputs.append(layer_out)

        return torch.stack(outputs, dim=1)


class AttentionResidualRESCAL(nn.Module):
    def __init__(self, n_features, depth, dropout=0.0):
        super().__init__()
        self.n_features = n_features
        self.depth = depth

        self.drug_attn_res = DepthAttentionResidual(depth, n_features, dropout=dropout, causal=True)
        self.prot_attn_res = DepthAttentionResidual(depth, n_features, dropout=dropout, causal=True)

        self.co_attn = CoAttentionLayer(n_features)
        # Start close to baseline and let training decide how much AttnRes/co-attn fusion to use.
        self.drug_res_mix_logit = nn.Parameter(torch.tensor(-4.0))
        self.prot_res_mix_logit = nn.Parameter(torch.tensor(-4.0))
        self.fusion_mix_logit = nn.Parameter(torch.tensor(-4.0))
        self.mlp = nn.Sequential(
            nn.Linear(depth * depth, 2)
        )

    def forward(self, heads, tails):
        # Gradually inject depth-wise attention residuals (stable warm start).
        heads_attn = self.drug_attn_res(heads)
        tails_attn = self.prot_attn_res(tails)
        drug_mix = torch.sigmoid(self.drug_res_mix_logit)
        prot_mix = torch.sigmoid(self.prot_res_mix_logit)
        heads = heads + drug_mix * (heads_attn - heads)
        tails = tails + prot_mix * (tails_attn - tails)

        alpha_scores = self.co_attn(heads, tails)
        fusion_scores = torch.softmax(alpha_scores, dim=-1)
        # Keep expected scale near 1 (softmax mean is 1/depth) to avoid shrinking interactions.
        fusion_scores = fusion_scores * fusion_scores.size(-1)
        fusion_mix = torch.sigmoid(self.fusion_mix_logit)

        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        score_matrix = heads @ tails.transpose(-2, -1)

        interaction_matrix = score_matrix * ((1.0 - fusion_mix) + fusion_mix * fusion_scores)
        interaction_repr = interaction_matrix.reshape(interaction_matrix.shape[0], -1)
        logits = self.mlp(interaction_repr)
        return logits, interaction_repr


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

        self.attn = AttentionResidualRESCAL(self.hidden_channels, self.depth * 2, dropout=0.0)
        # self.attn = RESCAL(self.hidden_channels, self.depth*2)
        # self.attn = PoolAttention(self.hidden_channels)

        self.to(device)

    def forward(self,data):
        #print(data)
        #print(data.mol_smiles_x)  #torch.Size([128, 200])
        #print(data.prot_seq_x.shape)  #torch.Size([128,1500])
        #print(data.mol_smiles_x.shape)  #[batch,200]
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
        logits, interaction_repr = self.attn(drug_repr, prot_repr)
        return logits, interaction_repr


class SimilarityGraphRefiner(nn.Module):
    """Builds a batch-level similarity graph and refines node features via GCN."""

    def __init__(self, hidden_dim, threshold=0.5, dropout=0.1):
        super().__init__()
        self.threshold = threshold
        self.dropout = dropout
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x):
        if x.size(0) <= 1:
            return x

        x_norm = F.normalize(x, dim=-1, eps=1e-8)
        sim = torch.matmul(x_norm, x_norm.transpose(0, 1))
        edge_index = (sim >= self.threshold).nonzero(as_tuple=False).t().contiguous()

        node_ids = torch.arange(x.size(0), device=x.device, dtype=torch.long)
        self_loops = torch.stack([node_ids, node_ids], dim=0)
        edge_index = torch.cat([edge_index, self_loops], dim=1)

        out = F.elu(self.conv1(x, edge_index))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out, edge_index)
        return x + out


class RSGCLDTI(nn.Module):
    """
    RSGCL-style DTI model adapted to MIF-DTI data format.
    - Sequence CNN branch (drug SMILES / protein sequence)
    - Graph branch (drug/protein structural graph)
    - Relational similarity refinement over each mini-batch
    """

    def __init__(self, hidden_channels=200, device='cuda:0'):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.device = device
        self.drug_in_channels = 43
        self.prot_in_channels = 33
        self.prot_evo_in_channels = 1280

        # Graph input projection
        self.atom_type_encoder = Embedding(20, self.hidden_channels)
        self.atom_feat_encoder = MLP(
            [self.drug_in_channels, self.hidden_channels * 2, self.hidden_channels],
            out_norm=True
        )
        self.prot_aa = MLP(
            [self.prot_in_channels, self.hidden_channels * 2, self.hidden_channels],
            out_norm=True
        )
        self.prot_evo = MLP(
            [self.prot_evo_in_channels, self.hidden_channels * 2, self.hidden_channels],
            out_norm=True
        )

        # Graph encoders (drug/protein)
        self.drug_gnn_1 = GATConv(self.hidden_channels, self.hidden_channels // 2, heads=2, dropout=0.1)
        self.drug_gnn_2 = GATConv(self.hidden_channels, self.hidden_channels // 2, heads=2, dropout=0.1)
        self.prot_gnn_1 = GATConv(self.hidden_channels, self.hidden_channels // 2, heads=2, dropout=0.1)
        self.prot_gnn_2 = GATConv(self.hidden_channels, self.hidden_channels // 2, heads=2, dropout=0.1)
        self.drug_gnn_norm = nn.LayerNorm(self.hidden_channels)
        self.prot_gnn_norm = nn.LayerNorm(self.hidden_channels)

        self.drug_graph_proj = nn.Linear(self.hidden_channels * 2, self.hidden_channels)
        self.prot_graph_proj = nn.Linear(self.hidden_channels * 2, self.hidden_channels)

        # Sequence encoders (RSGCL-style CNN)
        seq_embed_dim = self.hidden_channels * 2
        self.smiles_embedding = nn.Embedding(65, seq_embed_dim, padding_idx=0)
        self.protein_embedding = nn.Embedding(26, seq_embed_dim, padding_idx=0)

        self.CNN_smiles = nn.Sequential(
            nn.Conv1d(in_channels=seq_embed_dim, out_channels=512, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=self.hidden_channels, kernel_size=7, padding=3),
            nn.ReLU()
        )
        self.CNN_sequence = nn.Sequential(
            nn.Conv1d(in_channels=seq_embed_dim, out_channels=512, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=self.hidden_channels, kernel_size=7, padding=3),
            nn.ReLU()
        )

        # Side fusion (graph + sequence + optional external embedding)
        self.drug_side_proj = nn.Linear(self.hidden_channels * 3, self.hidden_channels)
        self.prot_side_proj = nn.Linear(self.hidden_channels * 3, self.hidden_channels)

        # Relational similarity refinement in-batch
        self.drug_rel_refiner = SimilarityGraphRefiner(self.hidden_channels, threshold=0.5, dropout=0.1)
        self.prot_rel_refiner = SimilarityGraphRefiner(self.hidden_channels, threshold=0.5, dropout=0.1)

        # Cross-modal interaction
        self.cross_attention = nn.MultiheadAttention(self.hidden_channels, num_heads=4, batch_first=True, dropout=0.1)

        # RSGCL-style classifier head
        self.d1 = nn.Dropout(p=0.1)
        self.d2 = nn.Dropout(p=0.1)
        self.d3 = nn.Dropout(p=0.1)
        self.leaky = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.hidden_channels * 6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

        self.to(device)

    def _pool_external_embedding(self, tensor, batch_size, device):
        if tensor is None or not torch.is_tensor(tensor):
            return torch.zeros(batch_size, self.hidden_channels, device=device)

        emb = tensor
        if emb.dim() == 3 and emb.size(1) == 1:
            emb = emb.squeeze(1)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        emb = emb.float()

        if emb.size(0) > batch_size:
            emb = emb[:batch_size]
        elif emb.size(0) < batch_size:
            pad = torch.zeros(batch_size - emb.size(0), emb.size(1), device=emb.device, dtype=emb.dtype)
            emb = torch.cat([emb, pad], dim=0)

        emb = F.adaptive_avg_pool1d(emb.unsqueeze(1), self.hidden_channels).squeeze(1)
        return emb.to(device)

    def _encode_sequence(self, tokens, embedding, cnn):
        if tokens.dim() == 3 and tokens.size(1) == 1:
            tokens = tokens.squeeze(1)
        seq = embedding(tokens.long()).permute(0, 2, 1)
        seq = cnn(seq)
        seq = F.adaptive_max_pool1d(seq, 1).squeeze(-1)
        return seq

    def forward(self, data):
        # Molecule fields
        atom_x = data.mol_x
        atom_x_feat = data.mol_x_feat
        atom_edge_index = data.mol_edge_index
        atom_batch = data.mol_x_batch
        smiles_x = data.mol_smiles_x

        # Protein fields
        aa_x = data.prot_node_aa
        aa_evo_x = data.prot_node_evo
        aa_edge_index = data.prot_edge_index
        aa_batch = data.prot_node_aa_batch
        seq_x = data.prot_seq_x

        # Node feature initialization
        atom_x = self.atom_type_encoder(atom_x.squeeze(-1)) + self.atom_feat_encoder(atom_x_feat)
        aa_x = self.prot_aa(aa_x) + self.prot_evo(aa_evo_x)

        # Graph encoders
        atom_x = F.elu(self.drug_gnn_norm(self.drug_gnn_1(atom_x, atom_edge_index)))
        atom_x = F.elu(self.drug_gnn_norm(self.drug_gnn_2(atom_x, atom_edge_index)))
        aa_x = F.elu(self.prot_gnn_norm(self.prot_gnn_1(aa_x, aa_edge_index)))
        aa_x = F.elu(self.prot_gnn_norm(self.prot_gnn_2(aa_x, aa_edge_index)))

        drug_graph = torch.cat([global_mean_pool(atom_x, atom_batch), global_max_pool(atom_x, atom_batch)], dim=-1)
        prot_graph = torch.cat([global_mean_pool(aa_x, aa_batch), global_max_pool(aa_x, aa_batch)], dim=-1)
        drug_graph = self.drug_graph_proj(drug_graph)
        prot_graph = self.prot_graph_proj(prot_graph)

        # Sequence encoders
        drug_seq = self._encode_sequence(smiles_x, self.smiles_embedding, self.CNN_smiles)
        prot_seq = self._encode_sequence(seq_x, self.protein_embedding, self.CNN_sequence)

        # Optional external embeddings from dataset (if available)
        batch_size = drug_graph.size(0)
        mol_emb = getattr(data, 'mol_embedding', None)
        prot_emb = getattr(data, 'prot_embedding', None)
        drug_external = self._pool_external_embedding(mol_emb, batch_size, drug_graph.device)
        prot_external = self._pool_external_embedding(prot_emb, batch_size, prot_graph.device)

        # Side fusion
        drug_side = torch.cat([drug_graph, drug_seq, drug_external], dim=-1)
        prot_side = torch.cat([prot_graph, prot_seq, prot_external], dim=-1)
        drug_side = self.d1(self.leaky(self.drug_side_proj(drug_side)))
        prot_side = self.d1(self.leaky(self.prot_side_proj(prot_side)))

        # In-batch relational similarity refinement
        drug_rel = self.drug_rel_refiner(drug_side)
        prot_rel = self.prot_rel_refiner(prot_side)

        # Cross interaction
        drug_ctx, _ = self.cross_attention(drug_rel.unsqueeze(1), prot_rel.unsqueeze(1), prot_rel.unsqueeze(1))
        prot_ctx, _ = self.cross_attention(prot_rel.unsqueeze(1), drug_rel.unsqueeze(1), drug_rel.unsqueeze(1))
        drug_ctx = drug_ctx.squeeze(1)
        prot_ctx = prot_ctx.squeeze(1)

        interaction_repr = torch.cat([
            drug_rel,
            prot_rel,
            drug_ctx,
            prot_ctx,
            torch.abs(drug_rel - prot_rel),
            drug_rel * prot_rel
        ], dim=-1)

        f1 = self.d2(self.leaky(self.fc1(self.d1(interaction_repr))))
        f2 = self.d3(self.leaky(self.fc2(f1)))
        f3 = self.leaky(self.fc3(f2))
        logits = self.fc4(f3)
        return logits, interaction_repr


def get_m2p_edge_from_batch(atom_batch, aa_batch, node_level=None):

    mask = atom_batch.unsqueeze(1) == aa_batch.unsqueeze(0)  # (num_a_nodes, num_b_nodes) 的bool矩阵
    if node_level is not None:
        mask = mask * (node_level==1).unsqueeze(1)
    a_idx, b_idx = torch.nonzero(mask, as_tuple=True)
    edge_list = torch.stack([a_idx, b_idx], dim=0)
    return edge_list
