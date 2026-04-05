import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxMarginContrastiveLoss(nn.Module):
    """
    PyTorch implementation of max-margin contrastive loss (Hadsell et al., 2006).
    Expects a per-sample embedding `z` and integer labels `y`.
    Computes pairwise distances and applies margin-based loss:
      L = mean( y_ij * d^2 + (1 - y_ij) * max(0, margin - d)^2 ) over i<j
    where y_ij = 1 if labels equal else 0.
    """
    def __init__(self, margin: float = 1.0, metric: str = 'euclidean'):
        super().__init__()
        assert metric in ('euclidean', 'cosine'), 'metric must be euclidean or cosine'
        self.margin = margin
        self.metric = metric

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Tensor [batch, dim] embeddings
            y: LongTensor [batch] labels
        Returns:
            Scalar contrastive loss
        """
        assert z.dim() == 2, 'z must be [batch, dim]'
        assert y.dim() == 1 and y.size(0) == z.size(0), 'y shape mismatch'

        # Pairwise distance matrix
        if self.metric == 'euclidean':
            # ||a-b|| = sqrt(||a||^2 - 2a.b + ||b||^2)
            r = (z * z).sum(dim=1, keepdim=True)
            D2 = torch.clamp(r - 2 * (z @ z.t()) + r.t(), min=0.0)
            D = torch.sqrt(D2 + 1e-12)
        else:  # cosine distance
            z_norm = F.normalize(z, dim=1)
            S = z_norm @ z_norm.t()  # cosine similarity [-1,1]
            D = 1.0 - S

        # Build pairwise labels y_ij
        yi = y.view(-1, 1)
        yj = y.view(1, -1)
        y_ij = (yi == yj).float()

        # Mask out diagonal
        batch = z.size(0)
        diag_mask = torch.eye(batch, device=z.device)
        y_ij = y_ij * (1.0 - diag_mask)
        D = D * (1.0 - diag_mask)

        # Positive pairs: y_ij==1 -> pull together
        pos_loss = (D ** 2) * y_ij

        # Negative pairs: y_ij==0 -> push apart with margin
        neg = torch.clamp(self.margin - D, min=0.0)
        neg_loss = (neg ** 2) * (1.0 - y_ij)

        # Average over all i!=j
        denom = batch * (batch - 1)
        loss = (pos_loss.sum() + neg_loss.sum()) / max(denom, 1)
        return loss


class NPairsLoss(nn.Module):
    """
    PyTorch implementation of Multi-class N-pair loss (Sohn, 2016).
    For each class in the batch, form anchor-positive pairs and perform
    softmax classification of anchors over positives (positives of other
    classes act as negatives).
    """
    def __init__(self, l2_reg: float = 0.0, normalize: bool = True):
        super().__init__()
        self.l2_reg = l2_reg
        self.normalize = normalize

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Tensor [batch, dim] embeddings
            y: LongTensor [batch] labels
        Returns:
            Scalar N-pair loss
        """
        assert z.dim() == 2
        assert y.dim() == 1 and y.size(0) == z.size(0)

        # group indices by label
        device = z.device
        anchors_idx = []
        positives_idx = []

        unique_labels = torch.unique(y)
        for lbl in unique_labels.tolist():
            idx = (y == lbl).nonzero(as_tuple=False).view(-1)
            if idx.numel() < 2:
                continue
            # random pairing to avoid bias
            perm = torch.randperm(idx.numel(), device=device)
            idx = idx[perm]
            half = idx.numel() // 2
            if half == 0:
                continue
            anchors_idx.append(idx[:half])
            positives_idx.append(idx[half:half*2])

        if len(anchors_idx) == 0:
            # no valid pairs in this batch
            return z.sum() * 0.0

        anchors_idx = torch.cat(anchors_idx)
        positives_idx = torch.cat(positives_idx)

        a = z[anchors_idx]
        p = z[positives_idx]

        if self.normalize:
            a = F.normalize(a, dim=1)
            p = F.normalize(p, dim=1)

        # similarity matrix between anchors and all positives
        S = a @ p.t()  # [M, M]
        targets = torch.arange(S.size(0), device=device)
        ce = F.cross_entropy(S, targets)

        if self.l2_reg > 0:
            reg = (a.norm(dim=1).pow(2).mean() + p.norm(dim=1).pow(2).mean()) * 0.5
            ce = ce + self.l2_reg * reg

        return ce


class TripletLoss(nn.Module):
    """
    PyTorch implementation of triplet loss.
    Supports hard, soft, and semihard mining strategies.
    
    Args:
        margin: Float, the margin parameter for triplet loss
        kind: String, one of ('hard', 'soft', 'semihard')
    """
    def __init__(self, margin: float = 1.0, kind: str = 'hard'):
        super().__init__()
        assert kind in ('hard', 'soft', 'semihard'), 'kind must be hard, soft, or semihard'
        self.margin = margin
        self.kind = kind
    
    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Tensor [batch, dim] embeddings (should be L2 normalized)
            y: LongTensor [batch] labels
        Returns:
            Scalar triplet loss
        """
        assert z.dim() == 2, 'z must be [batch, dim]'
        assert y.dim() == 1 and y.size(0) == z.size(0), 'y shape mismatch'
        
        # Ensure embeddings are normalized for cosine similarity
        z = F.normalize(z, dim=1)
        
        # Compute cosine similarity matrix
        # similarity[i,j] = cos(angle between z[i] and z[j])
        similarity = z @ z.t()
        
        if self.kind == 'hard' or self.kind == 'soft':
            # Find hardest positive and negative examples
            batch_size = z.size(0)
            loss = 0.0
            valid_triplets = 0
            
            for i in range(batch_size):
                # Positive samples are those with the same label
                pos_mask = (y == y[i]).float()
                pos_mask[i] = 0  # Exclude self
                
                # Negative samples are those with different labels
                neg_mask = (y != y[i]).float()
                
                # If no positive or negative samples, skip this example
                if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                    continue
                
                # For each anchor, find the hardest positive (minimum similarity)
                pos_sim = similarity[i] * pos_mask
                hard_pos_idx = torch.argmin(pos_sim)
                hard_pos_sim = pos_sim[hard_pos_idx]
                
                # Find the hardest negative (maximum similarity)
                neg_sim = similarity[i] * neg_mask
                hard_neg_idx = torch.argmax(neg_sim)
                hard_neg_sim = neg_sim[hard_neg_idx]
                
                # Compute triplet loss
                if self.kind == 'hard':
                    # Standard hard triplet loss
                    loss += F.relu(1 - hard_pos_sim + hard_neg_sim)
                else:  # soft
                    # Soft margin triplet loss
                    loss += torch.log(1 + torch.exp(-(hard_pos_sim - hard_neg_sim)))
                
                valid_triplets += 1
            
            if valid_triplets > 0:
                return loss / valid_triplets
            else:
                return torch.tensor(0.0, device=z.device)
        
        elif self.kind == 'semihard':
            # Semihard negative mining
            # Find semihard negatives where: pos_sim < neg_sim < pos_sim + margin
            batch_size = z.size(0)
            loss = 0.0
            valid_triplets = 0
            
            for i in range(batch_size):
                # Positive samples
                pos_mask = (y == y[i]).float()
                pos_mask[i] = 0  # Exclude self
                
                # Negative samples
                neg_mask = (y != y[i]).float()
                
                # Skip if no positive or negative samples
                if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                    continue
                
                # Find hardest positive (minimum similarity)
                pos_sim = similarity[i] * pos_mask
                hard_pos_idx = torch.argmin(pos_sim)
                hard_pos_sim = pos_sim[hard_pos_idx]
                
                # Semihard negatives: neg_sim > hard_pos_sim and neg_sim < hard_pos_sim + margin
                neg_sim = similarity[i] * neg_mask
                semihard_mask = (neg_sim > hard_pos_sim) & (neg_sim < (hard_pos_sim + self.margin))
                
                if semihard_mask.sum() > 0:
                    # Use all semihard negatives
                    semihard_neg_sim = neg_sim[semihard_mask]
                    loss += F.relu(1 - hard_pos_sim + semihard_neg_sim).mean()
                    valid_triplets += 1
                else:
                    # Fall back to hardest negative
                    hard_neg_idx = torch.argmax(neg_sim)
                    hard_neg_sim = neg_sim[hard_neg_idx]
                    loss += F.relu(1 - hard_pos_sim + hard_neg_sim)
                    valid_triplets += 1
            
            if valid_triplets > 0:
                return loss / valid_triplets
            else:
                return torch.tensor(0.0, device=z.device)


class SupervisedNTXentLoss(nn.Module):
    """
    PyTorch implementation of supervised normalized temperature-scaled cross entropy loss.
    A variant of Multi-class N-pair Loss from (Sohn 2016)
    Later used in SimCLR (Chen et al. 2020, Khosla et al. 2020).
    
    Args:
        temperature: Float, temperature scaling parameter
        base_temperature: Float, base temperature for loss scaling
    """
    def __init__(self, temperature: float = 0.5, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Tensor [batch, dim] embeddings
            y: LongTensor [batch] labels
        Returns:
            Scalar contrastive loss
        """
        assert z.dim() == 2, 'z must be [batch, dim]'
        assert y.dim() == 1 and y.size(0) == z.size(0), 'y shape mismatch'

        batch_size = z.size(0)
        device = z.device

        # Expand labels to compute mask
        y = y.unsqueeze(-1)
        mask = torch.eq(y, y.transpose(0, 1)).float()
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(z, z.transpose(0, 1)),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask to exclude self-contrast cases
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # Compute exponential logits and log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(1)
        # Only consider samples with at least one positive pair
        positive_mask = (mask_sum > 0).float()
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        mean_log_prob_pos = mean_log_prob_pos[mask_sum > 0]
        
        # Compute loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss