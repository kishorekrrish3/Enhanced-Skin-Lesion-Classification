import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss combats severe class imbalance by down-weighting the easily classified examples.
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Expected to be a tensor of class weights if provided
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt is the probability of the true class
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            # We assume alpha is a tensor of weights [num_classes] 
            # and targets are the class indices [batch_size]
            alpha_mtx = self.alpha.gather(0, targets)
            focal_loss = focal_loss * alpha_mtx
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
            
def get_criterion(class_counts, device, use_focal=True):
    """Creates appropriate loss function based on class distribution"""
    # Calculate inverse frequencies for weights
    # Adding small epsilon to prevent extremely large weights
    weights = 1.0 / (torch.tensor(class_counts, dtype=torch.float32) + 1e-6)
    
    # Normalize weights so they sum to 1 to maintain loss scale
    weights = weights / weights.sum()
    weights = weights.to(device)
    
    if use_focal:
        print("💡 Using Class-Weighted Focal Loss (gamma=2.0)")
        return FocalLoss(alpha=weights, gamma=2.0, reduction='mean')
    else:
        print("💡 Using Weighted Cross Entropy Loss")
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
