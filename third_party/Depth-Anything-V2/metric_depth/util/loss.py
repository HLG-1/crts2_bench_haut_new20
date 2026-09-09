import torch
from torch import nn
import torch.nn.functional as F


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5, label_smoothing=0.0):
        super().__init__()
        self.lambd = lambd
        self.label_smoothing = label_smoothing

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        
        # S'assurer que les dimensions sont cohérentes
        if pred.dim() == 3:
            pred = pred.squeeze(0)  # (1, H, W) -> (H, W)
        if target.dim() == 3:
            target = target.squeeze(0)  # (1, H, W) -> (H, W)
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.squeeze(0)  # (1, H, W) -> (H, W)
        
        # Label smoothing pour depth estimation
        if self.label_smoothing > 0:
            # Ajouter du bruit gaussien aux targets pour smooth les labels
            noise = torch.randn_like(target) * self.label_smoothing * target
            target = target + noise
        
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class GradientLoss(nn.Module):
    """Gradient loss pour préserver les contours et structures géométriques."""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, valid_mask):
        # Ajouter la dimension batch si nécessaire
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)  # (H, W) -> (1, H, W)
        if target.dim() == 2:
            target = target.unsqueeze(0)  # (H, W) -> (1, H, W)
        if valid_mask.dim() == 2:
            valid_mask = valid_mask.unsqueeze(0)  # (H, W) -> (1, H, W)
        
        # Calculer les gradients spatiaux
        pred_grad_x = torch.abs(pred[:, :, :-1] - pred[:, :, 1:])
        pred_grad_y = torch.abs(pred[:, :-1, :] - pred[:, 1:, :])
        
        target_grad_x = torch.abs(target[:, :, :-1] - target[:, :, 1:])
        target_grad_y = torch.abs(target[:, :-1, :] - target[:, 1:, :])
        
        # Adapter valid_mask aux dimensions de gradient
        valid_mask_x = valid_mask[:, :, :-1] & valid_mask[:, :, 1:]
        valid_mask_y = valid_mask[:, :-1, :] & valid_mask[:, 1:, :]
        
        # Loss sur les gradients
        loss_x = F.l1_loss(pred_grad_x[valid_mask_x], target_grad_x[valid_mask_x])
        loss_y = F.l1_loss(pred_grad_y[valid_mask_y], target_grad_y[valid_mask_y])
        
        return loss_x + loss_y


class ScaleInvariantLoss(nn.Module):
    """Scale-invariant loss additionnelle."""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        
        # S'assurer que les dimensions sont cohérentes
        if pred.dim() == 3:
            pred = pred.squeeze(0)  # (1, H, W) -> (H, W)
        if target.dim() == 3:
            target = target.squeeze(0)  # (1, H, W) -> (H, W)
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.squeeze(0)  # (1, H, W) -> (H, W)
        
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        
        # Scale-invariant: penaliser les erreurs de forme plus que d'échelle
        loss_diff = torch.pow(diff_log, 2).mean()
        loss_mean = torch.pow(diff_log.mean(), 2)
        
        return loss_diff + loss_mean


class MultiLoss(nn.Module):
    """Loss combinée: SiLog + L1 + Gradient + Scale-invariant."""
    def __init__(self, silog_weight=1.0, l1_weight=0.5, gradient_weight=0.3, scale_weight=0.2, 
                 label_smoothing=0.0, silog_lambda=0.5):
        super().__init__()
        self.silog_weight = silog_weight
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.scale_weight = scale_weight
        
        self.silog_loss = SiLogLoss(lambd=silog_lambda, label_smoothing=label_smoothing)
        self.gradient_loss = GradientLoss()
        self.scale_loss = ScaleInvariantLoss()
        
    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        
        # Sauvegarder les dimensions originales
        orig_pred_dim = pred.dim()
        orig_target_dim = target.dim()
        orig_mask_dim = valid_mask.dim()
        
        # SiLog Loss avec label smoothing
        silog_loss = self.silog_loss(pred, target, valid_mask)
        
        # L1 Loss pour stabilité - s'assurer que les dimensions sont cohérentes
        if orig_pred_dim == 3:
            pred_flat = pred.squeeze(0)
        else:
            pred_flat = pred
            
        if orig_target_dim == 3:
            target_flat = target.squeeze(0)
        else:
            target_flat = target
            
        if orig_mask_dim == 3:
            valid_mask_flat = valid_mask.squeeze(0)
        else:
            valid_mask_flat = valid_mask
            
        l1_loss = F.l1_loss(pred_flat[valid_mask_flat], target_flat[valid_mask_flat])
        
        # Gradient Loss pour contours
        gradient_loss = self.gradient_loss(pred, target, valid_mask)
        
        # Scale-invariant Loss
        scale_loss = self.scale_loss(pred, target, valid_mask)
        
        # Loss combinée
        total_loss = (self.silog_weight * silog_loss + 
                     self.l1_weight * l1_loss + 
                     self.gradient_weight * gradient_loss + 
                     self.scale_weight * scale_loss)
        
        return total_loss
