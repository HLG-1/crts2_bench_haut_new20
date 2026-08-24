import torch
from torch import nn
import torch.nn.functional as F
import wandb
import os
import numpy as np

from augmentations import TransformWeak, TransformStrong
from utils.data import compute_median, get_bicls_files

class DoubleConv(nn.Module):
    '''(convolution => [BN] => ReLU) * 2'''

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
        

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduuce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        X1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetMulti(nn.Module):
    def __init__(self, output_channels):
        super(UNetMulti, self).__init__()
        self.bilinear = True

        # Encoder
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        # Shared transformation
        self.shared_transform = DoubleConv(64, 64)
        self.regression_head = OutConv(64, 1)

        self.classification_transform = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.classification_head = OutConv(64, output_channels)
        

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        shared_features = self.shared_transform(x)
        regression_output = self.regression_head(shared_features)

        classification_features = self.classification_transform(regression_output.detach()) + shared_features
        classification_output = self.classification_head(classification_features)
        return classification_output, regression_output

class UNetSingle(nn.Module):
    def __init__(self):
        super(UNetSingle, self).__init__()
        self.bilinear = True

        # Encoder
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        # Shared transformation
        self.shared_transform = DoubleConv(64, 64)
        self.regression_head = OutConv(64, 1)      

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        shared_features = self.shared_transform(x)
        regression_output = self.regression_head(shared_features)
        return regression_output

def cyclic_slice(x, start, length):
    n = x.numel()
    start = start % n
    end = (start + length) % n
    if end > start:
        return x[start:end]
    else:
        # wrap around
        return torch.cat([x[start:], x[:end]])

def generate_output_and_label(pred, label, length=10, num_list=None):
    """
    Generate output and label tensors for ListMLE loss with hard sample mining
    based on balanced sampling: 1/3 low difficulty, 1/3 medium, 1/3 high difficulty.

    Difficulty is computed as the absolute difference between the ranks of pred and label.
    """
    # pdb.set_trace()
    pred = pred.view(-1)
    label = label.view(-1)
    N = pred.numel()

    if num_list is None:
        num_list = N // length

    # ---- Compute ranks ----
    # rank_pred[i] = rank of pred[i] (0 = lowest, N-1 = highest)
    rank_pred = torch.argsort(torch.argsort(pred))
    rank_label = torch.argsort(torch.argsort(label))

    # Difficulty = absolute difference between ranks
    difficulty = torch.abs(rank_pred - rank_label)

    # ---- Sort indices by difficulty ----
    _, sorted_idx = torch.sort(difficulty)

    # Split into thirds
    third = N // 3
    low_idx = sorted_idx[:third]
    mid_idx = sorted_idx[third:2*third]
    high_idx = sorted_idx[2*third:]

    preds_list, labels_list = [], []
    for i in range(num_list):
        n_each = length // 3

        idx_low = cyclic_slice(low_idx, i*n_each, n_each) #low_idx[i*n_each % low_idx.numel() : (i*n_each + n_each) % low_idx.numel()]
        idx_mid = cyclic_slice(mid_idx, i*n_each, n_each) # mid_idx[i*n_each % mid_idx.numel() : (i*n_each + n_each) % mid_idx.numel()]
        idx_high = cyclic_slice(high_idx, i*n_each, n_each) # high_idx[i*n_each % high_idx.numel() : (i*n_each + n_each) % high_idx.numel()]

        remainder = length - (idx_low.numel() + idx_mid.numel() + idx_high.numel())
        if remainder > 0:
            idx_high = torch.cat([idx_high, high_idx[:remainder]])

        indices = torch.cat([idx_low, idx_mid, idx_high])

        preds_list.append(pred[indices])
        labels_list.append(label[indices])
    return torch.stack(preds_list, dim=0), torch.stack(labels_list, dim=0)

def listMLE(y_pred, y_true, eps=1e-8, padded_value_indicator=-999, k=None):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    if k is not None:
        sublist_indices = (y_pred.shape[1] * torch.rand(k, device=y_pred.device)).long()
        y_pred = y_pred[:, sublist_indices]
        y_true = y_true[:, sublist_indices]

    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))

class TSENet(nn.Module):
    def __init__(self, cfgs):
        super(TSENet, self).__init__()
        self.device = cfgs.get("device", "cuda")

        self.test = cfgs.get("test", False)
        self.num_landcover = cfgs.get("num_landcover", 6)
        self.landcovers = cfgs.get("landcovers", ["background", "impervious", "building", " lowvegetation", "tree", "car"])
        
        data_dir = cfgs.get("data_dir", "/data/gbh_old")

        ### Class deinition variations.
        self.bicls = bool(cfgs.get("bicls", True))
        self.sid = bool(cfgs.get("sid", False))
        self.ud = bool(cfgs.get("ud", False))
        assert self.bicls + self.sid + self.ud == 1, "One and only one scheme from bicls, sid and ud should be activated!"

        qvs_file, ndsm_stats_file, ns_file = get_bicls_files(cfgs)
        qvs_index = cfgs["data_train"].split(".txt")[0].split("train")[1]
        self.bicls_thres = cfgs.get("bicls_thres", 0.02)
        _, _, _, self.ndsm_max, _ = torch.load(ndsm_stats_file)
        self.cls_edges = torch.tensor(np.load(qvs_file)).to(self.device)
        self.cls_edges = torch.clamp(self.cls_edges, min=1e-8)
        ns = torch.cat((torch.tensor(np.load(ns_file)), torch.zeros(1))).to(self.device)
        if self.bicls_thres < 1:
            self.bicls_thres = self.bicls_thres * ns.max()
        ns_mask = ns > self.bicls_thres

        self.cls_edges = torch.cat((
            torch.zeros(1,).to(self.device),
            self.cls_edges[ns_mask],
            (torch.ones(1,) * self.ndsm_max).to(self.device)
        ))
        self.num_classes = self.cls_edges.shape[0] - 1
        if not self.bicls:
            ndsm_stats_file = os.path.join(data_dir, f"bi_ndsm_stats_{qvs_index}.pickle")
            _, _, self.ndsm_min, self.ndsm_max, _ = torch.load(ndsm_stats_file)
            if self.sid:
                self.slope = torch.log(torch.tensor(self.ndsm_max-self.ndsm_min + 1)).to(self.device)
                self.cls_edges = (torch.exp(torch.arange(self.num_classes+1) * torch.log(torch.tensor(self.ndsm_max - self.ndsm_min + 1)) / self.num_classes) - 1 + self.ndsm_min).to(self.device)
            elif self.ud:
                self.cls_edges = torch.linspace(self.ndsm_min, self.ndsm_max, self.num_classes+1).to(self.device)
                

        self.ordinal = bool(cfgs.get("ordinal", True))

        self.alpha = float(cfgs.get("alpha", 0.999))
        self.alpha_step = nn.Parameter(torch.tensor(0.0, device=self.device), requires_grad=False)

        self.use_rank = cfgs.get("use_rank", True)
        self.dyn_rank_thres = cfgs.get("dynamic_rank_threshold", True)
        self.rank_factor = float(cfgs.get("rank_factor", 0.999))
        self.rank_thres = float(cfgs.get("rank_thres", 0.9))
        if self.dyn_rank_thres:
            self.rank_thres = nn.Parameter(torch.ones(1), requires_grad=False)
        else:
            self.rank_thres = 0.5

        self.augmentation = cfgs.get("augmentation", True)
        if self.augmentation:
            self.weak_augmentations = TransformWeak(cfgs)
            self.strong_augmentations = TransformStrong(cfgs)
        self.use_error_ranking_loss = cfgs.get("use_error_ranking_loss", True)
        self.num_list = cfgs.get("num_list", 500)
        self.list_length = cfgs.get("list_length", 50)

        self.teacher = UNetMulti(2*(self.num_classes-1))
        self.student = UNetSingle()
        self.exam = UNetSingle()

        self._init_teacher(cfgs)

    def _init_teacher(self, cfgs):
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            if teacher_param.data.shape == student_param.data.shape:
                teacher_param.data.copy_(student_param.data)

        for student_param, exam_param in zip(self.student.parameters(), self.exam.parameters()):
            if exam_param.data.shape == student_param.data.shape:
                exam_param.data.copy_(student_param.data)
    
    def forward(self, x, gt):
        gt.update({
            "ndsm_cls": self.ndsm2cls(gt["ndsm"])
        })
        student_h = self.student(x)
        pred = {
            "student_ndsm": student_h
        }

        with torch.no_grad():
            exam_h = self.exam(x)
            pred.update({
                "ndsm": exam_h
            })

        teacher_logits, teacher_h = self.teacher(x)
        teacher_probs = self.logits2probs_infer_cumprod(teacher_logits)
        teacher_cls = torch.argmax(teacher_probs, dim=1, keepdim=True)
        pred.update({
            "teacher_cls": teacher_cls,
            "teacher_logits": teacher_logits,
            "teacher_probs": teacher_probs,
            "teacher_ndsm": teacher_h
        })

        losses = self.get_losses(pred, gt)
        if self.training:
            return losses, pred
        else:
            metric_params = self.get_metric_params(pred, gt)
            return losses, pred, metric_params

    def forward_unlabeled(self, x, gt):
        gt.update({
            "ndsm_cls": self.ndsm2cls(gt["ndsm"])
        })

        if self.augmentation and self.training:
            with torch.no_grad():
                x_weak, gt_ndsm_weak = self.weak_augmentations(x, gt["ndsm"])
                teacher_logits, teacher_h_org = self.teacher(x_weak)
                teacher_probs_org = self.logits2probs_infer_cumprod(teacher_logits)
                if self.use_rank:
                    teacher_cls_org = torch.argmax(teacher_probs_org, dim=1, keepdim=True)
                    teacher_rank_cls_org = self.get_rank(teacher_probs_org)
                    teacher_rank_cls_org = (teacher_rank_cls_org * F.one_hot(teacher_cls_org.squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2)).sum(dim=1, keepdim=True)

                    x_strong, preds_augmented, x_valid_mask = self.strong_augmentations(x_weak, [teacher_probs_org, teacher_rank_cls_org, teacher_h_org, gt_ndsm_weak])
                    teacher_probs, teacher_rank_cls, teacher_h, gt_ndsm = preds_augmented
                else:
                    x_strong, preds_augmented, x_valid_mask = self.strong_augmentations(x_weak, [teacher_probs_org, teacher_h_org, gt_ndsm_weak])
                    teacher_probs, teacher_h, gt_ndsm = preds_augmented

                teacher_cls = torch.argmax(teacher_probs, dim=1, keepdim=True)
                valid_mask = ~torch.any(teacher_probs==-1, dim=1, keepdim=True)
            student_h = self.student(x_strong)
        else:
            with torch.no_grad():
                teacher_logits, teacher_h = self.teacher(x)
                teacher_probs = self.logits2probs_infer_cumprod(teacher_logits)
                teacher_cls = torch.argmax(teacher_probs, dim=1, keepdim=True)
                valid_mask = torch.ones_like(teacher_cls)
            
            student_h = self.student(x)

        teacher_cls_probs = torch.max(teacher_probs, dim=1, keepdim=True)[0]
        pred = {
            "student_ndsm": student_h,
            "teacher_cls": teacher_cls.detach(),
            "teacher_probs": teacher_cls_probs.detach(),
            "teacher_ndsm": teacher_h.detach()
        }

        if self.use_rank and self.training:
            valid_mask = valid_mask * (teacher_rank_cls > self.rank_thres)
        
        pred.update({
            "teacher_mask": valid_mask.detach()
        })

        if self.augmentation and self.training:
            pred.update({
                "image_strong": x_strong,
                "image_weak": x_weak,
                "gt_ndsm": gt_ndsm,
                "gt_ndsm_cls": self.ndsm2cls(gt_ndsm)
            })

        losses = self.get_losses_unlabeled(pred, gt)
        if self.training:
            return losses, pred
        else:
            metric_params = self.get_metric_params_unlabeled(pred, gt)
            return losses, pred, metric_params

    def error_ranking_loss(self, pred, gt):
        teacher_probs = pred["teacher_probs"]
        teacher_cls = pred["teacher_cls"]
        teacher_cls_rank = self.get_rank_for_loss(teacher_probs)
        teacher_cls_rank = (teacher_cls_rank * F.one_hot(teacher_cls.squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2)).sum(dim=1, keepdim=True)

        valid_mask = (teacher_cls == gt["ndsm_cls"]).detach()

        teacher_err = torch.abs(gt["ndsm"] - pred["teacher_ndsm"]).detach()
        pred, label = generate_output_and_label(teacher_cls_rank[valid_mask], -teacher_err[valid_mask], num_list=self.num_list, length=self.list_length)
        return listMLE(pred, label)


    def get_losses(self, pred, gt):
        losses = {}
        if self.ordinal:
            with torch.no_grad():
                ord_label = self.label2ordlabel(gt["ndsm_cls"].detach())
            
            if self.training:
                logits = pred["teacher_logits"]
            else:
                logits = pred["teacher_logits"]
            N, _, H, W = logits.shape

            logits = logits.view(-1, 2, self.num_classes-1, H, W)
            prob = logits.softmax(dim=1)
            ord_prob = prob.view(N, 2*(self.num_classes-1), H, W)
            ord_prob = torch.clamp(ord_prob, min=1e-9)
            losses.update({
                "cross_entropy_sup": (-(torch.log(ord_prob) * ord_label).sum(dim=1, keepdim=True)).sum()
            })
        else:
            losses.update({
                "cross_entropy_sup": F.cross_entropy(pred["teacher_logits"], gt["ndsm_cls"].squeeze(1))
            })
        
        valid_mask = gt["ndsm"] >= 0
        if self.training:
            losses.update({
                "mae_student": (F.l1_loss(pred["student_ndsm"], gt["ndsm"], reduction="none") * valid_mask).sum() / valid_mask.sum(),
                "mae": (F.l1_loss(pred["teacher_ndsm"], gt["ndsm"], reduction="none") * valid_mask).sum() / valid_mask.sum()
            })
        else:
            losses.update({
                "mae": (F.l1_loss(pred["teacher_ndsm"], gt["ndsm"], reduction="none") * valid_mask).sum() / valid_mask.sum()
            })

        if self.use_error_ranking_loss & self.training:
            losses.update({
                "rank_loss": self.error_ranking_loss(pred, gt) *  1e-3
            })

        losses.update({
            "loss_total": sum(losses.values())
        })
        return losses
    
    def get_losses_unlabeled(self, pred, gt):
        losses = {}
        valid_mask = pred["teacher_mask"]

        valid_mask_reg = valid_mask * (pred["teacher_ndsm"]>=0)

        if valid_mask_reg.sum() == 0:
            losses.update({
                "mae_unlabeled": 0
            })
        else:
            losses.update({
                "mae_unlabeled": (F.l1_loss(pred["student_ndsm"], pred["teacher_ndsm"], reduction="none") * valid_mask_reg).sum() / (valid_mask_reg>0).sum()
            })
        
        losses.update({
            "loss_total": sum(losses.values())
        })
        return losses

    def ndsm2cls(self, ndsm):
        ndsm = torch.clamp(ndsm, min=0, max=self.ndsm_max)
        flag = (ndsm >= self.cls_edges[None, :-1, None, None]) * (ndsm <= self.cls_edges[None, 1:, None, None])
        return torch.argmax(flag.to(torch.float), dim=1, keepdim=True) 
    
    def logits2probs_infer_cumprod(self, logits, gt_cls=None):
        N, _, H, W = logits.shape
        logits = logits.view(-1, 2, self.num_classes-1, H, W)
        prob = logits.softmax(dim=1)

        pos_prob = torch.ones(N, self.num_classes, H, W, device=self.device, requires_grad=False)
        neg_prob = torch.ones(N, self.num_classes, H, W, device=self.device, requires_grad=False)

        pos_prob[:, 1:, :, :] *= torch.cumprod(prob[:, 0, :, :, :], dim=1)
        neg_prob[:, :-1, :, :] *= prob[:, 1, :, :, :]

        probs = pos_prob * neg_prob
        return probs
    
    def label2ordlabel(self, labels):
        N, _, H, W = labels.shape

        ord_c0 = torch.ones(N, self.num_classes-1, H, W, device=self.device, requires_grad=False)
        lin_mask = torch.linspace(0, self.num_classes-2, self.num_classes-1, requires_grad=False, device=self.device).view(1, self.num_classes-1, 1, 1)

        mask = lin_mask >= labels
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0

        loss_mask = lin_mask <= labels
        loss_weight = torch.where(loss_mask.sum(dim=(0, 2, 3), keepdim=True)>0, self.num_classes/loss_mask.sum(dim=(0, 2, 3), keepdim=True), torch.tensor(0., device=self.device))
        ord_label = torch.cat((ord_c0*loss_mask*loss_weight, ord_c1*loss_mask*loss_weight), dim=1)
        return ord_label

    def get_rank_for_loss(self, probs):
        N, C, H, W = probs.shape

        # Move class channel to front → shape: (C, N, H, W)
        probs_perm = probs.permute(1, 0, 2, 3).contiguous()
        # Flatten N*H*W pixels for each class
        probs_flat = probs_perm.view(C, -1)
        # Argsort twice to get rank
        res = torch.argsort(probs_flat, dim=1)
        res = torch.argsort(res, dim=1)
        # Reshape back to (C, N, H, W) then permute back to (N, C, H, W)
        res = res.view(C, N, H, W).permute(1, 0, 2, 3)
        # Normalize rank to [0, 1]
        return res / (N * H * W)
    
    @torch.no_grad()
    def get_rank(self, probs):
        N, C, H, W = probs.shape

        # Move class channel to front → shape: (C, N, H, W)
        probs_perm = probs.permute(1, 0, 2, 3).contiguous()
        # Flatten N*H*W pixels for each class
        probs_flat = probs_perm.view(C, -1)
        # Argsort twice to get rank
        res = torch.argsort(probs_flat, dim=1)
        res = torch.argsort(res, dim=1)
        # Reshape back to (C, N, H, W) then permute back to (N, C, H, W)
        res = res.view(C, N, H, W).permute(1, 0, 2, 3)
        # Normalize rank to [0, 1]
        return res / (N * H * W)


    def get_metric_params(self, pred, gt):
        if "ndsm_cls" in gt:
            gt_cls = gt["ndsm_cls"]
        metric_params = {}

        if "teacher_cls" in pred.keys():
            ord_cls = pred["teacher_cls"]
            tp = 0
            sum = 0
            for i in range(self.num_classes):
                tp_i = torch.logical_and(ord_cls==i, gt_cls==i).sum()
                sum_i = (ord_cls == i).sum()
                gt_i = (gt_cls == i).sum()
                metric_params.update({
                    "tp_{:02d}".format(i): tp_i,
                    "cls_sum_{:02d}".format(i): sum_i,
                    "gt_sum_{:02d}".format(i): gt_i
                })

                tp += tp_i
                sum += sum_i

            metric_params.update({
                "tp": tp,
                "cls_sum": sum
            })

        if "student_cls" in pred.keys():
            ord_cls = pred["student_cls"]
            tp = 0
            sum = 0
            for i in range(self.num_classes):
                tp_i = torch.logical_and(ord_cls==i, gt_cls==i).sum()
                sum_i = (ord_cls == i).sum()
                gt_i = (gt_cls == i).sum()
                metric_params.update({
                    "student_tp_{:02d}".format(i): tp_i,
                    "student_cls_sum_{:02d}".format(i): sum_i,
                    "student_gt_sum_{:02d}".format(i): gt_i
                })

                tp += tp_i
                sum += sum_i

            metric_params.update({
                "student_tp": tp,
                "student_cls_sum": sum
            })

        if "ndsm" in pred.keys():
            metric_params.update(self.get_rmse_by_range(pred, gt))

        if "student_ndsm" in pred.keys():
            pred1 = {"ndsm": pred["student_ndsm"]}
            mp = self.get_rmse_by_range(pred1, gt)
            metric_params.update({
                "student_"+k: v for k, v in mp.items()
            })

        return metric_params
    
    def get_metric_params_unlabeled(self, pred, gt):
        metric_params = {}
        if "teacher_cls" in pred.keys():
            ord_cls = pred["teacher_cls"]
            valid_mask = pred["teacher_mask"]>0
            gt_cls = gt["ndsm_cls"]
            tp = 0
            sum = 0
            for i in range(self.num_classes):
                tp_i = torch.logical_and(ord_cls[valid_mask]==i, gt_cls[valid_mask]==i).sum()
                sum_i = (ord_cls[valid_mask] == i).sum()
                gt_i = (gt_cls[valid_mask] == i).sum()
                metric_params.update({
                    "tp_{:02d}".format(i): tp_i,
                    "cls_sum_{:02d}".format(i): sum_i,
                    "gt_sum_{:02d}".format(i): gt_i
                })

                tp += tp_i
                sum += sum_i

            metric_params.update({
                "tp": tp,
                "cls_sum": sum
            })
        if "ndsm" in pred.keys():
            metric_params.update(self.get_rmse_by_range(pred, gt))
            pred_ndsm_cls = self.ndsm2cls(pred["ndsm"])
            metric_params.update({
                "tp_consistent": (pred_ndsm_cls==gt["ndsm_cls"]).sum()
            })
        return metric_params

    def update_teacher(self):
        self.alpha_step.data += 1
        alpha = min(1-1/(self.alpha_step+1), self.alpha)
        for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
            if teacher_param.data.shape == student_param.data.shape:
                teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1-alpha)
        for student_param, teacher_param in zip(self.student.parameters(), self.exam.parameters()):
            if teacher_param.data.shape == student_param.data.shape:
                teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1-alpha)

    def update_rank(self):
        if self.dyn_rank_thres:
            self.rank_thres.data.mul_(self.rank_factor).clamp_(0.5)

    def evaluate_rmse_by_range(self, eval_dict):
        eval_res = {
            "mae": eval_dict["ae"] / eval_dict["sum"],
            "rmse": torch.sqrt(eval_dict["se"] / eval_dict["sum"]),
        }

        if "ae00" in eval_dict:
            eval_res.update({"mae{:02d}".format(i): eval_dict["ae{:02d}".format(i)] / eval_dict["sum{:02d}".format(i)] for i in range(self.num_classes)})
            eval_res.update({"rmse{:02d}".format(i): torch.sqrt(eval_dict["se{:02d}".format(i)] / eval_dict["sum{:02d}".format(i)]) for i in range(self.num_classes)})
            eval_res.update({
                "mae00>0": eval_dict["ae00>0"] / eval_dict["sum00>0"],
                "rmse00>0": torch.sqrt(eval_dict["se00>0"] / eval_dict["sum00>0"])
            })
        if self.test:
            eval_res.update({
                "mae_per_building": eval_dict["ae_per_building"] / eval_dict["sum_per_building"],
                "rmse_per_building": torch.sqrt(eval_dict["se_per_building"] / eval_dict["sum_per_building"]),
                "per_building_bin_count": eval_dict["per_building_bin_count"],
                "per_building_bin_rmse": torch.sqrt(eval_dict["per_building_bin_se"] / eval_dict["per_building_bin_count"])
            })
            eval_res.update({
                "ae_bincount": eval_dict["ae_bincount"]
            })
            for i in range(self.num_landcover):
                eval_res.update({
                    "mae_lc{:02d}".format(i): eval_dict["ae_lc{:02d}".format(i)] / eval_dict["sum_lc{:02d}".format(i)],
                    "rmse_lc{:02d}".format(i): torch.sqrt(eval_dict["se_lc{:02d}".format(i)] / eval_dict["sum_lc{:02d}".format(i)])
                })
        return eval_res

    def get_rmse_by_range(self, pred, gt, num_bin=101, interval=1):
        metric_params = {
            "sum": (gt["ndsm"]>=0).sum(),
            "ae": (torch.abs(gt["ndsm"] - pred["ndsm"]) * (gt["ndsm"]>=0)).sum(),
            "se": ((gt["ndsm"] - pred["ndsm"]) ** 2 * (gt["ndsm"]>=0)).sum()
        }

        if "ndsm_cls" in gt:
            gt_cls = gt["ndsm_cls"]
            gt_cls = F.one_hot(gt_cls.squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2).contiguous()
            sum_cls = gt_cls.sum(dim=-1).sum(dim=-1).sum(dim=0)
            ae_cls = (torch.abs(gt["ndsm"] - pred["ndsm"]) * gt_cls).sum(dim=-1).sum(dim=-1).sum(dim=0)
            se_cls = ((gt["ndsm"] - pred["ndsm"])**2 *gt_cls).sum(dim=-1).sum(dim=-1).sum(dim=0)
            metric_params.update({"sum{:02d}".format(i): sum_cls[i] for i in range(self.num_classes)})
            metric_params.update({"ae{:02d}".format(i): ae_cls[i] for i in range(self.num_classes)})
            metric_params.update({"se{:02d}".format(i): se_cls[i] for i in range(self.num_classes)})

            sum8 = (gt_cls[:, :1] * (gt["ndsm"]>0)).sum()
            ae8 = (torch.abs(gt["ndsm"] - pred["ndsm"]) * gt_cls[:, :1] * (gt["ndsm"]>0)).sum()
            se8 = ((gt["ndsm"] - pred["ndsm"])**2 *gt_cls[:, :1] * (gt["ndsm"]>0)).sum()
            metric_params.update({
                "sum00>0": sum8,
                "ae00>0": ae8,
                "se00>0": se8
            })

        if self.test:
            assert "mask" in gt, "During test, Ground Truth should contain masks."
            assert gt["mask"].shape[0] == 1, "During test, batch size should be 1."
            for i in range(self.num_landcover):
                mask = gt["mask"] == i
                metric_params.update({
                    "sum_lc{:02d}".format(i): (mask * (gt["ndsm"]>=0)).sum(),
                    "ae_lc{:02d}".format(i): (torch.abs(gt["ndsm"] - pred["ndsm"]) * mask * (gt["ndsm"]>=0)).sum(),
                    "se_lc{:02d}".format(i): ((gt["ndsm"] - pred["ndsm"])**2 * mask * (gt["ndsm"]>=0)).sum()
                })

            building_index = self.landcovers.index("building")
            pred_l1, gt_l1, pred_h, gt_h = compute_median((gt["mask"]==building_index).squeeze(), pred["ndsm"].squeeze(), gt["ndsm"].squeeze())
            pred_h0 = pred_h[:, None].repeat(1,  num_bin)
            gt_h0 = gt_h[:, None].repeat(1, num_bin)
            lb = (torch.arange(num_bin)*interval).repeat(len(pred_h), 1).to(pred_h0.device)
            ub = (torch.arange(1, num_bin+1)*interval).repeat(len(gt_h), 1).to(pred_h0.device)
            se_raw = (pred_h0-gt_h0)**2
            discrete_mask = torch.logical_and(gt_h0>=lb, gt_h0<ub)
            count = discrete_mask.sum(dim=0)
            se = (se_raw * discrete_mask).sum(dim=0)
            metric_params.update({
                "sum_per_building": pred_h.numel(),
                "ae_per_building": torch.abs(pred_h-gt_h).sum(),
                "se_per_building": ((pred_h-gt_h)**2).sum(),
                "per_building_bin_count": count,
                "per_building_bin_se": se
            })
            boundaries = torch.arange(-300, 300, 1, device=pred_l1.device)
            err = pred_h - gt_h
            err = torch.clamp(err, min=-299, max=299)
            err_bin = torch.bucketize(err, boundaries)
            count_bin_init = torch.zeros(600, device=pred_l1.device)
            count_bin = torch.bincount(err_bin)
            count_bin_init[:count_bin.shape[0]] = count_bin
            metric_params.update({
                "ae_bincount": count_bin_init
            })
        return metric_params

    def evaluate(self, eval_dict):
        eval_res = {}

        if "student_tp" in eval_dict.keys():
            for i in range(self.num_classes):
                eval_res.update({
                    "student_precision_{:02d}".format(i): eval_dict["student_tp_{:02d}".format(i)] / eval_dict["student_cls_sum_{:02d}".format(i)],
                    "student_recall_{:02d}".format(i): eval_dict["student_tp_{:02d}".format(i)] / eval_dict["student_gt_sum_{:02d}".format(i)]
                })

            eval_res.update({
                "student_accuracy": eval_dict["student_tp"] / eval_dict["student_cls_sum"]
            })

        if "tp" in eval_dict.keys():
            for i in range(self.num_classes):
                eval_res.update({
                    "precision_{:02d}".format(i): eval_dict["tp_{:02d}".format(i)] / eval_dict["cls_sum_{:02d}".format(i)],
                    "recall_{:02d}".format(i): eval_dict["tp_{:02d}".format(i)] / eval_dict["gt_sum_{:02d}".format(i)]
                })

            eval_res.update({
                "accuracy": eval_dict["tp"] / eval_dict["cls_sum"],
            })

        if "ae" in eval_dict.keys():
            eval_res.update(self.evaluate_rmse_by_range(eval_dict))

        if "student_ae" in eval_dict.keys():
            ed1 = {k[8:]: v for k, v in eval_dict.items() if k.startswith("student_")}
            er = self.evaluate_rmse_by_range(ed1)
            eval_res.update({
                "student_"+k: v for k, v in er.items()
            })

        if "unlabeled_tp" in eval_dict.keys():
            for i in range(self.num_classes):
                eval_res.update({
                    "precision_{:02d}".format(i): eval_dict["tp_{:02d}".format(i)] / eval_dict["cls_sum_{:02d}".format(i)],
                    "recall_{:02d}".format(i): eval_dict["tp_{:02d}".format(i)] / eval_dict["gt_sum_{:02d}".format(i)]
                })

            eval_res.update({
                "accuracy": eval_dict["tp"] / eval_dict["cls_sum"],
            })

        if "tp_consistent" in eval_dict.keys():
            eval_res.update({
                "accuracy_consistent": eval_dict["tp_consistent"] / eval_dict["cls_sum"]
            })

        return eval_res
    
    def vis(self, image, pred, gt=None, validation=False, unlabeled=False, image_idx=None):
        if unlabeled:
            prefix = 'train_unlabeled/'
        elif validation:
            prefix = 'val/'
        else:
            prefix = 'train/'
        image_idx = image_idx[0]
        ndsm_cls_dict = {}
        ndsm_dict = {}

        if gt is not None:
            if "ndsm_cls" in gt:
                gt_cls = gt["ndsm_cls"][0].float().cpu().numpy() / self.num_classes
                ndsm_cls_dict = {
                    "true": wandb.Image(gt_cls, caption=image_idx)
                }

            max_h = gt["ndsm"][0].max().item()
            gt_ndsm = gt["ndsm"][0].float().cpu().numpy() / max_h
            ndsm_dict = {
                'true': wandb.Image(gt_ndsm, caption=image_idx)
            }

        if "student_cls" in pred.keys():
            student_cls = pred["student_cls"][0].float().cpu().numpy() / self.num_classes
            ndsm_cls_dict.update({
                'pred_student': wandb.Image(student_cls, caption=image_idx),
            })

        if "teacher_cls" in pred.keys():
            teacher_cls = pred["teacher_cls"][0].float().cpu().numpy() / self.num_classes
            ndsm_cls_dict.update({
                'pred_teacher': wandb.Image(teacher_cls, caption=image_idx)
            })

        if "student_ndsm" in pred.keys():
            student_ndsm = pred["student_ndsm"][0].detach().float().cpu().numpy() / max_h
            ndsm_dict.update({
                "pred_student": wandb.Image(student_ndsm, caption=image_idx)
            })

        if "teacher_ndsm" in pred.keys():
            teacher_ndsm = pred["teacher_ndsm"][0].detach().float().cpu().numpy() / max_h
            ndsm_dict.update({
                "pred_teacher": wandb.Image(teacher_ndsm, caption=image_idx)
            })

        if "ndsm" in pred.keys():
            ndsm = pred["ndsm"][0].detach().float().cpu().numpy() / max_h
            ndsm_dict.update({
                "pred": wandb.Image(ndsm, caption=image_idx)
            })

        vis_dict = {
            prefix+"image": wandb.Image(image[0].cpu()),
            prefix+'ndsm_cls': ndsm_cls_dict,
            prefix+'ndsm': ndsm_dict
        }
        return vis_dict
