import torch
from torch import nn
import torch.nn.functional as F
import wandb
from skimage import io
import os
import numpy as np
from utils.data import compute_median, get_bicls_files

class BaseHeightPredictor(nn.Module):
    def __init__(self, cfgs=None):
        super(BaseHeightPredictor, self).__init__()
        self.test = cfgs.get("test", False)
        self.num_landcover = cfgs.get("num_landcover", 6)
        self.landcovers = cfgs.get("landcovers", ["background", "impervious", "building", " lowvegetation", "tree", "car"])
        self.stats_file = cfgs.get("stats_file", 'splits/stats.pkl')

        self.test_by_range = cfgs.get("test_by_range", True)
        if self.test_by_range:
            self.cls_edges = self.get_cls_edges(cfgs)
    
    def get_cls_edges(self, cfgs):
        data_dir = cfgs.get("data_dir", "/data/gbh_old")
        bicls = cfgs.get("bicls", True)
        bicls_thres = cfgs.get("bicls_thres", 2000)
        device = cfgs.get("device", "cuda")
        if bicls:
            qvs_file, ndsm_stats_file, ns_file = get_bicls_files(cfgs)
            _, _, _, self.ndsm_max, _ = torch.load(ndsm_stats_file)
            self.cls_edges = torch.tensor(np.load(qvs_file)).to(device)
            self.cls_edges = torch.clamp(self.cls_edges, min=1e-8)
            ns = torch.cat((torch.tensor(np.load(ns_file)), torch.zeros(1))).to(device)
            ns_mask = ns>bicls_thres
            return torch.cat((torch.zeros(1,).to(device), self.cls_edges[ns_mask], (torch.ones(1,)*self.ndsm_max).to(device)))

    @staticmethod
    def _get_losses(cls, pred, gt):
        mask = gt["ndsm"]>=0
        losses = {
            "mae": (F.l1_loss(pred["ndsm"], gt["ndsm"], reduction='none') * mask).sum() / mask.sum()
                }
        losses.update({"loss_total": sum(losses.values())})
        return losses

    def get_losses(self, pred, gt):
        return self._get_losses(self, pred, gt)

    def ndsm2cls(self, ndsm):
        ndsm = torch.clamp(ndsm, min=0, max=self.ndsm_max)
        flag = (ndsm >= self.cls_edges[None, :-1, None, None]) * (ndsm <= self.cls_edges[None, 1:, None, None])
        return torch.argmax(flag.to(torch.float), dim=1, keepdim=True) 
    
    @staticmethod
    def _get_metric_params(cls, pred, gt, num_bin=101, interval=1):
        if cls.test_by_range:
            cls.num_cls = cls.cls_edges.shape[0] + 1
            gt.update({
                "ndsm_cls": cls.ndsm2cls(gt["ndsm"])
            })
            metric_params = {
                "sum": (gt["ndsm"]>=0).sum(),
                "ae": (torch.abs(gt["ndsm"] - pred["ndsm"]) * (gt["ndsm"]>=0)).sum(),
                "se": ((gt["ndsm"] - pred["ndsm"]) ** 2 * (gt["ndsm"]>=0)).sum()
            }
            gt_cls = gt["ndsm_cls"]
            gt_cls = F.one_hot(gt_cls.squeeze(1), num_classes=cls.num_cls).permute(0, 3, 1, 2).contiguous()
            sum_cls = gt_cls.sum(dim=-1).sum(dim=-1).sum(dim=0)
            ae_cls = (torch.abs(gt["ndsm"] - pred["ndsm"]) * gt_cls).sum(dim=-1).sum(dim=-1).sum(dim=0)
            se_cls = ((gt["ndsm"] - pred["ndsm"])**2 *gt_cls).sum(dim=-1).sum(dim=-1).sum(dim=0)
            metric_params.update({"sum{:02d}".format(i): sum_cls[i] for i in range(cls.num_cls)})
            metric_params.update({"ae{:02d}".format(i): ae_cls[i] for i in range(cls.num_cls)})
            metric_params.update({"se{:02d}".format(i): se_cls[i] for i in range(cls.num_cls)})

            sum8 = (gt_cls[:, :1] * (gt["ndsm"]>0)).sum()
            ae8 = (torch.abs(gt["ndsm"] - pred["ndsm"]) * gt_cls[:, :1] * (gt["ndsm"]>0)).sum()
            se8 = ((gt["ndsm"] - pred["ndsm"])**2 *gt_cls[:, :1] * (gt["ndsm"]>0)).sum()
            metric_params.update({
                "sum00>0": sum8,
                "ae00>0": ae8,
                "se00>0": se8
            })
        else:
            stats = torch.load(cls.stats_file).long().to(pred["ndsm"].device)
            cls.num_cls = stats.max() + 1
            metric_params = {
                "sum": (gt["ndsm"]>=0).sum(),
                "ae": (torch.abs(gt["ndsm"] - pred["ndsm"]) * (gt["ndsm"]>=0)).sum(),
                "se": ((gt["ndsm"] - pred["ndsm"]) ** 2 * (gt["ndsm"]>=0)).sum()
            }
            gt_int = torch.clamp(gt["ndsm"], min=0, max=398).long()
            gt_cls = stats[gt_int].squeeze(1)
            gt_cls = F.one_hot(gt_cls, num_classes=cls.num_cls).permute(0, 3, 1, 2).contiguous()
            sum_cls = gt_cls.sum(dim=-1).sum(dim=-1).sum(dim=0)
            ae_cls = (torch.abs(gt["ndsm"] - pred["ndsm"]) * gt_cls).sum(dim=-1).sum(dim=-1).sum(dim=0)
            se_cls = ((gt["ndsm"] - pred["ndsm"])**2 *gt_cls).sum(dim=-1).sum(dim=-1).sum(dim=0)
            metric_params.update({"sum{:02d}".format(i): sum_cls[i] for i in range(cls.num_cls)})
            metric_params.update({"ae{:02d}".format(i): ae_cls[i] for i in range(cls.num_cls)})
            metric_params.update({"se{:02d}".format(i): se_cls[i] for i in range(cls.num_cls)})

            sum8 = (gt_cls[:, :1] * (gt["ndsm"]>0)).sum()
            ae8 = (torch.abs(gt["ndsm"] - pred["ndsm"]) * gt_cls[:, :1] * (gt["ndsm"]>0)).sum()
            se8 = ((gt["ndsm"] - pred["ndsm"])**2 *gt_cls[:, :1] * (gt["ndsm"]>0)).sum()
            metric_params.update({
                "sum00>0": sum8,
                "ae00>0": ae8,
                "se00>0": se8
            })

        if cls.test:
            assert "mask" in gt, "During test, Ground Truth should contain masks."
            assert gt["mask"].shape[0] == 1, "During test, batch size should be 1."
            for i in range(cls.num_landcover):
                mask = gt["mask"] == i
                metric_params.update({
                    "sum_lc{:02d}".format(i): (mask * (gt["ndsm"]>=0)).sum(),
                    "ae_lc{:02d}".format(i): (torch.abs(gt["ndsm"] - pred["ndsm"]) * mask * (gt["ndsm"]>=0)).sum(),
                    "se_lc{:02d}".format(i): ((gt["ndsm"] - pred["ndsm"])**2 * mask * (gt["ndsm"]>=0)).sum()
                })

            building_index = cls.landcovers.index("building")
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
    
    def get_metric_params(self, pred, gt):
        return self._get_metric_params(self, pred, gt)
    
    @staticmethod
    def _evaluate(cls, eval_dict):
        eval_res = {
            "mae": eval_dict["ae"] / eval_dict["sum"],
            "rmse": torch.sqrt(eval_dict["se"] / eval_dict["sum"]),
        }
        eval_res.update({"mae{:02d}".format(i): eval_dict["ae{:02d}".format(i)] / eval_dict["sum{:02d}".format(i)] for i in range(cls.num_cls)})
        eval_res.update({"rmse{:02d}".format(i): torch.sqrt(eval_dict["se{:02d}".format(i)] / eval_dict["sum{:02d}".format(i)]) for i in range(cls.num_cls)})
        if "ae08>0" in eval_dict:
            eval_res.update({
                "mae08>0": eval_dict["ae08>0"] / eval_dict["sum08>0"],
                "rmse08>0": torch.sqrt(eval_dict["se08>0"] / eval_dict["sum08>0"])
            })

        if "ae00>0" in eval_dict:
            eval_res.update({
                "mae00>0": eval_dict["ae00>0"] / eval_dict["sum00>0"],
                "rmse00>0": torch.sqrt(eval_dict["se00>0"] / eval_dict["sum00>0"])
            })
        if cls.test:
            eval_res.update({
                "mae_per_building": eval_dict["ae_per_building"] / eval_dict["sum_per_building"],
                "rmse_per_building": torch.sqrt(eval_dict["se_per_building"] / eval_dict["sum_per_building"]),
                "per_building_bin_count": eval_dict["per_building_bin_count"],
                "per_building_bin_rmse": torch.sqrt(eval_dict["per_building_bin_se"] / eval_dict["per_building_bin_count"])
            })
            eval_res.update({
                "ae_bincount": eval_dict["ae_bincount"]
            })

            for i in range(cls.num_landcover):
                eval_res.update({
                    "mae_lc{:02d}".format(i): eval_dict["ae_lc{:02d}".format(i)] / eval_dict["sum_lc{:02d}".format(i)],
                    "rmse_lc{:02d}".format(i): torch.sqrt(eval_dict["se_lc{:02d}".format(i)] / eval_dict["sum_lc{:02d}".format(i)])
                })
        return eval_res

    def evaluate(self, eval_dict):
        return self._evaluate(self, eval_dict)

    @staticmethod
    def _vis(cls, image, pred, gt, validation=False, image_idx=None):
        prefix = 'val/' if validation else 'train/'
        max_h = max(gt['ndsm'][0].max(), pred["ndsm"][0].detach().max())
        return {
            prefix+'image': wandb.Image(image[0].cpu()),
            prefix+'ndsm': {
                'true': wandb.Image((gt['ndsm'][0]/max_h).float().cpu().numpy(), caption=image_idx),
                'pred': wandb.Image((pred['ndsm'][0]/max_h).detach().float().cpu().numpy(), caption=image_idx)
            }
        }
    
    def vis(self, image, pred, gt, validation=False, image_idx=None):
        return self._vis(self, image, pred, gt, validation, image_idx)

    def adjust_learning_rate(self, initial_lr, optimizer, iters, total_iters):
        return
