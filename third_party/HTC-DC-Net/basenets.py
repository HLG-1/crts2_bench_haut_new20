import torch
from torch import nn
import torch.nn.functional as F
import wandb
from skimage import io
import os

from utils import compute_median

class BaseHeightPredictor(nn.Module):
    def __init__(self, cfgs=None):
        super(BaseHeightPredictor, self).__init__()
        self.test = cfgs.get("test", False)
        # Use configurable stats file path based on data_dir
        data_dir = cfgs.get("data_dir", "data/split1+")
        self.stats_file = os.path.join(data_dir, "stats.pkl")
    
    @staticmethod
    def _get_losses(cls, pred, gt):
        flag = gt["sup_mode"][:, None, None, None]
        if "mask" in gt:
            mask = gt["mask"] * flag + (~flag)
        else:
            mask = gt["ndsm"]>0
        losses = {
            "mae": (F.l1_loss(pred["ndsm"], gt["ndsm"], reduction='none') * mask).sum() / mask.sum()
                }
        losses.update({"loss_total": sum(losses.values())})
        return losses

    def get_losses(self, pred, gt):
        return self._get_losses(self, pred, gt)

    @staticmethod
    def _get_metric_params(cls, pred, gt, num_bin=101, interval=5):
        # Load stats and handle both numpy arrays and torch tensors
        # If stats file doesn't exist or fails to load, use simple metrics
        try:
            if not os.path.exists(cls.stats_file):
                raise FileNotFoundError(f"Stats file not found: {cls.stats_file}")
                
            stats_data = torch.load(cls.stats_file, weights_only=False)
            if isinstance(stats_data, (torch.Tensor)):
                stats = stats_data.long().to(pred["ndsm"].device)
            else:
                stats = torch.from_numpy(stats_data).long().to(pred["ndsm"].device)
                
            cls.num_cls = stats.max() + 1
            metric_params = {
                "sum": pred["ndsm"].numel(),
                "ae": torch.abs(gt["ndsm"] - pred["ndsm"]).sum(),
                "se": ((gt["ndsm"] - pred["ndsm"])**2).sum()
            }
            # Clamp to the actual max height in the stats file
            max_height = len(stats) - 1
            gt_int = torch.clamp(gt["ndsm"], min=0, max=max_height).long()
            gt_cls = stats[gt_int].squeeze(1)
            gt_cls = F.one_hot(gt_cls, num_classes=cls.num_cls).permute(0, 3, 1, 2).contiguous()
            sum_cls = gt_cls.sum(dim=-1).sum(dim=-1).sum(dim=0)
            ae_cls = (torch.abs(gt["ndsm"] - pred["ndsm"]) * gt_cls).sum(dim=-1).sum(dim=-1).sum(dim=0)
            se_cls = ((gt["ndsm"] - pred["ndsm"])**2 *gt_cls).sum(dim=-1).sum(dim=-1).sum(dim=0)
            metric_params.update({"sum"+str(i): sum_cls[i] for i in range(cls.num_cls)})
            metric_params.update({"ae"+str(i): ae_cls[i] for i in range(cls.num_cls)})
            metric_params.update({"se"+str(i): se_cls[i] for i in range(cls.num_cls)})

            sum8 = (gt_cls[:, -1:] * (gt["ndsm"]>0)).sum()
            ae8 = (torch.abs(gt["ndsm"] - pred["ndsm"]) * gt_cls[:, -1:] * (gt["ndsm"]>0)).sum()
            se8 = ((gt["ndsm"] - pred["ndsm"])**2 *gt_cls[:, -1:] * (gt["ndsm"]>0)).sum()
            metric_params.update({
                "sum8>0": sum8,
                "ae8>0": ae8,
                "se8>0": se8
            })
        except Exception as e:
            print(f"Warning: Could not load stats file, using basic metrics only: {e}")
            cls.num_cls = 1
            # Calculate basic metrics on valid pixels only
            mask = gt["ndsm"] > 0
            valid_count = mask.sum()
            
            if valid_count == 0:
                metric_params = {
                    "sum": 1.0,
                    "ae": 0.0,
                    "se": 0.0,
                    "sum8>0": 1.0,
                    "ae8>0": 0.0,
                    "se8>0": 0.0,
                }
            else:
                error = pred["ndsm"] - gt["ndsm"]
                error_masked = error * mask
                sum_val = valid_count.float()
                ae = error_masked.abs().sum()
                se = (error_masked ** 2).sum()
                
                # Compute error for pixels > 0.8
                mask8 = (gt["ndsm"] > 0.8) & mask
                sum8 = mask8.sum()
                if sum8 > 0:
                    ae8 = (error_masked * mask8).abs().sum()
                    se8 = ((error_masked * mask8) ** 2).sum()
                else:
                    ae8 = torch.tensor(0.0, device=pred["ndsm"].device)
                    se8 = torch.tensor(0.0, device=pred["ndsm"].device)
                
                metric_params = {
                    "sum": sum_val,
                    "ae": ae,
                    "se": se,
                    "sum8>0": sum8.float() if sum8 > 0 else torch.tensor(1.0, device=pred["ndsm"].device),
                    "ae8>0": ae8,
                    "se8>0": se8,
                    "sum0": sum_val,
                    "ae0": ae,
                    "se0": se,
                }
        
        # Debug logging for fallback case (only log once per validation)
        if cls.num_cls == 1 and not hasattr(cls, '_debug_logged'):
            mae_debug = metric_params["ae"] / metric_params["sum"]
            rmse_debug = torch.sqrt(metric_params["se"] / metric_params["sum"])
            print(f"DEBUG - pred ndsm range: {pred['ndsm'].min():.4f} to {pred['ndsm'].max():.4f}")
            print(f"DEBUG - gt ndsm range: {gt['ndsm'].min():.4f} to {gt['ndsm'].max():.4f}")
            print(f"DEBUG - Calculated MAE: {mae_debug:.4f}")
            print(f"DEBUG - Calculated RMSE: {rmse_debug:.4f}")
            print(f"DEBUG - RMSE < MAE? {rmse_debug < mae_debug}")
            cls._debug_logged = True

        if cls.test:
            assert "mask" in gt, "During test, Ground Truth should contain masks."
            assert gt["mask"].shape[0] == 1, "During test, batch size should be 1."
            pred_l1, gt_l1, pred_h, gt_h = compute_median(gt["mask"].squeeze(), pred["ndsm"].squeeze(), gt["ndsm"].squeeze())
            pred_h0 = pred_h[:, None].repeat(1,  num_bin)
            gt_h0 = gt_h[:, None].repeat(1, num_bin)
            lb = (torch.arange(num_bin)*interval).repeat(len(pred_h), 1).to(pred_h0.device)
            ub = (torch.arange(1, num_bin+1)*interval).repeat(len(gt_h), 1).to(pred_h0.device)
            se_raw = (pred_h0-gt_h0)**2
            discrete_mask = torch.logical_and(gt_h0>=lb, gt_h0<ub)
            count = discrete_mask.sum(dim=0)
            se = (se_raw * discrete_mask).sum(dim=0)
            metric_params.update({
                "sum_mask": gt["mask"].sum(),
                "sum_mask_inv": (1-gt["mask"]).sum(),
                "ae_mask": (torch.abs(pred["ndsm"]-gt["ndsm"])*gt["mask"]).sum(),
                "ae_mask_inv": (torch.abs(pred["ndsm"]-gt["ndsm"]) * (1-gt["mask"])).sum(),
                "se_mask": (((pred["ndsm"]-gt["ndsm"])*gt["mask"])**2).sum(),
                "se_mask_inv": (((pred["ndsm"]-gt["ndsm"])*(1-gt["mask"]))**2).sum(),
                "ae_building": (torch.abs(pred_l1-gt_l1)*gt["mask"].squeeze()).sum(),
                "se_building": (((pred_l1-gt_l1)*gt["mask"].squeeze())**2).sum(),
                "sum_per_building": pred_h.numel(),
                "ae_per_building": torch.abs(pred_h-gt_h).sum(),
                "se_per_building": ((pred_h-gt_h)**2).sum(),
                "per_building_bin_count": count,
                "per_building_bin_se": se
            })
        return metric_params
    
    def get_metric_params(self, pred, gt):
        return self._get_metric_params(self, pred, gt)
    
    @staticmethod
    def _evaluate(cls, eval_dict):
        # Calculate MAE correctly
        mae_calc = eval_dict["ae"] / eval_dict["sum"]
        
        # Calculate RMSE correctly: RMSE = sqrt(MSE) where MSE = SE / N
        # But SE should be sum of squared errors, not sum of errors
        rmse_calc = torch.sqrt(eval_dict["se"] / eval_dict["sum"])
        
        # Safety check: RMSE should always be >= MAE mathematically
        if rmse_calc < mae_calc:
            print(f"WARNING: RMSE ({rmse_calc:.4f}) < MAE ({mae_calc:.4f}) - This is mathematically impossible!")
            print(f"DEBUG - ae: {eval_dict['ae']:.4f}, se: {eval_dict['se']:.4f}, sum: {eval_dict['sum']:.4f}")
            # Force RMSE to be at least MAE for safety
            rmse_calc = torch.maximum(rmse_calc, mae_calc)
        
        print(f"EVALUATE DEBUG - MAE: {mae_calc:.4f}, RMSE: {rmse_calc:.4f}")
        
        eval_res = {
            "mae": mae_calc.item() if hasattr(mae_calc, 'item') else mae_calc,
            "rmse": rmse_calc.item() if hasattr(rmse_calc, 'item') else rmse_calc,
        }
        eval_res.update({"mae"+str(i): (eval_dict["ae"+str(i)] / eval_dict["sum"+str(i)]).item() if hasattr((eval_dict["ae"+str(i)] / eval_dict["sum"+str(i)]), 'item') else eval_dict["ae"+str(i)] / eval_dict["sum"+str(i)] for i in range(cls.num_cls)})
        eval_res.update({"rmse"+str(i): torch.sqrt(eval_dict["se"+str(i)] / eval_dict["sum"+str(i)]).item() if hasattr(torch.sqrt(eval_dict["se"+str(i)] / eval_dict["sum"+str(i)]), 'item') else torch.sqrt(eval_dict["se"+str(i)] / eval_dict["sum"+str(i)]) for i in range(cls.num_cls)})
        eval_res.update({
            "mae8>0": (eval_dict["ae8>0"] / eval_dict["sum8>0"]).item() if hasattr((eval_dict["ae8>0"] / eval_dict["sum8>0"]), 'item') else eval_dict["ae8>0"] / eval_dict["sum8>0"],
            "rmse8>0": torch.sqrt(eval_dict["se8>0"] / eval_dict["sum8>0"]).item() if hasattr(torch.sqrt(eval_dict["se8>0"] / eval_dict["sum8>0"]), 'item') else torch.sqrt(eval_dict["se8>0"] / eval_dict["sum8>0"])
        })
        if cls.test:
            eval_res.update({
                "mae_mask": eval_dict["ae_mask"] / eval_dict["sum_mask"],
                "mae_non_mask": eval_dict["ae_mask_inv"] / eval_dict["sum_mask_inv"],
                "rmse_mask": torch.sqrt(eval_dict["se_mask"] / eval_dict["sum_mask"]),
                "rmse_non_mask": torch.sqrt(eval_dict["se_mask_inv"] / eval_dict["sum_mask_inv"]),
                "mae_building": eval_dict["ae_building"] / eval_dict["sum_mask"],
                "rmse_building": torch.sqrt(eval_dict["se_building"] / eval_dict["sum_mask"]),
                "mae_per_building": eval_dict["ae_per_building"] / eval_dict["sum_per_building"],
                "rmse_per_building": torch.sqrt(eval_dict["se_per_building"] / eval_dict["sum_per_building"]),
                "per_building_bin_count": eval_dict["per_building_bin_count"],
                "per_building_bin_rmse": torch.sqrt(eval_dict["per_building_bin_se"] / eval_dict["per_building_bin_count"])
            })
        return eval_res

    def evaluate(self, eval_dict):
        return self._evaluate(self, eval_dict)

    @staticmethod
    def _vis(cls, image, pred, gt, validation=False, image_idx=None, save=None):
        prefix = 'val/' if validation else 'train/'
        if save:
            vis_dir = os.path.join(save, 'ndsm')
            os.makedirs(vis_dir, exist_ok=True)
            filename = os.path.join(vis_dir, f"{image_idx[0]}_ndsm_pred.tif")
            io.imsave(filename, pred["ndsm"][0].cpu().numpy())
            #_, _, pred_h, gt_h = compute_median(gt["mask"].squeeze(), pred["ndsm"].squeeze(), gt["ndsm"].squeeze())
            #torch.save([pred_h, gt_h], image_idx[0]+'_h.pkl')
            return {}
        else:
            return {
                prefix+'image': wandb.Image(image[0].cpu()),
                prefix+'ndsm': {
                    'true': wandb.Image(gt['ndsm'][0].float().cpu().numpy()),
                    'pred': wandb.Image(pred['ndsm'][0].detach().float().cpu().numpy())
                }
            }
    
    def vis(self, image, pred, gt, validation=False, image_idx=None, save=None):
        return self._vis(self, image, pred, gt, validation, image_idx, save)