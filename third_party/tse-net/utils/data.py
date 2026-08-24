import numpy as np
import torch
from skimage.measure import label
import os

def compute_median(mask, pred, gt):
    pred_l1 = torch.zeros_like(pred, device=pred.device)
    gt_l1 = torch.zeros_like(pred, device=pred.device)
    pred_h = []
    gt_h = []
    mask = mask.cpu().numpy().astype(np.uint16)
    if mask.max() > 2:
        num_instances = mask.max()
        for i in range(num_instances):
            m = torch.tensor(mask == i+1, device=pred.device)
            if m.max()<1:
                continue
            h_gt = torch.median(gt[m>0])
            h_pred = torch.median(pred[m>0])
            pred_l1 += (h_pred * m)
            gt_l1 += (h_gt * m)
            pred_h.append(h_pred)
            gt_h.append(h_gt)
    elif mask.max() == 1:
        instances, num_instances = label(mask, return_num=True)
        for i in range(num_instances):
            m = torch.tensor(instances == i+1, device=pred.device)
            if m.max()<1:
                continue
            h_gt = torch.median(gt[m>0])
            h_pred = torch.median(pred[m>0])
            pred_l1 += (h_pred * m)
            gt_l1 += (h_gt * m)
            pred_h.append(h_pred)
            gt_h.append(h_gt)
    if len(pred_h) == 0:
        return pred_l1, gt_l1, torch.zeros(0).to(pred_l1.device), torch.zeros(0).to(pred_l1.device)
    else:
        return pred_l1, gt_l1, torch.stack(pred_h), torch.stack(gt_h)

def get_bicls_files(cfgs):
    data_dir = cfgs.get("data_dir", "/data/gbh_old")
    qvs_index = cfgs.get("data_train").split(".txt")[0].split("train")[1]
    dataset_name = cfgs.get("data_train").split("/")[1]
    if dataset_name == "vaihingen":
        qvs_file = f"qvs_vaihingen_{qvs_index}.npy"
        ndsm_stats_file = os.path.join(data_dir, f"bi_ndsm_stats_{qvs_index}.pickle")
        ns_file = f"samples_vaihingen_{qvs_index}.npy"
    elif dataset_name == "gbh1+":
        qvs_file = f"qvs_gbh1+_{qvs_index}.npy"
        ndsm_stats_file = os.path.join(data_dir, f"bi_ndsm_stats_{qvs_index}.pickle")
        ns_file = f"samples_gbh1+_{qvs_index}.npy"
    elif dataset_name == "syn":
        qvs_file = f"qvs_syn_{qvs_index}.npy"
        ndsm_stats_file = os.path.join(data_dir, f"bi_ndsm_stats_{qvs_index}.pickle")
        ns_file = f"samples_syn_{qvs_index}.npy"
    else:
        raise NotImplementedError("This dataset is not supported!")
    return qvs_file, ndsm_stats_file, ns_file
