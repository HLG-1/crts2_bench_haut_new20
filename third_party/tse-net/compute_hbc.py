import os
import torch
import numpy as np
import math
from skimage import io
from tqdm import tqdm

def online_get_ndsm_stats(data_dir, image_paths, proportion):
    sum_x = 0
    sum_x2 = 0
    sum_size = 0
    minh = 0
    maxh = 0
    for image_path in tqdm(image_paths, desc="Computing ndsm stats ..."):
        img = io.imread(image_path).flatten()
        img = np.nan_to_num(img)
        # img = img[img>0]
        img = np.clip(img, a_min=0, a_max=None)
        sum_x += img.sum()
        sum_x2 += (img**2).sum()
        sum_size += img.size

        minh = img.min() if img.min()<minh else minh
        maxh = img.max() if img.max()>maxh else maxh
    minh = 0 if minh < 0 else minh
    mean = sum_x / sum_size
    std = np.sqrt(sum_x2/sum_size - mean**2)
    
    count = np.zeros(int(np.ceil(maxh)))
    for image_path in image_paths:
        img = io.imread(image_path).flatten()
        img = np.nan_to_num(img)
        # img = img[img>0]
        # img = np.clip(img, a_min=0)
        img = np.clip(np.floor(img), a_min=0, a_max=None).astype(np.int64)
        res = np.bincount(img)
        count[:res.size] += res
        
    torch.save([mean, std, minh, maxh, count], os.path.join(data_dir, f"bi_ndsm_stats_{proportion}.pickle"))
    return mean, std, minh, maxh, count

def get_image_list(dir, mode, ndsm=False, vis=False, use_basemap=False):
    images = []
    for m in mode:
        with open(m, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if ndsm:
                    images.append(os.path.join(dir, 'ndsm', line+'_AGL.tif'))
                elif vis:
                    images.append(os.path.join(dir, 'visualization', line+'_VIS.tif'))
                elif use_basemap:
                    images.append(os.path.join(dir, 'basemap', line+'_BAM.tif'))
                else:
                    images.append(os.path.join(dir, 'image', line+'_IMG.tif'))
    return images

def get_element_from_array(array, index):
    sorted_array = np.sort(array)
    if (index % 1) == 0:
        return sorted_array[int(index-1)]
    else:
        ind1 = math.floor(index)
        ind2 = math.ceil(index)
        return (sorted_array[ind1] + sorted_array[ind2]) * 0.5

def discrete_bin(ndsm_list, lower_edge, upper_edge, order):
    count = np.zeros(10)
    edges = np.linspace(lower_edge, upper_edge, 11)
    edges = np.round(edges, decimals=order).astype(np.float64)
    times = 10 ** order
    for ndsm_path in ndsm_list:
        ndsm = io.imread(ndsm_path).flatten()
        ndsm = ndsm.astype(np.float64)
        ndsm = np.nan_to_num(ndsm)
        ndsm = np.clip(ndsm, a_min=0, a_max=None)
        # ndsm = ndsm[ndsm>0]

        mask = np.logical_and(1000*ndsm>=1000*lower_edge, 1000*ndsm<1000*upper_edge)
        if mask.sum() == 0:
            continue
        ndsm = np.floor((ndsm[mask]*1000-lower_edge*1000)/1000 * times).astype(np.int64)
        res = np.bincount(ndsm)
        # print(ndsm.size, res.shape, res.min(), res.max(), res.sum())
        count[:res.size] += res
    return count, edges

def collect_values(ndsm_list, lower_edge, upper_edge):
    value_list = []
    for ndsm_path in ndsm_list:
        ndsm = io.imread(ndsm_path).flatten()
        ndsm = ndsm.astype(np.float64)
        ndsm = np.nan_to_num(ndsm)
        ndsm = np.clip(ndsm, a_min=0, a_max=None)
        # ndsm = ndsm[ndsm>0]
        mask = np.logical_and(ndsm*1000>=lower_edge*1000, ndsm*1000<upper_edge*1000)
        if mask.sum() == 0:
            continue
        value_list.append(ndsm[mask])
        # print(ndsm[mask].shape)
    value = np.concatenate(value_list)
    return value
    
def get_quantile_value(ndsm_list, count, edges, cumcount, quantile_point, order):
    if np.all(count[1:]==0):
        return edges[0]  # If all counts are zero, return the lower edge
    bin_ind = np.argmax(cumcount >= quantile_point)
    lower_edge = edges[bin_ind]
    upper_edge = edges[bin_ind+1]
    # print(bin_ind, count, edges, cumcount, quantile_point, lower_edge, upper_edge)
    num = count[bin_ind]
    # print(quantile_point, cumcount[bin_ind-1] if bin_ind>0 else 0, num, count.sum())
    # print(count, num, quantile_point)
    if lower_edge == upper_edge:
        return lower_edge

    if num > 100000:
        new_count, new_edges = discrete_bin(ndsm_list, lower_edge, upper_edge, order)
        new_cumcount = np.cumsum(new_count)
        new_quantile_point = quantile_point - cumcount[bin_ind-1] if bin_ind>0 else quantile_point
        return get_quantile_value(ndsm_list, new_count, new_edges, new_cumcount, new_quantile_point, order+1)
    else:
        # print(bin_ind, cumcount, quantile_point, lower_edge, upper_edge)
        quantile_ind = quantile_point - cumcount[bin_ind-1] if bin_ind>0 else quantile_point
        array = collect_values(ndsm_list, lower_edge, upper_edge)
        return get_element_from_array(array, quantile_ind)

def main(ndsm_list, count, edges, quantile_points):
    cumcount = np.cumsum(count)
    qvs = []
    for i, quantile_point in enumerate(quantile_points):
        print(f"Getting the quantile point {i+1}!")
        qv = get_quantile_value(ndsm_list, count, edges, cumcount, quantile_point, order=1)
        qvs.append(qv)
    return qvs

if __name__ == "__main__":
    data_name = "vaihingen"

    split_dir = "./splits/" + data_name
    data_dir = "./data/" + data_name

    for m in ["train0.1", "train0.5", "train1", "train5", "train10", "train100"]:
        proportion = m.split("train")[1]
        mode = os.path.join(split_dir, m+".txt")
        ndsm_list = get_image_list(data_dir, [mode], ndsm=True)

        num_classes = 25
        quantile = np.arange(1, num_classes+1)
        quantile = 1 - 0.5 ** quantile

        ndsm_stats_file = os.path.join(data_dir, f"bi_ndsm_stats_{proportion}.pickle")
        _, _, _, maxh, count = online_get_ndsm_stats(data_dir, ndsm_list, proportion)
        quantile_points = np.floor(quantile*count.sum())
        sample_nums = np.diff(quantile_points)

        sample_filter = sample_nums > 100
        num_valid_quantile = sample_filter.sum()

        quantile_points = quantile_points[:num_valid_quantile+1]
        sample_nums = sample_nums[sample_filter]

        edges = np.linspace(0, math.ceil(maxh), math.ceil(maxh)+1)
        qvs = main(ndsm_list, count, edges, quantile_points)

        np.save(f"./samples_{data_name}_{proportion}.npy", sample_nums)
        np.save(f"./qvs_{data_name}_{proportion}.npy", qvs)
