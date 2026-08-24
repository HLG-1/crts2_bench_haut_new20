import os
import numpy as np
from tqdm import tqdm
import torch
from skimage import io
import torchvision.transforms as tfs

def get_train_val_dataloaders(cfgs):
    batch_size = int(cfgs.get('batch_size', 8))
    num_workers = int(cfgs.get('num_workers', 4))
    image_size = int(cfgs.get('image_size', 256))
    crop = cfgs.get('crop', None)

    normalize = bool(cfgs.get('normalize', True))
    use_basemap = bool(cfgs.get('use_basemap', False))

    data_dir = cfgs.get('data_dir', 'data/gbh/')
    data_train = cfgs.get("data_train", 'data/gbh/train.txt')
    data_val = cfgs.get("data_val", 'data/gbh/val.txt')

    if isinstance(data_train, str):
        data_train = [data_train]
    if isinstance(data_val, str):
        data_val = [data_val]

    if use_basemap:
        if not os.path.exists(os.path.join(data_dir, 'basemap_stats.pickle')):
            image_paths = get_image_list(data_dir, data_train+data_val, False, True)
            online_get_image_stats(data_dir, image_paths)
    else:
        if not os.path.exists(os.path.join(data_dir, 'image_stats.pickle')):
            image_paths = get_image_list(data_dir, data_train+data_val, False)
            online_get_image_stats(data_dir, image_paths)

    if not os.path.exists(os.path.join(data_dir, 'ndsm_stats.pickle')):
        ndsm_paths = get_image_list(data_dir, data_train+data_val, True)
        online_get_ndsm_stats(data_dir, ndsm_paths)
    
    overfit = bool(cfgs.get('overfit', False))
    train_loader = val_loader = None
    
    get_loader = lambda **kargs: get_tri_image_loader(**kargs, data_dir=data_dir, image_size=image_size, \
        crop=crop, num_workers=num_workers, normalize=normalize, overfit=overfit,
        use_basemap=use_basemap)

    assert os.path.isdir(data_dir), "Data directory does not exist: %s" %data_dir
    print(f"Loading training data from {data_train}")
    train_loader = get_loader(data_split=data_train, is_validation=False, batch_size=batch_size)
    print(f"Loading validation data from {data_val}")
    if overfit:
        val_loader = train_loader
    else:
        val_loader = get_loader(data_split=data_val, is_validation=True, batch_size=batch_size)

    return train_loader, val_loader

def get_pseudo_dataloaders(cfgs):
    batch_size = int(cfgs.get('batch_size', 8))
    num_workers = int(cfgs.get('num_workers', 4))
    image_size = int(cfgs.get('image_size', 256))
    crop = cfgs.get('crop', None)

    normalize = bool(cfgs.get('normalize', True))
    use_basemap = bool(cfgs.get('use_basemap', False))

    data_dir = cfgs.get('unlabeled_data_dir', 'data/gbh_old/')
    data_train = cfgs.get("unlabeled_data_train", 'splits/gbh/split2+/train.txt')

    if isinstance(data_train, str):
        data_train = [data_train]

    if use_basemap:
        if not os.path.exists(os.path.join(data_dir, 'basemap_stats.pickle')):
            image_paths = get_image_list(data_dir, data_train+data_val, False, True)
            online_get_image_stats(data_dir, image_paths)
    else:
        if not os.path.exists(os.path.join(data_dir, 'image_stats.pickle')):
            image_paths = get_image_list(data_dir, data_train+data_val, False)
            online_get_image_stats(data_dir, image_paths)

    if not os.path.exists(os.path.join(data_dir, 'ndsm_stats.pickle')):
        ndsm_paths = get_image_list(data_dir, data_train, True)
        online_get_ndsm_stats(data_dir, ndsm_paths)
    
    overfit = bool(cfgs.get('overfit', False))
    train_loader = None
    
    get_loader = lambda **kargs: get_tri_image_loader(**kargs, data_dir=data_dir, image_size=image_size, \
        crop=crop, num_workers=num_workers, normalize=normalize, overfit=overfit, use_basemap=use_basemap)

    assert os.path.isdir(data_dir), "Data directory does not exist: %s" %data_dir
    print(f"Loading unlabeled training data from {data_train}")
    train_loader = get_loader(data_split=data_train, is_validation=False, batch_size=batch_size)

    return train_loader


def get_test_dataloaders(cfgs):
    num_workers = int(cfgs.get('num_workers', 4))
    image_size = int(cfgs.get('image_size', 256))
    crop = cfgs.get('crop', None)
    normalize = bool(cfgs.get('normalize', True))
    use_basemap = bool(cfgs.get('use_basemap', False))

    data_dir = cfgs.get('data_dir', 'data/gbh/')
    data_test = cfgs.get("data_test", None)

    if not isinstance(data_test, dict):
        data_test = {"test": data_test}

    test_loader = {}
    get_loader = lambda **kargs: get_tri_image_loader(**kargs, data_dir=data_dir, image_size=image_size, \
        crop=crop, normalize=normalize, num_workers=num_workers, batch_size=1, use_basemap=use_basemap)

    assert os.path.isdir(data_dir), "Data directory does not exist: %s" %data_dir
    print(f"Loading testing data from {data_test}")
    for name, data in data_test.items():
        test_loader.update({name: get_loader(data_split=[data], is_validation=True)})
    return test_loader

def online_get_image_stats(data_dir, image_paths):
    sum_x = np.zeros(3)
    sum_x2 = np.zeros(3)
    sum_size = 0
    for image_path in tqdm(image_paths, desc="Computing image stats ..."):
        img = io.imread(image_path)[:, :, :3].astype(int).reshape(-1,3)
        sum_x += img.sum(axis=0)
        sum_x2 += (img*img).sum(axis=0)
        sum_size += img[:, 0].size
    mean = sum_x / sum_size
    std = list(np.sqrt(sum_x2/sum_size - mean*mean))
    mean = list(mean)
    if "_BAM" in image_paths[0]:
        torch.save([mean, std], os.path.join(data_dir, 'basemap_stats.pickle'))
    else:
        torch.save([mean, std], os.path.join(data_dir, 'image_stats.pickle'))
    return mean, std

def online_get_ndsm_stats(data_dir, image_paths):
    sum_x = 0
    sum_x2 = 0
    sum_size = 0
    minh = 0
    maxh = 0
    for image_path in tqdm(image_paths, desc="Computing ndsm stats ..."):
        img = io.imread(image_path).flatten()
        img = np.nan_to_num(img)
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
        img = np.clip(np.floor(img), a_min=0, a_max=None).astype(np.int64)
        res = np.bincount(img)
        count[:res.size] += res
        
    torch.save([mean, std, minh, maxh, count], os.path.join(data_dir, 'ndsm_stats.pickle'))
    return mean, std, minh, maxh, count


class GBHDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_mode, image_size=256, crop=None, normalize=True, is_validation=False,\
        overfit=False, use_basemap=False):
        super(GBHDataset, self).__init__()
        self.root = data_dir
        self.paths = make_gbh_dataset(data_dir, data_mode, overfit=overfit, use_basemap=use_basemap)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.normalize = normalize
        self.use_basemap = use_basemap
        image_stats_file = os.path.join(data_dir, 'image_stats.pickle') if not self.use_basemap else os.path.join(data_dir, "basemap_stats.pickle")
        self.mean, self.std = torch.load(image_stats_file)

    def transform(self, img):
        if img.shape[0]>3:
            img = img[:3]
        if self.crop:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop, antialias=None)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop, antialias=None)
        if self.normalize:
            img = tfs.functional.normalize(img, self.mean, self.std)
        return img

    def __getitem__(self, index):
        image_path, ndsm_path, mask_path = self.paths[index]
        file_idx = os.path.basename(image_path).split('_IMG')[0] if not self.use_basemap else os.path.basename(image_path).split('_BAM')[0]
        image = torch.tensor(io.imread(image_path).astype(np.float32).transpose(2, 0, 1))

        if os.path.exists(ndsm_path):
            ndsm = np.nan_to_num(np.float32(io.imread(ndsm_path)))
        gt_dict = {"ndsm": torch.tensor(ndsm)[None, :, :].clamp(0, None)}

        if os.path.exists(mask_path):
            mask = np.float32(io.imread(mask_path))
        else:
            mask = np.zeros_like(ndsm)
        gt_dict.update({"mask": torch.tensor(mask)[None, :, :]})

        return file_idx, self.transform(image), gt_dict

    def __len__(self):
        return self.size

    def name(self):
        return 'GBHDataset'

## multiple image dataset with every subfolder in the data_dir ##
def make_gbh_dataset(dir, mode, overfit=False, use_basemap=False):
    folder_name = ['image', 'ndsm', 'mask'] if not use_basemap else ["basemap", "ndsm", "mask"]
    file_suffix = ['_IMG.tif', '_AGL.tif', '_BLG.tif'] if not use_basemap else ["_BAM.tif", "_AGL.tif", "_BLG.tif"]

    images = []
    with open(mode, 'r') as f:
        lines = f.readlines()
        if overfit:
            lines = lines[:2]
        for line in lines:
            image = []
            line = line.rstrip()
            for folder, suffix in zip(folder_name, file_suffix):
                image.append(os.path.join(dir, folder, line+suffix))
            images.append(image)
    return images

def get_image_list(dir, mode, ndsm=False, basemap=False):
    images = []
    for m in mode:
        with open(m, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if ndsm:
                    images.append(os.path.join(dir, 'ndsm', line+'_AGL.tif'))
                elif basemap:
                    images.append(os.path.join(dir, 'basemap', line+'_BAM.tif'))
                else:
                    images.append(os.path.join(dir, 'image', line+'_IMG.tif'))
    return images


def get_tri_image_loader(data_dir, data_split, is_validation=False, batch_size=8, num_workers=4, \
    image_size=256, crop=None, normalize=True, overfit=False, use_basemap=False, drop_last=False):
    datasets = []
    for ds in data_split:
        datasets.append(GBHDataset(data_dir, ds, image_size=image_size, crop=crop, normalize=normalize, \
        is_validation=is_validation, overfit=overfit, use_basemap=use_basemap))
    dataset = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )
    return loader

if __name__ == "__main__":
    data_dir = 'data/gbh_old'
    data_mode = '/repos/data/split1/train.txt'
    image_list = make_gbh_dataset(data_dir, data_mode)
    print(len(image_list), image_list[0])
