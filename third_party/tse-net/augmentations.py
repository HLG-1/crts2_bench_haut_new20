import random
import torchvision.transforms.functional as TF
import torchvision.transforms as tf
import os
import torch
from typing import List

class TransformWeak:
    def __init__(self, cfgs):
        pass

    def __call__(self, img, mask=None):
        img, mask = self.random_horizontal_flip(img, mask)
        img, mask = self.random_vertical_flip(img, mask)
        img, mask = self.random_rotate_90(img, mask)
        return img, mask

    def random_horizontal_flip(self, img, mask):
        if random.random() > 0.5:
            img = TF.hflip(img)
            if mask is not None:
                mask = TF.hflip(mask)
        return img, mask

    def random_vertical_flip(self, img, mask):
        if random.random() > 0.5:
            img = TF.vflip(img)
            if mask is not None:
                mask = TF.vflip(mask)
        return img, mask

    def random_rotate_90(self, img, mask):
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])  # Rotate by 90, 180, or 270 degrees
            img = TF.rotate(img, angle)
            if mask is not None:
                mask = TF.rotate(mask, angle)
        return img, mask

class TransformStrong:
    def __init__(self, cfgs):
        data_dir = cfgs.get("data_dir", "/data/gbh_old")
        use_basemap = cfgs.get("use_basemap", False)
        image_stats_file = os.path.join(data_dir, "image_stats.pickle") if not use_basemap else os.path.join(data_dir, "basemap_stats.pickle")
        self.mean, self.std = torch.load(image_stats_file)
        self.enable_spectral_aug = cfgs.get("enable_spectral_aug", True)

        self.unnormalize = tf.Normalize(
            mean = [-m / s for m, s in zip(self.mean, self.std)],
            std = [1 / s for s in self.std]
        )

        self.normalize = tf.Normalize(mean=self.mean, std=self.std)

        self.gamma_min = 0.9
        self.gamma_max = 1.1

        self.brightness_min = 0.9
        self.brightness_max = 1.1

        self.contrast_min = 0.9
        self.contrast_max = 1.1

        self.blur_min = 0.1
        self.blur_max = 2

    def __call__(self, img, mask=None):
        return self.transform(img, mask)

    def transform(self, img, mask):
        img, mask = self.random_rotate(img, mask)
        img, mask = self.random_crop(img, mask) 

        if self.enable_spectral_aug:
            img = self.unnormalize(img)
            img = torch.clamp(img, min=0, max=255)
            img = img / 255.
            img = self.random_gamma(img)
            img = self.random_brightness(img)
            img = self.random_contrast(img)
            img = self.gaussian_blur(img)
            img = img * 255.

            valid_per_channel = (img >= 0) & (img <= 255)
            valid_mask = valid_per_channel.all(dim=1, keepdim=True)
            img = self.normalize(img)
        return img, mask, valid_mask

    def random_rotate(self, img, mask):
        if random.random() > 0.5:
            angle = random.random() * 360.
            img = TF.rotate(img, angle, fill=0.)
            if mask is not None:
                if isinstance(mask, List):
                    mask = [TF.rotate(m, angle, fill=-1) for m in mask]
                else:
                    mask = TF.rotate(mask, angle, fill=-1)
        return img, mask

    def random_crop(self, img, mask):
        if random.random() > 0.5:
            img_width, img_height = img.shape[2:]
            output_width, output_height = img_width // 2, img_height // 2
            max_x = img_width - output_width
            max_y = img_height - output_width

            if max_x > 0 and max_y > 0:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)

            else:
                x, y = 0, 0

            img = TF.crop(img, y, x, output_width, output_height)
            if mask is not None:
                if isinstance(mask, List):
                    mask = [TF.crop(m, y, x, output_width, output_height) for m in mask]
                else:
                    mask = TF.crop(mask, y, x, output_width, output_height)
        return img, mask

    def random_gamma(self, img):
        if random.random() > 0.5:
            gamma = random.uniform(self.gamma_min, self.gamma_max)
            img = TF.adjust_gamma(img, gamma)
        return img

    def random_brightness(self, img):
        if random.random() > 0.5:
            brightness_factor = random.uniform(self.brightness_min, self.brightness_max)
            img = TF.adjust_brightness(img, brightness_factor)
        return img

    def random_contrast(self, img):
        if random.random() > 0.5:
            contrast_factor = random.uniform(self.contrast_min, self.contrast_max)
            img = TF.adjust_contrast(img, contrast_factor)
        return img

    def gaussian_blur(self, img):
        if random.random() > 0.5:
            blur_radius = random.uniform(self.blur_min, self.blur_max)
            if blur_radius > 0:
                img = TF.gaussian_blur(img, kernel_size=(5, 5), sigma=blur_radius)
        return img