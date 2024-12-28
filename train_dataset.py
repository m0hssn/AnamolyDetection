import os
import glob
import random
import math
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np

class TrainDataset(Dataset):
    def __init__(self, image_dir, anomaly_source_path, resize, transforms_img, transforms_mask):
        self.image_dir = image_dir
        self.image_paths = sorted(glob.glob(f"{image_dir}/*.jpg"))
        self.anomaly_source_paths = sorted(glob.glob(f"{anomaly_source_path}/*/*.jpg"))
        self.resize = resize
        self.transforms_img = transforms_img
        self.transforms_mask = transforms_mask

        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and resize the image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.resize(image, dsize=(self.resize[0], self.resize[1]))

        # Case 1: No anomaly, return original image and mask
        if random.random() > 0.5:
            mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)
            image = self.transforms_img(image)
            mask = self.transforms_mask(mask)
            return image, mask, image

        # Case 2: Generate augmented anomaly image
        anomaly_source_img = cv2.imread(
            self.anomaly_source_paths[random.randint(0, len(self.anomaly_source_paths) - 1)]
        )
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize[0], self.resize[1]))

        # Apply augmentations
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[i] for i in aug_ind])
        anomaly_img_augmented = aug(image=anomaly_source_img)

        # Generate Perlin noise
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).item())
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).item())

        perlin_noise = rand_perlin_2d_np(
            (self.resize[0], self.resize[1]), (perlin_scalex, perlin_scaley)
        )
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, 1.0, 0.0)
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        # Create the augmented image
        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
        beta = torch.rand(1).item() * 0.8
        augmented_image = (
            image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * perlin_thr
        ).astype(np.float32)
        mask = perlin_thr.astype(np.float32)

        # Apply transformations
        augmented_image = mask * augmented_image + (1 - mask) * image
        image = self.transforms_img(image)
        mask = self.transforms_mask(mask)
        augmented_image = self.transforms_img(augmented_image)

        return image, mask, augmented_image