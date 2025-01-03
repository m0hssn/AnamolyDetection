import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ValidationSet(Dataset):
    def __init__(self, image_paths, mask_dir, transforms_img, transforms_mask):
        """
        Args:
            image_paths (list): List of image file paths.
            mask_dir (str): Directory with corresponding masks.
            transforms_img (callable, optional): Optional transform to be applied on an image.
            transforms_mask (callable, optional): Optional transform to be applied on a mask.
        """
        self.mask_dir = mask_dir
        self.transform_img = transforms_img
        self.transform_mask = transforms_mask

        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))
        
        img = Image.open(img_path).convert("RGB")

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new('L', img.size, 0) 

        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        return img, mask
