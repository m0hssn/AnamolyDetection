import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class TestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms_img, transforms_mask):
        """
        Args:
            image_dir (str): Directory with images.
            mask_dir (str): Directory with corresponding masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transforms_img
        self.transform_mask = transforms_mask

        self.image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.png') or filename.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path))

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        return img, mask