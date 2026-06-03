"""
Pascal VOC dataset loader for class-agnostic center detection.
Handles automatic download and preprocessing of VOC2007 dataset.
"""
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision import transforms
from PIL import Image


class VOCCenterDataset(Dataset):
    """Pascal VOC dataset for center detection training."""
    
    def __init__(self, root="./vocdata", year="2007", image_set="trainval", size=512, download=True):
        """
        Initialize VOC dataset.
        
        Args:
            root: Directory to store VOC data
            year: VOC year (2007 or 2012)
            image_set: 'trainval', 'train', 'val', or 'test'
            size: Input image size (will be letterboxed)
            download: Whether to download if not present
        """
        self.ds = VOCDetection(root=root, year=year, image_set=image_set, download=download)
        self.size = size
        self.tf = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)

    def __len__(self):
        return len(self.ds)

    def _letterbox(self, im):
        """
        Letterbox image to square size while maintaining aspect ratio.
        
        Args:
            im: PIL Image
            
        Returns:
            tuple: (resized_image, canvas, scale, pad_x, pad_y)
        """
        w, h = im.size
        s = min(self.size / w, self.size / h)
        nw, nh = int(w * s), int(h * s)
        imr = im.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        pw, ph = (self.size - nw) // 2, (self.size - nh) // 2
        canvas.paste(imr, (pw, ph))
        return imr, canvas, s, pw, ph

    def _boxes(self, ann):
        """
        Extract bounding boxes from VOC annotation.
        
        Args:
            ann: VOC annotation dict
            
        Returns:
            np.array: (N, 4) array of [x1, y1, x2, y2] boxes
        """
        objs = ann['annotation'].get('object', [])
        if not isinstance(objs, list):
            objs = [objs]
        
        B = []
        for o in objs:
            b = o['bndbox']
            x1 = float(b['xmin'])
            y1 = float(b['ymin'])
            x2 = float(b['xmax'])
            y2 = float(b['ymax'])
            if x2 > x1 and y2 > y1:
                B.append([x1, y1, x2, y2])
        
        return np.array(B, dtype=np.float32) if B else np.zeros((0, 4), np.float32)

    def __getitem__(self, i):
        """
        Get dataset item.
        
        Args:
            i: Index
            
        Returns:
            tuple: (image_tensor, boxes_tensor)
        """
        im, ann = self.ds[i]
        im = self.tf(im)
        boxes = self._boxes(ann)
        
        imr, canvas, s, pw, ph = self._letterbox(im)
        
        # Transform boxes to letterboxed coordinates
        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * s + pw
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * s + ph
            boxes = np.clip(boxes, 0, self.size)
        
        return transforms.ToTensor()(canvas), torch.tensor(boxes, dtype=torch.float32)


def collate_fn(batch):
    """
    Custom collate function for VOC dataset.
    
    Args:
        batch: List of (image, boxes) tuples
        
    Returns:
        tuple: (images_tensor, boxes_list)
    """
    imgs, boxes = zip(*batch)
    return torch.stack(imgs, 0), list(boxes)

