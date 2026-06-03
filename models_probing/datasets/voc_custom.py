"""
Custom VOC dataset loader that works with Kaggle downloaded data.
"""
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class CustomVOCCenterDataset(Dataset):
    """Custom VOC dataset loader for center detection."""
    
    def __init__(self, root="./vocdata", year="2007", image_set="trainval", size=512):
        """
        Initialize custom VOC dataset.
        
        Args:
            root: Directory containing VOC2007
            year: VOC year (2007)
            image_set: 'trainval', 'train', 'val', or 'test'
            size: Input image size (will be letterboxed)
        """
        self.root = os.path.join(root, "VOC2007")
        self.size = size
        self.tf = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        
        # Load image set
        if image_set == "trainval":
            imageset_file = os.path.join(self.root, "ImageSets", "Main", "train_test.txt")
        else:
            imageset_file = os.path.join(self.root, "ImageSets", "Main", f"{image_set}.txt")
        
        with open(imageset_file, 'r') as f:
            # Split by space and take first part (image ID)
            self.image_ids = [line.strip().split()[0] for line in f.readlines()]
        
        print(f"Loaded {len(self.image_ids)} images for {image_set}")

    def __len__(self):
        return len(self.image_ids)

    def _letterbox(self, im):
        """Letterbox image to square size while maintaining aspect ratio."""
        w, h = im.size
        s = min(self.size / w, self.size / h)
        nw, nh = int(w * s), int(h * s)
        imr = im.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        pw, ph = (self.size - nw) // 2, (self.size - nh) // 2
        canvas.paste(imr, (pw, ph))
        return imr, canvas, s, pw, ph

    def _parse_annotation(self, image_id):
        """Parse VOC annotation file and return boxes and class ids."""
        annotation_file = os.path.join(self.root, "Annotations", f"{image_id}.xml")
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        
        boxes = []
        labels = []
        # VOC 20 classes
        VOC_CLASSES = [
            'aeroplane','bicycle','bird','boat','bottle',
            'bus','car','cat','chair','cow',
            'diningtable','dog','horse','motorbike','person',
            'pottedplant','sheep','sofa','train','tvmonitor'
        ]
        name_to_id = {n:i for i,n in enumerate(VOC_CLASSES)}
        for obj in root.findall('object'):
            name = obj.find('name').text.strip()
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(name_to_id.get(name, -1))
        
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), np.int64)
        return boxes, labels

    def __getitem__(self, idx):
        """Get dataset item."""
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = os.path.join(self.root, "JPEGImages", f"{image_id}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.tf(image)
        
        # Load annotations
        boxes, labels = self._parse_annotation(image_id)
        
        # Letterbox image and transform boxes
        imr, canvas, s, pw, ph = self._letterbox(image)
        
        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * s + pw
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * s + ph
            boxes = np.clip(boxes, 0, self.size)
            # Normalize to [0,1] relative to 512 canvas
            boxes = boxes / float(self.size)
        
        return transforms.ToTensor()(canvas), torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch):
    """Custom collate function: returns images, boxes_list, labels_list."""
    # Support both (img, boxes) and (img, boxes, labels)
    imgs = []
    boxes_list = []
    labels_list = []
    for item in batch:
        if len(item) == 2:
            img, boxes = item
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            img, boxes, labels = item
        imgs.append(img)
        boxes_list.append(boxes)
        labels_list.append(labels)
    return torch.stack(imgs, 0), boxes_list, labels_list
