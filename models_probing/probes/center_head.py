"""
Center detection head and training utilities.
Implements lightweight center-based object detection with focal loss.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


class CenterHead(nn.Module):
    """YOLO-style coupled head on top of a feature map.
    Shared tower then split into heatmap, wh, offset branches.
    """
    
    def __init__(self, in_ch, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
        )
        self.hm = nn.Conv2d(hidden, 1, 1)
        self.wh = nn.Conv2d(hidden, 2, 1)
        self.off = nn.Conv2d(hidden, 2, 1)

        # Initialize heatmap bias for focal loss
        with torch.no_grad():
            self.hm.bias.data.fill_(-2.19)
            # Initialize wh and off with small positive bias
            self.wh.bias.data.fill_(0.1)
            self.off.bias.data.fill_(0.0)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            tuple: (heatmap, width_height, offset)
        """
        # Cast input to match parameter dtype (float32)
        x = x.float()
        x = self.shared(x)
        return self.hm(x), self.wh(x), self.off(x)


def focal_loss_hm(pred, gt):
    """
    Focal loss for heatmap prediction.
    
    Args:
        pred: Predicted heatmap (B, 1, H, W)
        gt: Ground truth heatmap (B, 1, H, W)
        
    Returns:
        torch.Tensor: Focal loss
    """
    pred = torch.clamp(torch.sigmoid(pred), 1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    neg_w = torch.pow(1 - gt, 4)
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_w * neg
    
    npos = pos.sum()
    loss = -(pos_loss + neg_loss).sum()
    return loss / npos.clamp_min(1.0)


def gaussian2d(shape, sx=1, sy=1):
    """
    Generate 2D Gaussian kernel.
    
    Args:
        shape: (height, width) of kernel
        sx: Standard deviation in x
        sy: Standard deviation in y
        
    Returns:
        np.array: 2D Gaussian kernel
    """
    m, n = [(ss - 1.) / 2 for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    return np.exp(-(x*x/(2*sx*sx) + y*y/(2*sy*sy)))


def gaussian_radius(sz, min_overlap=0.7):
    """
    Calculate Gaussian radius for object size.
    
    Args:
        sz: (height, width) of object
        min_overlap: Minimum overlap threshold
        
    Returns:
        int: Gaussian radius
    """
    h, w = sz
    
    a1 = 1
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - np.sqrt(b1*b1 - 4*a1*c1)) / 2
    
    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 - np.sqrt(b2*b2 - 4*a2*c2)) / 2
    
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + np.sqrt(b3*b3 - 4*a3*c3)) / (2 * a3)
    
    return int(max(0, min(r1, r2, r3)))


def draw_gaussian(hm, cx, cy, r):
    """
    Draw Gaussian on heatmap.
    
    Args:
        hm: Heatmap array
        cx: Center x coordinate
        cy: Center y coordinate
        r: Gaussian radius
    """
    d = 2 * r + 1
    g = gaussian2d((d, d), d/6, d/6)
    
    x, y = int(cx), int(cy)
    H, W = hm.shape
    
    l = min(x, r)
    r2 = min(W - x - 1, r)
    t = min(y, r)
    b = min(H - y - 1, r)
    
    patch = hm[y-t:y+b+1, x-l:x+r2+1]
    g2 = g[r-t:r+b+1, r-l:r+r2+1]
    
    np.maximum(patch, g2, out=patch)


def build_targets(boxes_list, Hf, Wf, stride, max_objs=100):
    """
    Build training targets from bounding boxes.
    
    Args:
        boxes_list: List of bounding boxes for each image
        Hf: Feature map height
        Wf: Feature map width
        stride: Stride from input to feature map
        max_objs: Maximum objects per image
        
    Returns:
        tuple: (heatmap, indices, regression, width_height, mask)
    """
    B = len(boxes_list)
    hm = torch.zeros((B, 1, Hf, Wf), dtype=torch.float32)
    ind = torch.zeros((B, max_objs), dtype=torch.long)
    reg = torch.zeros((B, max_objs, 2), dtype=torch.float32)
    wh = torch.zeros((B, max_objs, 2), dtype=torch.float32)
    mask = torch.zeros((B, max_objs), dtype=torch.float32)
    
    for b, boxes in enumerate(boxes_list):
        num = min(boxes.shape[0], max_objs)
        for j in range(num):
            # boxes are normalized [0,1] relative to 512 canvas
            x1, y1, x2, y2 = boxes[j].tolist()
            # Convert to pixels at 512 then to feature cells via stride
            x1 *= 512; y1 *= 512; x2 *= 512; y2 *= 512
            
            w = max(1., (x2 - x1) / stride)
            h = max(1., (y2 - y1) / stride)
            cx = (x1 + x2) / 2 / stride
            cy = (y1 + y2) / 2 / stride
            
            icx, icy = int(cx), int(cy)
            if not (0 <= icx < Wf and 0 <= icy < Hf):
                continue
            
            r = gaussian_radius((math.ceil(h), math.ceil(w)))
            draw_gaussian(hm[b, 0].numpy(), icx, icy, r)
            
            ind[b, j] = icy * Wf + icx
            reg[b, j, 0] = cx - icx
            reg[b, j, 1] = cy - icy
            wh[b, j, 0] = w
            wh[b, j, 1] = h
            mask[b, j] = 1.0
    
    return hm, ind, reg, wh, mask


def _gather(feat, ind):
    """
    Gather features at specified indices.
    
    Args:
        feat: Features (B, C, HW)
        ind: Indices (B, N)
        
    Returns:
        torch.Tensor: Gathered features (B, N, C)
    """
    B, C, HW = feat.shape
    N = ind.shape[1]
    ind = ind.unsqueeze(1).expand(B, C, N)
    feat = torch.gather(feat, 2, ind)
    return feat.permute(0, 2, 1).contiguous()


def decode(hm, wh, reg, K=100, stride=16, thr=0.3):
    """
    Decode predictions to bounding boxes.
    
    Args:
        hm: Heatmap (B, 1, H, W)
        wh: Width-height (B, 2, H, W)
        reg: Offset (B, 2, H, W)
        K: Top-K detections
        stride: Feature stride
        thr: Confidence threshold
        
    Returns:
        list: List of (boxes, scores) for each image
    """
    B, _, H, W = hm.shape
    hm = torch.sigmoid(hm)
    
    # Non-maximum suppression
    keep = F.max_pool2d(hm, 3, 1, 1).eq(hm).float()
    hm = hm * keep
    
    scores, inds = torch.topk(hm.view(B, -1), K)
    ys = (inds // W).float()
    xs = (inds % W).float()
    
    wh = wh.view(B, 2, -1)
    reg = reg.view(B, 2, -1)
    
    wh_g = _gather(wh, inds).squeeze(1)
    reg_g = _gather(reg, inds).squeeze(1)
    
    xs = (xs + reg_g[..., 0]) * stride
    ys = (ys + reg_g[..., 1]) * stride
    # wh in feature cells → pixels, enforce positivity
    ws = (wh_g[..., 0] * stride).abs().clamp(1.0, 512.0)
    hs = (wh_g[..., 1] * stride).abs().clamp(1.0, 512.0)
    x1 = xs - ws / 2
    y1 = ys - hs / 2
    x2 = xs + ws / 2
    y2 = ys + hs / 2
    boxes = torch.stack([x1, y1, x2, y2], -1) / 512.0  # normalize to [0,1]
    
    out = []
    for b in range(B):
        sel = scores[b] > thr
        out.append((boxes[b][sel], scores[b][sel]))
    
    return out


def ap50(dataset_preds, dataset_gts):
    """
    Calculate AP@0.5 metric.
    
    Args:
        dataset_preds: List of (boxes, scores) predictions
        dataset_gts: List of ground truth boxes
        
    Returns:
        float: AP@0.5 score
    """
    all_scores = []
    all_tp = []
    all_fp = []
    num_gt = 0
    
    for (pb, ps), gt in zip(dataset_preds, dataset_gts):
        num_gt += gt.shape[0]
        
        if pb.numel() == 0:
            continue
        
        # Sort by confidence
        idx = torch.argsort(ps, descending=True)
        pb, ps = pb[idx], ps[idx]
        
        matched = torch.zeros(gt.shape[0], dtype=torch.bool, device=gt.device)
        ious = box_iou(pb, gt)
        
        for i in range(pb.shape[0]):
            all_scores.append(ps[i].item())
            
            if gt.shape[0] == 0:
                all_tp.append(0)
                all_fp.append(1)
                continue
            
            j = torch.argmax(ious[i])
            if ious[i, j] >= 0.5 and not matched[j]:
                matched[j] = True
                all_tp.append(1)
                all_fp.append(0)
            else:
                all_tp.append(0)
                all_fp.append(1)
    
    if len(all_scores) == 0:
        return 0.0
    
    # Calculate AP
    order = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[order]
    fp = np.array(all_fp)[order]
    
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    
    rec = tp_c / max(1, num_gt)
    prec = tp_c / np.maximum(1, tp_c + fp_c)
    
    ap = 0.0
    for rl in np.linspace(0, 1, 101):
        p = prec[rec >= rl].max() if np.any(rec >= rl) else 0
        ap += p / 101.0
    
    return float(ap)


def train_epoch(head, fp_fn, loader, device, hm_w=1.0, wh_w=0.1, off_w=1.0, lr=2e-3):
    """
    Train one epoch.
    
    Args:
        head: Center detection head
        fp_fn: Feature provider function
        loader: Data loader
        device: Device
        hm_w: Heatmap loss weight
        wh_w: Width-height loss weight
        off_w: Offset loss weight
        lr: Learning rate
        
    Returns:
        dict: Average losses
    """
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    
    avg = {"hm": 0.0, "wh": 0.0, "off": 0.0}
    steps = 0
    total_samples = len(loader.dataset)
    processed_samples = 0
    
    print(f"Starting training epoch...")
    print(f"Total samples: {total_samples}, Batches: {len(loader)}")
    
    for batch_idx, (imgs, boxes_list) in enumerate(loader):
        imgs = imgs.to(device)
        boxes_list = [b.to(device) for b in boxes_list]
        
        with torch.no_grad():
            feat, stride = fp_fn(imgs)  # (B, C, Hf, Wf), int
        
        B, C, Hf, Wf = feat.shape
        hm_gt, ind, reg_gt, wh_gt, mask = build_targets(boxes_list, Hf, Wf, stride)
        hm_gt, ind, reg_gt, wh_gt, mask = hm_gt.to(device), ind.to(device), reg_gt.to(device), wh_gt.to(device), mask.to(device)
        
        hm, wh, off = head(feat)
        
        # Heatmap loss
        Lhm = focal_loss_hm(hm, hm_gt)
        
        # Regression losses
        def gather(X):
            X = X.view(B, X.shape[1], -1)
            return _gather(X, ind)
        
        wh_g = gather(wh)
        off_g = gather(off)
        
        m = mask.unsqueeze(-1)
        Lwh = F.l1_loss(wh_g * m, wh_gt * m, reduction="sum") / (m.sum() + 1e-4)
        Loff = F.l1_loss(off_g * m, reg_gt * m, reduction="sum") / (m.sum() + 1e-4)
        
        loss = hm_w * Lhm + wh_w * Lwh + off_w * Loff
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        
        avg["hm"] += Lhm.item()
        avg["wh"] += Lwh.item()
        avg["off"] += Loff.item()
        steps += 1
        processed_samples += imgs.shape[0]
        
        # Log progress every 10 batches
        if batch_idx % 10 == 0:
            progress = (processed_samples / total_samples) * 100
            print(f"Batch {batch_idx:3d}/{len(loader)} | "
                  f"Progress: {progress:5.1f}% ({processed_samples}/{total_samples}) | "
                  f"Loss: {loss.item():.4f} (hm: {Lhm.item():.3f}, wh: {Lwh.item():.3f}, off: {Loff.item():.3f})")
    
    for k in avg:
        avg[k] /= max(1, steps)
    
    return avg


@torch.no_grad()
def evaluate(head, fp_fn, loader, device, thr=0.3, K=100):
    """
    Evaluate model on validation set.
    
    Args:
        head: Center detection head
        fp_fn: Feature provider function
        loader: Data loader
        device: Device
        thr: Confidence threshold
        K: Top-K detections
        
    Returns:
        float: AP@0.5 score
    """
    head.eval()
    preds = []
    gts = []
    
    total_batches = len(loader)
    print(f"Evaluating on {total_batches} batches...")
    
    for batch_idx, (imgs, boxes_list) in enumerate(loader):
        imgs = imgs.to(device)
        feat, stride = fp_fn(imgs)
        
        hm, wh, off = head(feat)
        out = decode(hm, wh, off, K=K, stride=stride, thr=thr)
        
        preds.extend(out)
        gts.extend([b.to(device) for b in boxes_list])
        
        # Log evaluation progress
        if batch_idx % 50 == 0:
            progress = ((batch_idx + 1) / total_batches) * 100
            print(f"Eval progress: {progress:5.1f}% ({batch_idx+1}/{total_batches})")
    
    return ap50(preds, gts)

