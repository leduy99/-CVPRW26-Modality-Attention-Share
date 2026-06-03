"""
DETR detection head (encoder–decoder) with Hungarian matching and losses.
- Stable FP32 parameters (use AMP at train time if desired)
- Correct query positional encoding
- Hungarian matcher with class/L1/GIoU costs
- Loss weighting & normalization following DETR spirit
- Image-size aware scaling (no 512 hardcode)
- Self-contained AP@0.5 evaluation (class-agnostic)

Author: ChatGPT
"""
from __future__ import annotations
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from scipy.optimize import linear_sum_assignment


# -----------------------------
# Positional Encoding (2D spatial)
# -----------------------------
class SpatialPositionalEncoding(nn.Module):
    """2D spatial positional encoding using sine-cosine encoding."""
    
    def __init__(self, d_model: int, temperature: int = 10000):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.temperature = temperature

    @torch.no_grad()
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        device = x.device
        dtype = torch.float32  # calculate PE in FP32 for stability, then cast to src dtype
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        yy = yy.flatten().to(dtype)
        xx = xx.flatten().to(dtype)

        # Each encoding gets d_model//4 dimensions
        quarter = self.d_model // 4
        dim_t = torch.arange(quarter, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / quarter)

        pos_x = xx[:, None] / dim_t
        pos_y = yy[:, None] / dim_t
        
        # Create sin/cos encodings - each has quarter dimensions
        pos_x_sin = pos_x.sin()  # (HW, quarter)
        pos_x_cos = pos_x.cos()  # (HW, quarter)
        pos_y_sin = pos_y.sin()  # (HW, quarter)
        pos_y_cos = pos_y.cos()  # (HW, quarter)
        
        # Interleave y and x encodings to get correct dimensions
        pos = torch.zeros(h*w, self.d_model, device=device, dtype=dtype)
        pos[:, 0::4] = pos_y_sin
        pos[:, 1::4] = pos_y_cos
        pos[:, 2::4] = pos_x_sin
        pos[:, 3::4] = pos_x_cos
        return pos.unsqueeze(0).to(x.dtype)


# -----------------------------
# DETR Head
# -----------------------------
class DETRHead(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden_dim: int = 256,
        num_queries: int = 50,  # Reduced for probe
        num_classes: int = 20,  # Multi-class
        nheads: int = 8,
        num_enc: int = 2,  # Reduced for probe
        num_dec: int = 2,  # Reduced for probe
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Conv2d(in_ch, hidden_dim, kernel_size=1)
        
        # Encoder over image tokens
        enc_layer = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, dim_feedforward=2048, dropout=0.1, batch_first=True
        )
        self.encoder = TransformerEncoder(enc_layer, num_layers=num_enc)

        # Decoder over object queries (cross-attend to memory)
        dec_layer = TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nheads, dim_feedforward=2048, dropout=0.1, batch_first=True
        )
        self.decoder = TransformerDecoder(dec_layer, num_layers=num_dec)

        # Learned queries and their positional embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # content
        self.query_pos = nn.Embedding(num_queries, hidden_dim)    # position

        # Positional encoding for image tokens
        self.spatial_pos_encoding = SpatialPositionalEncoding(hidden_dim)

        # Prediction heads (MLP for boxes is common in DETR)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.class_embed.weight)
        nn.init.constant_(self.class_embed.bias, 0.0)
        for m in self.bbox_embed:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, features: torch.Tensor) -> dict:
        """features: (B, C, H, W) from your backbone/encoder"""
        features = features.float()  # keep params FP32; use AMP outside
        B, C, H, W = features.shape
        
        # 1) project & flatten image features
        src = self.input_proj(features)            # (B, D, H, W)
        src = src.flatten(2).transpose(1, 2)       # (B, HW, D)

        # Get 2D spatial positional embeddings
        pos = self.spatial_pos_encoding(src, H, W).to(src.device).to(src.dtype)  # (1, HW, D)


        # 2) Skip encoder - use raw features + spatial PE as memory for probing
        memory = src + pos  # (B, HW, D)

        # 3) decode queries: (tgt + query_pos) cross-attend to (memory)
        query_content = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B,Q,D)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)        # (B,Q,D)
        hs = self.decoder(tgt=query_content + query_pos, memory=memory)  # (B,Q,D)

        # 4) predictions
        pred_logits = self.class_embed(hs)         # (B,Q,C+1)
        pred_boxes = self.bbox_embed(hs).sigmoid() # (B,Q,4) in [0,1] (cx,cy,w,h)

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


# -----------------------------
# Box utilities
# -----------------------------

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    xyxy = torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)
    # Clamp to [0,1] to avoid edge cases
    return torch.cat([xyxy[..., :2].clamp(0, 1), xyxy[..., 2:].clamp(0, 1)], -1)


def generalized_box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Generalized IoU between boxes1 (N,4) and boxes2 (M,4), both in [0,1]."""
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 is invalid"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 is invalid"
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-7)

    lti = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rbi = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    whi = (rbi - lti).clamp(min=0)
    areai = whi[..., 0] * whi[..., 1]
    giou = iou - (areai - union) / (areai + 1e-7)
    return giou


def iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for xyxy boxes (N,4) and (M,4) in same scale."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)


# -----------------------------
# Hungarian matcher
# -----------------------------
@torch.no_grad()
def hungarian_matcher(
    pred_logits: torch.Tensor,  # (B,Q,C+1)
    pred_boxes: torch.Tensor,   # (B,Q,4) cxcywh in [0,1]
    target_boxes: List[torch.Tensor],  # list of (N_i,4) xyxy in pixel or [0,1]
    target_labels: List[torch.Tensor], # list of (N_i,) in [0..C-1]
    num_classes: int,
    img_size_hw: Optional[Tuple[int, int]] = None,  # (H,W) if targets are in pixels
    cost_weights: Tuple[float, float, float] = (1.0, 5.0, 2.0),  # (cls, L1, GIoU) - enable classification
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    B, Q, _ = pred_logits.shape
    device = pred_logits.device
    cls_w, l1_w, giou_w = cost_weights

    pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)  # (B,Q,4) in [0,1]

    indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for b in range(B):
        tgt_boxes = target_boxes[b]
        tgt_labels = target_labels[b]
        N = int(tgt_boxes.shape[0])

        if N == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            indices.append((empty, empty))
            continue
        
        # Normalize targets to [0,1] if needed
        if img_size_hw is not None:
            H_img, W_img = img_size_hw
            tgt_xyxy_n = tgt_boxes.clone().to(device).float()
            tgt_xyxy_n[:, [0, 2]] /= float(W_img)
            tgt_xyxy_n[:, [1, 3]] /= float(H_img)
        else:
            tgt_xyxy_n = tgt_boxes.to(device).float()

        # (a) classification cost: -log P(y)
        logprob = pred_logits[b].log_softmax(-1)[:, :num_classes]  # (Q,C)
        cost_class = -logprob[:, tgt_labels.long()]                 # (Q,N)

        # (b) L1 on cxcywh in [0,1]
        tgt_cxcywh = torch.stack([
            0.5 * (tgt_xyxy_n[:, 0] + tgt_xyxy_n[:, 2]),
            0.5 * (tgt_xyxy_n[:, 1] + tgt_xyxy_n[:, 3]),
            (tgt_xyxy_n[:, 2] - tgt_xyxy_n[:, 0]),
            (tgt_xyxy_n[:, 3] - tgt_xyxy_n[:, 1]),
        ], dim=-1)
        l1_cost = torch.cdist(pred_boxes[b].float(), tgt_cxcywh.float(), p=1)

        # (c) GIoU cost
        giou = generalized_box_iou_matrix(pred_boxes_xyxy[b], tgt_xyxy_n)  # (Q,N)
        giou_cost = 1.0 - giou

        cost = cls_w * cost_class + l1_w * l1_cost + giou_w * giou_cost
        pi, ti = linear_sum_assignment(cost.detach().cpu().numpy())
        indices.append(
            (
                torch.as_tensor(pi, device=device, dtype=torch.long),
                torch.as_tensor(ti, device=device, dtype=torch.long),
            )
        )
    return indices


# -----------------------------
# DETR Loss
# -----------------------------

def detr_loss(
    outputs: dict,
    targets_boxes: List[torch.Tensor],
    targets_labels: List[torch.Tensor],
    num_classes: int,
    img_size_hw: Optional[Tuple[int, int]] = None,  # (H,W) if targets are in pixels
    eos_coef: float = 0.1,
    cls_w: float = 1.0,  # Enable classification
    bbox_w: float = 5.0,
    giou_w: float = 2.0,
) -> dict:
    pred_logits = outputs["pred_logits"]  # (B,Q,C+1)
    pred_boxes = outputs["pred_boxes"]    # (B,Q,4)
    B, Q, _ = pred_logits.shape

    indices = hungarian_matcher(
        pred_logits,
        pred_boxes,
        targets_boxes,
        targets_labels,
        num_classes=num_classes,
        img_size_hw=img_size_hw,
        cost_weights=(cls_w, bbox_w, giou_w),
    )

    device = pred_logits.device
    empty_weight = torch.ones(num_classes + 1, device=device)
    empty_weight[-1] = eos_coef  # background weight

    ce_losses: List[torch.Tensor] = []
    num_boxes: int = 0
    l1_sum = pred_boxes.new_tensor(0.0)
    giou_sum = pred_boxes.new_tensor(0.0)

    for b in range(B):
        pi, ti = indices[b]
        N = targets_boxes[b].shape[0]
        num_boxes += N

        # Classification targets for all queries (unmatched -> background)
        target_classes = torch.full((Q,), num_classes, dtype=torch.long, device=device)
        if len(pi) > 0:
            target_classes[pi] = targets_labels[b][ti].long().clamp(min=0, max=num_classes - 1)

        ce = F.cross_entropy(pred_logits[b], target_classes, weight=empty_weight, label_smoothing=0.1)
        ce_losses.append(ce)

        if len(pi) > 0:
            pb = pred_boxes[b][pi]  # (M,4) cxcywh
            tb_xyxy = targets_boxes[b][ti].to(device).float()

            if img_size_hw is not None:
                H_img, W_img = img_size_hw
                tb_xyxy = tb_xyxy.clone()
                tb_xyxy[:, [0, 2]] /= float(W_img)
                tb_xyxy[:, [1, 3]] /= float(H_img)

            tb = torch.stack([
                0.5 * (tb_xyxy[:, 0] + tb_xyxy[:, 2]),
                0.5 * (tb_xyxy[:, 1] + tb_xyxy[:, 3]),
                (tb_xyxy[:, 2] - tb_xyxy[:, 0]),
                (tb_xyxy[:, 3] - tb_xyxy[:, 1]),
            ], dim=-1)

            l1_sum = l1_sum + F.l1_loss(pb, tb, reduction="sum")
            giou_diag = generalized_box_iou_matrix(box_cxcywh_to_xyxy(pb), tb_xyxy).diag()
            giou_sum = giou_sum + (1.0 - giou_diag).sum()

    ce_loss = torch.stack(ce_losses).mean() if ce_losses else pred_logits.sum() * 0
    if num_boxes > 0:
        l1_loss = l1_sum / num_boxes
        giou_loss = giou_sum / num_boxes
    else:
        l1_loss = pred_boxes.sum() * 0
        giou_loss = pred_boxes.sum() * 0
    
    total = cls_w * ce_loss + bbox_w * l1_loss + giou_w * giou_loss
    return {
        "total": total,
        "loss_cls": ce_loss,
        "loss_l1": l1_loss,
        "loss_giou": giou_loss,
        "num_boxes": num_boxes,
    }


# -----------------------------
# Training / Evaluation
# -----------------------------

def train_detr_epoch(
    head: DETRHead,
    fp_fn,
    loader,
    device: torch.device,
    lr: float = 2e-3,  # Match YOLO's effective LR
    num_classes: int = 20,  # Multi-class
    amp: bool = True,
    batch_log_file: str = None,
) -> dict:
    """One training epoch for DETR head.

    fp_fn: imgs -> (features, stride)  # stride unused here
    loader: yields (imgs, boxes_list[, labels_list])
    """
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)  # Match YOLO's weight_decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(loader), eta_min=lr*0.1)
    use_amp = amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    total, steps = 0.0, 0
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 2:
            imgs, boxes_list = batch
            # Use actual labels if available, otherwise assign label 0 to all GT boxes
            labels_list = [torch.zeros((b.shape[0],), dtype=torch.long, device=device) for b in boxes_list]
        else:
            imgs, boxes_list, labels_list = batch

        imgs = imgs.to(device)
        boxes_list = [b.to(device) for b in boxes_list]
        labels_list = [l.to(device) for l in labels_list]
        H_img, W_img = imgs.shape[-2:]
        
        with torch.no_grad():
            feat, _ = fp_fn(imgs)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = head(feat)
            losses = detr_loss(
                outputs,
                boxes_list,
                labels_list,
                num_classes=num_classes,
                img_size_hw=(H_img, W_img),  # Normalize GT to [0,1] in loss
            )

        opt.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)  # Match YOLO's grad clipping
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        
        total += float(losses["total"].item())
        steps += 1
        
        
        if batch_idx % 10 == 0:
            log_msg = f"Batch {batch_idx:4d}/{len(loader)} | Loss: {losses['total'].item():.4f} | L_cls: {losses['loss_cls'].item():.4f} | L_l1: {losses['loss_l1'].item():.4f} | L_giou: {losses['loss_giou'].item():.4f}"
            print(log_msg)
            # Stream to file if provided (batch writes for performance)
            if batch_log_file:
                with open(batch_log_file, 'a') as f:
                    f.write(log_msg + "\n")
                    # Only flush every 50 batches to reduce I/O overhead
                    if batch_idx % 50 == 0:
                        f.flush()

    return {"loss": total / max(1, steps)}


@torch.no_grad()
def evaluate_detr(
    head: DETRHead,
    fp_fn,
    loader,
    device: torch.device,
    topk: int = 100,
    conf_threshold: float = 0.0,
) -> float:
    head.eval()
    preds: List[Tuple[torch.Tensor, torch.Tensor]] = []  # (boxes_xyxy, scores)
    gts: List[torch.Tensor] = []
    
    total_batches = len(loader)
    print(f"Evaluating DETR on {total_batches} batches...")
    
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 2:
            imgs, boxes_list = batch
        else:
            imgs, boxes_list, _ = batch
        imgs = imgs.to(device)
        H_img, W_img = imgs.shape[-2:]
        
        feat, _ = fp_fn(imgs)
        outputs = head(feat)
        pred_logits = outputs["pred_logits"]  # (B,Q,C+1)
        pred_boxes = outputs["pred_boxes"]    # (B,Q,4)

        probs = pred_logits.softmax(-1)[:, :, :-1]   # drop background
        scores, _ = probs.max(-1)                    # (B,Q)
        boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)  # [0,1]
        # Scale to pixels to match ap50() expectations
        scale = torch.tensor([W_img, H_img, W_img, H_img], device=boxes_xyxy.device).view(1,1,4)
        boxes_xyxy = boxes_xyxy * scale


        B, Q = scores.shape
        K = min(topk, Q)
        batch_detections = 0
        for b in range(B):
            if conf_threshold > 0.0:
                keep = torch.nonzero(scores[b] > conf_threshold, as_tuple=False).squeeze(1)
                if keep.numel() == 0:
                    keep = torch.topk(scores[b], k=K).indices
            else:
                keep = torch.topk(scores[b], k=K).indices

            bb = boxes_xyxy[b][keep]
            ss = scores[b][keep]
            
            # Apply NMS to remove overlapping detections
            if len(bb) > 0:
                from torchvision.ops import nms
                keep2 = nms(bb, ss, 0.5)
                bb, ss = bb[keep2], ss[keep2]
            
            batch_detections += len(bb)
            preds.append((bb.cpu(), ss.cpu()))
            gts.append(boxes_list[b].cpu())

        if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
            print(f"  Batch {batch_idx + 1:4d}/{total_batches} | Dets: {batch_detections}")

    print("Computing AP@0.5 (class-agnostic)...")
    ap_score = ap50(preds, gts)
    print(f"Final AP@0.5: {ap_score:.4f}")
    return ap_score


# -----------------------------
# Simple AP@0.5 (class-agnostic) implementation
# -----------------------------

def ap50(
    preds: List[Tuple[torch.Tensor, torch.Tensor]],  # [(N_i,4 xyxy), (N_i,)] per image
    gts: List[torch.Tensor],                         # [(M_i,4 xyxy)] per image
) -> float:
    """Compute class-agnostic AP@0.5 over the dataset.
    - preds: list over images of (boxes, scores) in pixel coordinates
    - gts: list over images of GT boxes in pixel coordinates
    """
    assert len(preds) == len(gts)
    all_scores: List[float] = []
    all_is_tp: List[int] = []
    total_positives = 0

    for (pb, ps), gb in zip(preds, gts):
        pb = pb.float()
        gb = gb.float()
        total_positives += gb.size(0)
        if pb.numel() == 0:
            continue
        # Sort predictions in this image by score desc
        order = torch.argsort(ps, descending=True)
        pb = pb[order]
        ps = ps[order]
        matched = torch.zeros(gb.size(0), dtype=torch.bool)
        ious = iou_matrix(pb, gb)  # (P, G)
        for i in range(pb.size(0)):
            all_scores.append(float(ps[i].item()))
            if gb.size(0) == 0:
                all_is_tp.append(0)
                continue
            iou_row = ious[i]
            j = int(torch.argmax(iou_row).item())
            if iou_row[j] >= 0.5 and not matched[j]:
                matched[j] = True
                all_is_tp.append(1)
            else:
                all_is_tp.append(0)

    if total_positives == 0:
        return 0.0

    if len(all_scores) == 0:
        return 0.0

    # Global sort by score desc
    import numpy as np

    scores_np = np.array(all_scores)
    is_tp_np = np.array(all_is_tp, dtype=np.int32)
    order = np.argsort(-scores_np)
    is_tp_np = is_tp_np[order]

    tp_cum = np.cumsum(is_tp_np)
    fp_cum = np.cumsum(1 - is_tp_np)
    recall = tp_cum / max(1, total_positives)
    precision = tp_cum / np.maximum(1, tp_cum + fp_cum)

    return float(voc_ap(recall, precision))


def voc_ap(recall: "np.ndarray", precision: "np.ndarray") -> float:
    """11-point interpolated AP or continuous AP.
    Here we implement continuous AP by integrating the precision envelope.
    """
    import numpy as np

    # Append boundary points
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Sum over recall steps
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)