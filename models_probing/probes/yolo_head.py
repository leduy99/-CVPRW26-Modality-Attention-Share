import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms
from .ema import EMA


def iou_aligned_xyxy(a, b, eps=1e-7):
    """Calculate IoU between aligned boxes a and b (same indices).
    a, b: (N,4) xyxy format
    Returns: (N,) IoU values
    """
    x1 = torch.max(a[:,0], b[:,0])
    y1 = torch.max(a[:,1], b[:,1])
    x2 = torch.min(a[:,2], b[:,2])
    y2 = torch.min(a[:,3], b[:,3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:,2]-a[:,0]).clamp(min=0) * (a[:,3]-a[:,1]).clamp(min=0)
    area_b = (b[:,2]-b[:,0]).clamp(min=0) * (b[:,3]-b[:,1]).clamp(min=0)
    return inter / (area_a + area_b - inter + eps)


class YoloHead(nn.Module):
    """
    Minimal anchor-free YOLO-style head (single scale).
    Predicts per cell: (cx, cy, w, h, obj, cls[num_classes]).
    All values are normalized to [0,1] relative to image size after decode.
    """

    def __init__(self, in_ch: int, num_classes: int = 20, hidden: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True),
        )
        # (cx,cy,w,h,obj,cls[num_classes]) = 4 + 1 + C
        self.pred = nn.Conv2d(hidden, 5 + num_classes, 1)

        # Initialize bias for better training start
        with torch.no_grad():
            if self.pred.bias is not None:
                # objectness bias - higher to get better objectness
                self.pred.bias.data[4] = -1.0
                # class bias
                self.pred.bias.data[5:] = -2.19

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast input to match parameter dtype (float32)
        x = x.float()
        x = self.shared(x)
        return self.pred(x)  # (B, 5+C, Hf, Wf)


def ciou_loss(pred_boxes, target_boxes, eps: float = 1e-7):
    # boxes in [x1,y1,x2,y2], normalized [0,1]
    px1, py1, px2, py2 = pred_boxes.unbind(-1)
    tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)
    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    tcx = (tx1 + tx2) / 2
    tcy = (ty1 + ty2) / 2

    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = pw * ph + tw * th - inter + eps
    iou = inter / union

    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw * cw + ch * ch + eps
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(tw / th) - torch.atan(pw / ph), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ciou = iou - (rho2 / c2 + v * alpha)
    return 1 - ciou


def yolo_decode(out: torch.Tensor, stride: int, img_hw: tuple[int, int]) -> torch.Tensor:
    """
    out: (B, 5+C, Hf, Wf)
    img_hw: (H_img, W_img) theo pixel của batch hiện tại
    return: boxes (B,Hf,Wf,4 in [0,1]), obj_logit, cls_logit
    """
    B, D, Hf, Wf = out.shape
    out = out.permute(0, 2, 3, 1).contiguous()
    cxcy = out[..., 0:2].sigmoid()
    wh   = out[..., 2:4].sigmoid()
    obj_logit = out[..., 4:5]
    cls_logit = out[..., 5:]

    H_img, W_img = img_hw
    gy = torch.arange(Hf, device=out.device, dtype=torch.float32).view(1, Hf, 1, 1)
    gx = torch.arange(Wf, device=out.device, dtype=torch.float32).view(1, 1, Wf, 1)

    # tọa độ tuyệt đối (chuẩn hóa về [0,1])
    cx = (gx + cxcy[..., 0:1]) * stride / float(W_img)
    cy = (gy + cxcy[..., 1:2]) * stride / float(H_img)
    w = wh[..., 0:1]
    h = wh[..., 1:2]

    x1 = (cx - w / 2).clamp(0, 1)
    y1 = (cy - h / 2).clamp(0, 1)
    x2 = (cx + w / 2).clamp(0, 1)
    y2 = (cy + h / 2).clamp(0, 1)
    boxes = torch.cat([x1, y1, x2, y2], -1)
    return boxes, obj_logit, cls_logit


def _assign_targets(boxes_list, labels_list, Hf, Wf, stride, num_classes, img_hw, device, pos_radius=1):
    B = len(boxes_list)
    H_img, W_img = img_hw
    T_boxes = torch.zeros((B, Hf, Wf, 4), dtype=torch.float32, device=device)
    T_obj   = torch.zeros((B, Hf, Wf, 1), dtype=torch.float32, device=device)
    T_cls   = torch.zeros((B, Hf, Wf, num_classes), dtype=torch.float32, device=device)

    for b, boxes in enumerate(boxes_list):
        if boxes.numel() == 0: 
            continue
        for j in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[j].tolist()  # GT đã chuẩn hóa [0,1]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            gx_f = (cx * W_img) / stride
            gy_f = (cy * H_img) / stride
            gx, gy = int(gx_f), int(gy_f)
            
            # Center-radius assignment
            for yy in range(max(0, gy-pos_radius), min(Hf, gy+pos_radius+1)):
                for xx in range(max(0, gx-pos_radius), min(Wf, gx+pos_radius+1)):
                    # Check if cell is empty or current GT is closer to center
                    if T_obj[b, yy, xx, 0] == 0:
                        # Empty cell - assign this GT
                        T_boxes[b, yy, xx] = torch.tensor([x1, y1, x2, y2], device=device)
                        T_obj[b, yy, xx] = 1.0
                        if len(labels_list) > b and labels_list[b].numel() > j:
                            cls_id = int(labels_list[b][j].item())
                            if 0 <= cls_id < num_classes:
                                T_cls[b, yy, xx, cls_id] = 1.0
                    else:
                        # Cell already has GT - check which is closer to center
                        current_dist = abs(xx + 0.5 - gx_f) + abs(yy + 0.5 - gy_f)
                        new_dist = abs(xx + 0.5 - gx_f) + abs(yy + 0.5 - gy_f)
                        if new_dist < current_dist:
                            # Current GT is closer - replace
                            T_boxes[b, yy, xx] = torch.tensor([x1, y1, x2, y2], device=device)
                            T_obj[b, yy, xx] = 1.0
                            if len(labels_list) > b and labels_list[b].numel() > j:
                                cls_id = int(labels_list[b][j].item())
                                if 0 <= cls_id < num_classes:
                                    T_cls[b, yy, xx, cls_id] = 1.0
    return T_boxes, T_obj, T_cls


def train_yolo_epoch(head: YoloHead, fp_fn, loader, device, lr=1e-4, batch_log_file=None, bottleneck=None, ema=None, feature_adapter=None):
    head.train()
    if bottleneck is not None:
        bottleneck.train()
    if feature_adapter is not None:
        feature_adapter.train()
    
    # Collect parameters to train
    params_to_train = list(head.parameters())
    if feature_adapter is not None:
        params_to_train += list(feature_adapter.parameters())
        print(f"Training head + feature_adapter with LR={lr}")
    else:
        print(f"Training head only with LR={lr} (bottleneck already applied in fp_fn)")
    
    opt = torch.optim.AdamW(params_to_train, lr=lr, weight_decay=1e-3)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(loader), eta_min=lr*0.1)
    total, steps = 0.0, 0
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 2:
            imgs, boxes_list = batch
            labels_list = [torch.zeros((0,), dtype=torch.long, device=device) for _ in boxes_list]
        else:
            imgs, boxes_list, labels_list = batch
        imgs = imgs.to(device)
        boxes_list = [b.to(device) for b in boxes_list]
        labels_list = [l.to(device) for l in labels_list]

        with torch.no_grad():
            feat, stride = fp_fn(imgs)
        
        # Note: bottleneck is already applied in fp_fn, no need to apply again
        out = head(feat)  # (B,5+C,Hf,Wf)
        B, _, Hf, Wf = out.shape
        pred_boxes, pred_obj_logit, pred_cls_logit = yolo_decode(out, stride, img_hw=imgs.shape[-2:])

        # Targets with center-radius assignment
        T_boxes, T_obj, T_cls = _assign_targets(boxes_list, labels_list, Hf, Wf, stride,
                                                head.num_classes, imgs.shape[-2:], device, pos_radius=1)


        # Losses (cast to float32 for numerical stability)
        pred_boxes_f = pred_boxes.float(); T_boxes_f = T_boxes.float()
        pred_obj_logit_f = pred_obj_logit.float();     T_obj_f = T_obj.float()
        pred_cls_logit_f = pred_cls_logit.float();     T_cls_f = T_cls.float()

        # Box loss on positive cells
        pos = T_obj_f.squeeze(-1) > 0.5
        if pos.any():
            L_box = ciou_loss(pred_boxes_f[pos], T_boxes_f[pos]).mean()
        else:
            L_box = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        # IoU-aware objectness targets
        obj_target = T_obj_f.clone()
        if pos.any():
            # Calculate IoU between predictions and GT at positive cells
            iou_pos = iou_aligned_xyxy(pred_boxes_f[pos], T_boxes_f[pos]).detach()
            obj_target[pos] = iou_pos.unsqueeze(-1)
        
        # Objectness loss with fixed pos_weight
        L_obj = F.binary_cross_entropy_with_logits(pred_obj_logit_f, obj_target, pos_weight=torch.tensor(3.0, device=device))
        
        # Classification loss ONLY on positive cells
        L_cls = (F.binary_cross_entropy_with_logits(pred_cls_logit_f[pos], T_cls_f[pos])
                 if pos.any() else pred_cls_logit_f.sum()*0)
        
        # Weighted loss
        loss = 1.0 * L_box + 1.0 * L_obj + 1.0 * L_cls

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 0.5)
        opt.step()
        scheduler.step()
        
        # Skip EMA for simplicity

        total += float(loss.item()); steps += 1
        if batch_idx % 10 == 0:
            log_msg = f"Batch {batch_idx:4d}/{len(loader)} | Loss: {loss.item():.4f} | L_box: {L_box.item():.4f} | L_obj: {L_obj.item():.4f} | L_cls: {L_cls.item():.4f}"
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
def evaluate_yolo(head: YoloHead, fp_fn, loader, device, conf=0.25, nms_thresh=0.5, ema=None):
    # Use regular model (no EMA)
    head.eval()
    preds, gts = [], []
    total_batches = len(loader)
    print(f"Evaluating YOLO on {total_batches} batches...")
    
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 2:
            imgs, boxes_list = batch
            labels_list = [None] * len(boxes_list)
        else:
            imgs, boxes_list, labels_list = batch
        imgs = imgs.to(device)
        feat, stride = fp_fn(imgs)
        out = head(feat)

        boxes, obj_logit, cls_logit = yolo_decode(out, stride, img_hw=imgs.shape[-2:])
        obj = obj_logit.sigmoid()
        cls = cls_logit.sigmoid()
        cls_score, cls_idx = cls.max(-1)  # (B,Hf,Wf)
        score = (obj.squeeze(-1) * cls_score).cpu()
        boxes = boxes.cpu()

        B, Hf, Wf, _ = boxes.shape
        
        # Debug confidence calculation
        if batch_idx < 3:
            print(f"    Debug confidence batch {batch_idx}:")
            print(f"      Obj range: {obj.min().item():.4f} - {obj.max().item():.4f}")
            print(f"      Cls sigmoid range: {cls.min().item():.4f} - {cls.max().item():.4f}")
            print(f"      Cls max range: {cls_score.min().item():.4f} - {cls_score.max().item():.4f}")
            print(f"      Score range: {score.min().item():.4f} - {score.max().item():.4f}")
            print(f"      Score > {conf}: {(score > conf).sum().item()}")
        
        batch_detections = 0
        for b in range(B):
            m = score[b] > conf
            bb = boxes[b][m]
            ss = score[b][m]
            ll = cls_idx[b][m].flatten().cpu().to(torch.int64)
            
            # Apply NMS if we have detections
            if len(bb) > 0:
                # Convert to float32 for NMS (NMS doesn't support bfloat16)
                bb_f32 = bb.float()
                ss_f32 = ss.float()
                keep = batched_nms(bb_f32, ss_f32, ll, nms_thresh)
                bb = bb[keep]
                ss = ss[keep]
                ll = ll[keep]
                batch_detections += len(bb)
            
            preds.append((bb, ss))  # chỉ boxes và scores cho ap50
            gts.append(boxes_list[b])   # GT box (chuẩn hóa [0,1])
        
        # Progress logging
        if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
            print(f"  Batch {batch_idx+1:4d}/{total_batches} | Detections: {batch_detections}")
    
    print("Computing AP@0.5...")
    # AP@0.5 using shared ap50
    from .center_head import ap50
    ap_score = ap50(preds, gts)
    print(f"Final AP@0.5: {ap_score:.4f}")
    return ap_score
