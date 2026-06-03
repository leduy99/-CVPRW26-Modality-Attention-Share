"""
Classification head for testing feature quality.
Simple design to focus on feature signal quality.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Classification head that works with both Center and DETR detection heads."""
    
    def __init__(self, input_dim, num_classes=20):
        """
        Initialize classification head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of VOC classes (20)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # Convert to bfloat16 to match VLM features
        self.to(torch.bfloat16)
        
    def forward(self, feat):
        """
        Forward pass.
        
        Args:
            feat: (B, C, H, W) feature map
            
        Returns:
            dict: {'logits': (B, num_classes)}
        """
        # Keep everything in bfloat16 for consistency
        logits = self.classifier(feat)
        
        return {"logits": logits}


def train_classification_epoch(head, fp_fn, loader, device, lr=1e-3):
    """Train classification head for one epoch."""
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    steps = 0
    
    print(f"Starting classification training epoch...")
    print(f"Total samples: {len(loader.dataset)}, Batches: {len(loader)}")
    
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 2:
            imgs, boxes_list = batch
            labels_list = [torch.zeros((0,), dtype=torch.long, device=device) for _ in boxes_list]
        else:
            imgs, boxes_list, labels_list = batch
        imgs = imgs.to(device)
        # labels_list not used in this function
        
        # Create labels from boxes (simplified: just detect if objects exist)
        # For VOC, we have 20 classes, but we'll use binary classification first
        # (object vs no-object) to test feature quality
        labels = []
        for boxes in boxes_list:
            if boxes.shape[0] > 0:
                labels.append(1)  # Has objects
            else:
                labels.append(0)  # No objects
        
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        
        with torch.no_grad():
            feat, stride = fp_fn(imgs)
        
        # Forward pass
        outputs = head(feat)
        logits = outputs['logits']
        
        # For now, use binary classification (object vs no-object)
        # This tests basic feature quality
        binary_logits = logits[:, :2] if logits.shape[1] > 2 else logits
        
        # Compute loss
        loss = F.cross_entropy(binary_logits, labels)
        
        # Backward pass
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 0.1)
        opt.step()
        
        # Statistics
        total_loss += loss.item()
        pred = binary_logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total_samples += labels.shape[0]
        steps += 1
        
        # Log progress
        if batch_idx % 10 == 0:
            progress = (batch_idx / len(loader)) * 100
            acc = (total_correct / total_samples) * 100
            print(f"Batch {batch_idx:3d}/{len(loader)} | "
                  f"Progress: {progress:5.1f}% | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {acc:.2f}%")
    
    avg_loss = total_loss / max(1, steps)
    avg_acc = (total_correct / max(1, total_samples)) * 100
    
    return {'loss': avg_loss, 'accuracy': avg_acc}


@torch.no_grad()
def evaluate_classification(head, fp_fn, loader, device):
    """Evaluate classification head."""
    head.eval()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    total_batches = len(loader)
    print(f"Evaluating classification on {total_batches} batches...")
    
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 2:
            imgs, boxes_list = batch
            labels_list = [torch.zeros((0,), dtype=torch.long, device=device) for _ in boxes_list]
        else:
            imgs, boxes_list, labels_list = batch
        imgs = imgs.to(device)
        # labels_list not used in this function
        
        # Create labels (binary: object vs no-object)
        labels = []
        for boxes in boxes_list:
            if boxes.shape[0] > 0:
                labels.append(1)  # Has objects
            else:
                labels.append(0)  # No objects
        
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        
        feat, stride = fp_fn(imgs)
        
        outputs = head(feat)
        logits = outputs['logits']
        
        # Binary classification
        binary_logits = logits[:, :2] if logits.shape[1] > 2 else logits
        loss = F.cross_entropy(binary_logits, labels)
        
        pred = binary_logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total_samples += labels.shape[0]
        total_loss += loss.item()
        
        # Log progress
        if batch_idx % 50 == 0:
            progress = ((batch_idx + 1) / total_batches) * 100
            print(f"Classification eval progress: {progress:5.1f}% ({batch_idx+1}/{total_batches})")
    
    avg_loss = total_loss / max(1, total_batches)
    accuracy = (total_correct / max(1, total_samples)) * 100
    
    return accuracy


def train_center_and_classification_epoch(center_head, cls_head, fp_fn, loader, device, lr=2e-3):
    """Train Center head and Classification head simultaneously."""
    center_head.train()
    cls_head.train()
    
    # Optimizers for both heads
    center_opt = torch.optim.AdamW(center_head.parameters(), lr=lr, weight_decay=1e-4)
    cls_opt = torch.optim.AdamW(cls_head.parameters(), lr=lr, weight_decay=1e-4)
    
    total_center_loss = 0
    total_cls_loss = 0
    total_correct = 0
    total_samples = 0
    total_objects = 0
    steps = 0
    
    print(f"Starting simultaneous Center + Classification training epoch...")
    print(f"Total samples: {len(loader.dataset)}, Batches: {len(loader)}")
    
    for batch_idx, batch in enumerate(loader):
        # Backward compat with collate
        if len(batch) == 2:
            imgs, boxes_list = batch
            labels_list = [torch.zeros((0,), dtype=torch.long, device=device) for _ in boxes_list]
        else:
            imgs, boxes_list, labels_list = batch
        imgs = imgs.to(device)
        # labels_list not used in this function
        boxes_list = [b.to(device) for b in boxes_list]
        
        with torch.no_grad():
            feat, stride = fp_fn(imgs)
        
        # Center head forward
        hm, wh, off = center_head(feat)
        
        # Classification head forward
        cls_outputs = cls_head(feat)
        cls_logits = cls_outputs['logits']
        
        # Center head loss with proper GT targets (match center_head.train_epoch)
        from .center_head import focal_loss_hm, build_targets, _gather
        B, _, Hf, Wf = feat.shape
        hm_gt, ind, reg_gt, wh_gt, mask = build_targets(boxes_list, Hf, Wf, stride)
        hm_gt = hm_gt.to(device)
        ind = ind.to(device)
        reg_gt = reg_gt.to(device)
        wh_gt = wh_gt.to(device)
        mask = mask.to(device)

        # Heatmap focal loss
        hm_loss = focal_loss_hm(hm, hm_gt)

        # Gather WH/OFF at GT indices
        def gather_at(X):
            X = X.view(B, X.shape[1], -1)
            return _gather(X, ind)

        wh_g = gather_at(wh)
        off_g = gather_at(off)

        m = mask.unsqueeze(-1)
        # Masked L1 losses (normalized by number of valid points)
        wh_loss = F.l1_loss(wh_g * m, wh_gt * m, reduction="sum") / (m.sum() + 1e-4)
        off_loss = F.l1_loss(off_g * m, reg_gt * m, reduction="sum") / (m.sum() + 1e-4)

        center_total_loss = hm_loss + 0.1 * wh_loss + off_loss
        
        # Classification loss (multi-label from ground truth classes present in image)
        num_classes = cls_logits.shape[1]
        targets = torch.zeros((imgs.shape[0], num_classes), device=device, dtype=torch.float32)
        for i, lbls in enumerate(labels_list):
            for l in lbls.tolist():
                if 0 <= l < num_classes:
                    targets[i, int(l)] = 1.0
        # BCE with logits for multi-label
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits.float(), targets)
        
        # Backward pass for both heads
        center_opt.zero_grad(set_to_none=True)
        cls_opt.zero_grad(set_to_none=True)
        
        center_total_loss.backward(retain_graph=True)
        cls_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(center_head.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(cls_head.parameters(), 0.1)
        
        center_opt.step()
        cls_opt.step()
        
        # Statistics
        total_center_loss += center_total_loss.item()
        total_cls_loss += cls_loss.item()
        total_objects += sum(len(boxes) for boxes in boxes_list)
        
        # Compute micro accuracy over all classes
        pred_bin = (cls_logits.sigmoid() > 0.5).to(targets.dtype)
        total_correct += (pred_bin == targets).sum().item()
        total_samples += targets.numel()
        steps += 1
        
        # Log progress
        if batch_idx % 10 == 0:
            progress = (batch_idx / len(loader)) * 100
            acc = (total_correct / total_samples) * 100
            print(f"Batch {batch_idx:3d}/{len(loader)} | "
                  f"Progress: {progress:5.1f}% | "
                  f"Center: {center_total_loss.item():.4f} | "
                  f"CLS: {cls_loss.item():.4f} | "
                  f"Acc: {acc:.2f}%")
    
    avg_center_loss = total_center_loss / max(1, steps)
    avg_cls_loss = total_cls_loss / max(1, steps)
    avg_acc = (total_correct / max(1, total_samples)) * 100
    
    # Calculate average objects per batch
    avg_objects = total_objects / max(1, steps)
    
    return {'loss': avg_center_loss, 'objects': avg_objects}, {'loss': avg_cls_loss, 'accuracy': avg_acc}


def train_detr_and_classification_epoch(detr_head, cls_head, fp_fn, loader, device, lr=1e-4):
    """Train DETR head and Classification head simultaneously."""
    detr_head.train()
    cls_head.train()
    
    # Optimizers for both heads
    detr_opt = torch.optim.AdamW(detr_head.parameters(), lr=lr, weight_decay=1e-4)
    cls_opt = torch.optim.AdamW(cls_head.parameters(), lr=lr, weight_decay=1e-4)
    
    total_detr_loss = 0
    total_cls_loss = 0
    total_correct = 0
    total_samples = 0
    total_objects = 0
    steps = 0
    
    print(f"Starting simultaneous DETR + Classification training epoch...")
    print(f"Total samples: {len(loader.dataset)}, Batches: {len(loader)}")
    
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
        
        # DETR head forward
        detr_outputs = detr_head(feat)
        pred_logits = detr_outputs['pred_logits']
        pred_boxes = detr_outputs['pred_boxes']
        
        # Classification head forward
        cls_outputs = cls_head(feat)
        cls_logits = cls_outputs['logits']
        
        # DETR loss
        from .detr_head import detr_loss
        detr_losses = detr_loss({'pred_logits': pred_logits, 'pred_boxes': pred_boxes}, 
                               boxes_list, labels_list, num_classes=20, img_size_hw=None)
        detr_total_loss = detr_losses['total']
        
        # Classification loss
        labels = []
        for boxes in boxes_list:
            if boxes.shape[0] > 0:
                labels.append(1)  # Has objects
            else:
                labels.append(0)  # No objects
        
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        binary_logits = cls_logits[:, :2] if cls_logits.shape[1] > 2 else cls_logits
        cls_loss = F.cross_entropy(binary_logits, labels)
        
        # Backward pass for both heads
        detr_opt.zero_grad(set_to_none=True)
        cls_opt.zero_grad(set_to_none=True)
        
        detr_total_loss.backward(retain_graph=True)
        cls_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(detr_head.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(cls_head.parameters(), 0.1)
        
        detr_opt.step()
        cls_opt.step()
        
        # Statistics
        total_detr_loss += detr_total_loss.item()
        total_cls_loss += cls_loss.item()
        total_objects += detr_losses['num_objects']
        
        pred = binary_logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total_samples += labels.shape[0]
        steps += 1
        
        # Log progress
        if batch_idx % 10 == 0:
            progress = (batch_idx / len(loader)) * 100
            acc = (total_correct / total_samples) * 100
            print(f"Batch {batch_idx:3d}/{len(loader)} | "
                  f"Progress: {progress:5.1f}% | "
                  f"DETR: {detr_total_loss.item():.4f} | "
                  f"CLS: {cls_loss.item():.4f} | "
                  f"Acc: {acc:.2f}%")
    
    avg_detr_loss = total_detr_loss / max(1, steps)
    avg_cls_loss = total_cls_loss / max(1, steps)
    avg_objects = total_objects / max(1, steps)
    avg_acc = (total_correct / max(1, total_samples)) * 100
    
    return {'loss': avg_detr_loss, 'num_objects': avg_objects}, {'loss': avg_cls_loss, 'accuracy': avg_acc}


@torch.no_grad()
def evaluate_center_and_classification(center_head, cls_head, fp_fn, loader, device):
    """Evaluate Center head and Classification head simultaneously."""
    center_head.eval()
    cls_head.eval()
    
    # Center head evaluation
    preds = []
    gts = []
    
    # Classification evaluation
    total_correct = 0
    total_samples = 0
    
    total_batches = len(loader)
    print(f"Evaluating Center + Classification on {total_batches} batches...")
    
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 2:
            imgs, boxes_list = batch
            labels_list = [torch.zeros((0,), dtype=torch.long, device=device) for _ in boxes_list]
        else:
            imgs, boxes_list, labels_list = batch
        imgs = imgs.to(device)
        # labels_list not used in this function
        feat, stride = fp_fn(imgs)
        
        # Center head evaluation
        hm, wh, off = center_head(feat)
        from .center_head import decode
        # Use a lower threshold and larger K to better recall early training
        out = decode(hm, wh, off, K=300, stride=stride, thr=0.05)
        
        preds.extend([(pb, ps) for (pb, ps) in out])
        # boxes_list are normalized [0,1] already
        gts.extend([b.to(device) for b in boxes_list])
        
        # Classification evaluation
        cls_outputs = cls_head(feat)
        cls_logits = cls_outputs['logits']
        
        # Multi-label evaluation from ground truth
        num_classes = cls_logits.shape[1]
        targets = torch.zeros((imgs.shape[0], num_classes), device=device, dtype=torch.float32)
        for i, lbls in enumerate(labels_list):
            for l in lbls.tolist():
                if 0 <= l < num_classes:
                    targets[i, int(l)] = 1.0
        pred_bin = (cls_logits.sigmoid() > 0.5).to(targets.dtype)
        total_correct += (pred_bin == targets).sum().item()
        total_samples += targets.numel()
        
        # Log progress
        if batch_idx % 50 == 0:
            progress = ((batch_idx + 1) / total_batches) * 100
            print(f"Center+CLS eval progress: {progress:5.1f}% ({batch_idx+1}/{total_batches})")
    
    # Compute AP@0.5 for Center head
    from .center_head import ap50
    ap50_score = ap50(preds, gts)
    
    # Compute accuracy for Classification head
    accuracy = (total_correct / max(1, total_samples)) * 100
    
    return ap50_score, accuracy


@torch.no_grad()
def evaluate_detr_and_classification(detr_head, cls_head, fp_fn, loader, device):
    """Evaluate DETR head and Classification head simultaneously."""
    detr_head.eval()
    cls_head.eval()
    
    # DETR evaluation
    preds = []
    gts = []
    
    # Classification evaluation
    total_correct = 0
    total_samples = 0
    
    total_batches = len(loader)
    print(f"Evaluating DETR + Classification on {total_batches} batches...")
    
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 2:
            imgs, boxes_list = batch
            labels_list = [torch.zeros((0,), dtype=torch.long, device=device) for _ in boxes_list]
        else:
            imgs, boxes_list, labels_list = batch
        imgs = imgs.to(device)
        # labels_list not used in this function
        feat, stride = fp_fn(imgs)
        
        # DETR evaluation
        outputs = detr_head(feat)
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        # Convert to detection format
        pred_probs = pred_logits.softmax(-1)
        pred_scores = pred_probs[:, :, 1]  # Object scores (class 1 = object, class 0 = no-object)
        
        # Convert boxes to [x1, y1, x2, y2]
        from .detr_head import box_cxcywh_to_xyxy
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        
        # Filter by confidence
        for b in range(pred_logits.shape[0]):
            mask = pred_scores[b] > 0.5
            if mask.any():
                boxes = pred_boxes_xyxy[b][mask].to(device)
                scores = pred_scores[b][mask].to(device)
                preds.append((boxes, scores))
            else:
                preds.append((torch.empty(0, 4, device=device), torch.empty(0, device=device)))
            
            gts.append(boxes_list[b].to(device))
        
        # Classification evaluation
        cls_outputs = cls_head(feat)
        cls_logits = cls_outputs['logits']
        
        # Create labels (binary: object vs no-object)
        labels = []
        for boxes in boxes_list:
            if boxes.shape[0] > 0:
                labels.append(1)  # Has objects
            else:
                labels.append(0)  # No objects
        
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        
        # Binary classification
        binary_logits = cls_logits[:, :2] if cls_logits.shape[1] > 2 else cls_logits
        pred = binary_logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total_samples += labels.shape[0]
        
        # Log progress
        if batch_idx % 50 == 0:
            progress = ((batch_idx + 1) / total_batches) * 100
            print(f"DETR+CLS eval progress: {progress:5.1f}% ({batch_idx+1}/{total_batches})")
    
    # Compute AP@0.5 for DETR head
    from .center_head import ap50
    ap50_score = ap50(preds, gts)
    
    # Compute accuracy for Classification head
    accuracy = (total_correct / max(1, total_samples)) * 100
    
    return ap50_score, accuracy
