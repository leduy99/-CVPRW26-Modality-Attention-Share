"""
Training script to compare encoder vs projector features on Pascal VOC.
Trains center detection heads and evaluates AP@0.5 performance.
"""
import argparse
import torch
import json
import os
from datetime import datetime

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets.voc_custom import CustomVOCCenterDataset, collate_fn
from torch.utils.data import DataLoader
from probes.center_head import CenterHead, train_epoch, evaluate
from probes.detr_head import DETRHead, train_detr_epoch, evaluate_detr
from probes.classification_head import (
    ClassificationHead, train_classification_epoch, evaluate_classification,
    train_center_and_classification_epoch, train_detr_and_classification_epoch,
    evaluate_center_and_classification, evaluate_detr_and_classification
)
from torchvision.transforms.functional import to_pil_image


def build_feature_provider(adapter, tap="encoder"):
    """
    Build feature provider function for given adapter and tap point.
    
    Args:
        adapter: VLM adapter instance
        tap: "encoder" or "projector"
        
    Returns:
        function: Feature provider function
    """
    dev = adapter.model.device
    
    # cache 1x1 proj khi kênh quá lớn (per-tap)
    proj = {"layer": None, "in_ch": None, "out_ch": 1024}

    @torch.no_grad()
    def _fp(imgs):
        """
        Feature provider function - FIXED: process each image in batch separately.
        
        Args:
            imgs: Batch of images (B, 3, H, W)
            
        Returns:
            tuple: (features, stride)
        """
        B = imgs.shape[0]
        feats = []
        strides = []
        Hf0 = Wf0 = None

        for b in range(B):
            # to_pil_image kỳ vọng [0,1] → clamp tránh nan do normalize
            pil = to_pil_image(imgs[b].detach().cpu().clamp(0, 1))

            if tap == "encoder":
                G = adapter.encode_grid(pil)  # {"grid": (1,Hf,Wf,D), "meta": {...}}
                grid = G["grid"]
                Hf, Wf = G["meta"]["Hf"], G["meta"]["Wf"]

            elif tap == "projector":
                Gt = adapter.project_tokens(pil)  # tokens: (1,N,D), meta có Hf/Wf đúng
                tokens = Gt["tokens"]; B1, N, D = tokens.shape
                Hf, Wf = Gt["meta"]["Hf"], Gt["meta"]["Wf"]
                assert N == Hf * Wf, f"Token count {N} != grid size {Hf}x{Wf}"
                grid = tokens.view(B1, Hf, Wf, D).contiguous()

            elif tap == "llm":
                Gt = adapter.extract_llm_features(pil)  # image-tokens (1,N,D)
                tokens = Gt["tokens"]; B1, N, D = tokens.shape
                # Use Hf, Wf from LLM metadata (not encoder!)
                Hf, Wf = Gt["meta"]["Hf"], Gt["meta"]["Wf"]
                # Verify token count matches grid size
                expected = Hf * Wf
                if N != expected:
                    # Pad or truncate if mismatch
                    if N < expected:
                        pad = torch.zeros(B1, expected - N, D, device=tokens.device, dtype=tokens.dtype)
                        tokens = torch.cat([tokens, pad], dim=1)
                    else:
                        tokens = tokens[:, :expected, :]
                grid = tokens.view(B1, Hf, Wf, D).contiguous()
            else:
                raise ValueError("tap must be 'encoder', 'projector', or 'llm'")

            F = adapter.to_feature_map(grid)  # {"feat": (1,C,Hf,Wf), "stride": float}
            feat = F["feat"].to(dev)          # (1,C,Hf,Wf)

            C = feat.shape[1]
            if C >= 8192:
                import torch.nn as nn
                if (proj["layer"] is None) or (proj["in_ch"] != C):
                    proj["layer"] = nn.Conv2d(C, proj["out_ch"], 1).to(dev, dtype=feat.dtype)
                    proj["in_ch"] = C
                feat = proj["layer"](feat)  # (1,1024,Hf,Wf)

            # ép mọi ảnh về cùng Hf,Wf trong batch (giữ Hf0,Wf0 của ảnh đầu)
            if Hf0 is None:
                Hf0, Wf0 = feat.shape[-2:]
            elif (feat.shape[-2:] != (Hf0, Wf0)):
                feat = torch.nn.functional.interpolate(feat, size=(Hf0, Wf0),
                                                       mode="bilinear", align_corners=False)

            feats.append(feat)
            strides.append(F["stride"])

        feat = torch.cat(feats, dim=0).contiguous()         # (B,C,Hf0,Wf0)
        # stride: kỳ vọng giống nhau → lấy mean và assert độ lệch nhỏ
        s = float(sum(strides)) / len(strides)
        if max(abs(si - s) for si in strides) > 1e-3:
            print(f"[WARN] varied stride in batch: {strides}")
        return feat, s

    return _fp


def main():
    """Main training and comparison function."""
    parser = argparse.ArgumentParser(description="Compare VLM feature maps")
    parser.add_argument("--model", type=str, default="qwen25",
                       choices=["qwen25", "qwen3", "ovis25", "internvl25",  # Default: 2-4B models
                               "llava", "llava15", "internvl3", "internvl35"],  # Optional: larger models
                       help="Model to use (default models are 2-4B for fair comparison)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--bs", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3,
                       help="Learning rate")
    parser.add_argument("--hidden", type=int, default=128,
                       help="Hidden layer size in detection head")
    parser.add_argument("--head", type=str, default="center",
                       choices=["center", "detr", "classification", "yolo"],
                       help="Detection head type")
    parser.add_argument("--taps", type=str, default="all",
                       help="Feature taps to train: 'all' (encoder,projector,llm), 'encoder', 'projector', 'llm', or comma-separated list like 'projector,llm'")
    parser.add_argument("--param", type=str, default="auto",
                       help="Model parameter size: 'auto' (use default), '2b', '3b', '7b', '8b', etc. For models with multiple size variants.")
    parser.add_argument("--adapt_features", action="store_true",
                       help="Use learnable MLP adapter to refine raw encoder/LLM features (makes comparison fairer with projector)")
    
    args = parser.parse_args()
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{args.model}_{args.head}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Parse taps argument
    if args.taps == "all":
        taps_to_train = ["encoder", "projector", "llm"]
    else:
        taps_to_train = [t.strip() for t in args.taps.split(",")]
        # Validate tap names
        valid_taps = ["encoder", "projector", "llm"]
        for tap in taps_to_train:
            if tap not in valid_taps:
                raise ValueError(f"Invalid tap '{tap}'. Must be one of: {valid_taps}")
    
    print(f"Starting training with {args.model} for {args.epochs} epochs")
    print(f"Detection head: {args.head.upper()}")
    print(f"Batch size: {args.bs}, Learning rate: {args.lr}")
    print(f"Feature taps: {', '.join(taps_to_train)}")
    print(f"Log directory: {log_dir}")
    
    # Initialize results logging
    results_log = {
        "model": args.model,
        "head": args.head,
        "epochs": args.epochs,
        "batch_size": args.bs,
        "learning_rate": args.lr,
        "timestamp": timestamp,
        "features": {}
    }
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Determine model size
    param_size = args.param if args.param != "auto" else None
    
    # Initialize adapter
    if args.model == "qwen25":
        from adapters.qwen25vl import Qwen25VLAdapter as Adapter
    elif args.model == "qwen3":
        from adapters.qwen3vl import Qwen3VLAdapter
        # Use Qwen3-VL-4B-Instruct by default (4B regular, not MoE)
        print(f"Using Qwen3-VL-4B-Instruct (4B regular)")
        Adapter = Qwen3VLAdapter  # Default uses 4B
    elif args.model == "llava":
        from adapters.llava import LlavaAdapter as Adapter
    elif args.model == "llava15":
        from adapters.llava15 import Llava15Adapter as Adapter
    elif args.model == "internvl25":
        from adapters.internvl25 import InternVL25Adapter as Adapter
    elif args.model == "internvl3":
        from adapters.internvl3 import InternVL3Adapter as Adapter
    elif args.model == "internvl35":
        from adapters.internvl35 import InternVL35Adapter as Adapter
    else:
        from adapters.ovis25 import Ovis25Adapter as Adapter
    
    print(f"Loading {args.model} adapter...")
    adapter = Adapter()
    
    # FREEZE VLM MODEL - only train detection head!
    print("Freezing VLM model parameters...")
    for param in adapter.model.parameters():
        param.requires_grad = False
    print("VLM model frozen!")
    
    # Load datasets
    print("Loading Pascal VOC dataset...")
    train_ds = CustomVOCCenterDataset(root="./vocdata", year="2007", image_set="trainval")
    val_ds = CustomVOCCenterDataset(root="./vocdata", year="2007", image_set="test")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        shuffle=True,
        num_workers=0,  # Avoid fork issues with tokenizers
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=0,  # Avoid fork issues with tokenizers
        collate_fn=collate_fn
    )
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # Compare features based on selected taps
    results = {}
    final_ap = 0
    
    for tap in taps_to_train:
        print(f"\n{'='*50}")
        print(f"{args.model.upper()} | TAP = {tap}")
        print(f"{'='*50}")
        
        # Build feature provider
        fp_fn = build_feature_provider(adapter, tap=tap)
        
        # Get input channel count by peeking at first batch
        first_batch = next(iter(train_loader))
        if len(first_batch) == 2:
            imgs, _ = first_batch
        else:
            imgs, _, _ = first_batch
        feat0, stride0 = fp_fn(imgs.to(adapter.model.device))
        in_ch = feat0.shape[1]
        
        # Optional: Add learnable feature adapter for encoder/LLM
        # (Projector already has built-in MLP, so skip it)
        feature_adapter = None
        if args.adapt_features and tap in ["encoder", "llm"]:
            from probes.feature_adapter import FeatureAdapter
            print(f"\n🔧 Adding learnable FeatureAdapter for {tap} ({in_ch} dims)")
            print(f"   This refines raw {tap} features with RMSNorm + MLP (like projector does internally)")
            feature_adapter = FeatureAdapter(in_ch, out_dim=in_ch).to(adapter.model.device, dtype=feat0.dtype)
            
            # Wrap fp_fn to include feature adapter
            original_fp = fp_fn
            def fp_with_adapter(imgs):
                feat, stride = original_fp(imgs)
                feat = feature_adapter(feat)
                return feat, stride
            fp_fn = fp_with_adapter
            print(f"   ✅ Feature adapter enabled!")
        elif args.adapt_features and tap == "projector":
            print(f"\n⚠️  Skipping feature adapter for projector (already has built-in MLP + normalization)")
        
        print(f"Feature map shape: {feat0.shape}, stride: {stride0:.3f}")
        # Get feature map info from the feature provider
        feat_info, _ = fp_fn(imgs.to(adapter.model.device))
        print(f"Feature map dimensions: Hf={feat_info.shape[2]}, Wf={feat_info.shape[3]}, C={feat_info.shape[1]}")
        
        # Initialize progress file for this feature
        progress_file = os.path.join(log_dir, f"{tap}_progress.txt")
        with open(progress_file, 'w') as f:
            f.write(f"Training Progress for {tap.upper()} features with {args.head.upper()} head\n")
            f.write(f"Feature map shape: {feat0.shape}, stride: {stride0}\n")
            f.write("="*80 + "\n")
        
        # Initialize feature-specific results
        feature_log = {
            "feature_map_shape": list(feat0.shape),
            "stride": stride0,
            "epochs": []
        }
        
        # Initialize detection head based on user choice
        if args.head == "detr":
            # DETR head for detection with bottleneck for large channels
            from probes.bottleneck import SimpleBottleneck
            
            # Always use bottleneck to ensure consistent channel dimensions
            # Use DETR default hidden_dim=256 for better performance
            detr_hidden_dim = 256
            if in_ch != detr_hidden_dim:
                # Use bottleneck to match DETR hidden_dim expectations
                # Initialize with same dtype as VLM features
                feat_dtype = feat0.dtype
                bottleneck = SimpleBottleneck(in_ch, out_ch=detr_hidden_dim).to(adapter.model.device, dtype=feat_dtype)
                head = DETRHead(detr_hidden_dim, hidden_dim=detr_hidden_dim, num_queries=100).to(adapter.model.device)
                # Store original fp_fn before reassigning
                original_fp_fn = fp_fn
                # Wrap fp_fn to include bottleneck
                def fp_bottleneck(imgs):
                    feat, stride = original_fp_fn(imgs)
                    return bottleneck(feat), stride
                fp_fn = fp_bottleneck
                print(f"Added bottleneck: {in_ch} -> {detr_hidden_dim} channels")
            else:
                head = DETRHead(in_ch, hidden_dim=detr_hidden_dim, num_queries=100).to(adapter.model.device)
                print(f"Direct DETR: {in_ch} channels (no bottleneck needed)")
            
            # Classification head for additional signal testing (use original in_ch)
            cls_head = ClassificationHead(in_ch, num_classes=20).to(adapter.model.device)
            
            # Create single batch log file for this tap
            batch_log_file = os.path.join(log_dir, f"{tap}_batches.txt")
            with open(batch_log_file, 'w') as f:
                f.write(f"DETR Training - {tap.upper()} Features\n")
                f.write("="*60 + "\n")
            
            # Training loop - train both heads simultaneously
            best_ap = 0.0
            best_acc = 0.0
            for ep in range(args.epochs):
                # Log epoch start
                with open(batch_log_file, 'a') as f:
                    f.write(f"\n--- EPOCH {ep+1}/{args.epochs} ---\n")
                
                # Train DETR head only (multi-class)
                losses = train_detr_epoch(head, fp_fn, train_loader, adapter.model.device, lr=args.lr, num_classes=20, amp=True,
                                        batch_log_file=batch_log_file)
                
                # Evaluate DETR head only
                ap50 = evaluate_detr(head, fp_fn, val_loader, adapter.model.device, topk=100, conf_threshold=0.0)
                
                if ap50 > best_ap:
                    best_ap = ap50
                
                # Log epoch
                epoch_log = {
                    "epoch": ep + 1,
                    "detr_loss": losses['loss'],
                    "ap50": ap50,
                }
                feature_log["epochs"].append(epoch_log)

                progress_file = os.path.join(log_dir, f"{tap}_progress.txt")
                with open(progress_file, 'a') as f:
                    f.write(f"Epoch {ep+1:02d} | DETR: {losses['loss']:.5f} | AP@0.5: {ap50:.5f}\n")

                print(f"Epoch {ep+1:02d} | DETR: {losses['loss']:.5f} | AP@0.5: {ap50:.5f}")
            
            # Set final_ap for DETR results
            final_ap = best_ap
        elif args.head == "yolo":
            # YOLO-style head
            from probes.bottleneck import SimpleBottleneck
            from probes.yolo_head import YoloHead, train_yolo_epoch, evaluate_yolo
            # Skip EMA for simplicity
            
            # Only use bottleneck if channels are too high
            if in_ch > 512:
                bottleneck = SimpleBottleneck(in_ch, out_ch=512).to(adapter.model.device)
                head = YoloHead(512, num_classes=20).to(adapter.model.device)
                
                # Wrap fp_fn to include bottleneck
                def fp_bottleneck(imgs):
                    feat, stride = fp_fn(imgs)
                    return bottleneck(feat), stride
                print(f"Using bottleneck: {in_ch} -> 512 channels")
            else:
                bottleneck = None
                head = YoloHead(in_ch, num_classes=20).to(adapter.model.device)
                
                # Use original fp_fn
                def fp_bottleneck(imgs):
                    return fp_fn(imgs)
                print(f"Direct YOLO: {in_ch} channels (no bottleneck needed)")
            
            # Initialize EMA
            ema = None  # No EMA

            # Create single batch log file for this tap
            batch_log_file = os.path.join(log_dir, f"{tap}_batches.txt")
            with open(batch_log_file, 'w') as f:
                f.write(f"YOLO Training - {tap.upper()} Features\n")
                f.write("="*60 + "\n")
            
            best_ap = 0.0
            best_ema_ap = 0.0
            ap_history = []
            
            # Simple training - no phases
            
            for ep in range(args.epochs):
                # Log epoch start
                with open(batch_log_file, 'a') as f:
                    f.write(f"\n--- EPOCH {ep+1}/{args.epochs} ---\n")
                
                # Simple training with normal LR
                losses = train_yolo_epoch(head, fp_bottleneck, train_loader, adapter.model.device, lr=args.lr, 
                                        batch_log_file=batch_log_file, bottleneck=bottleneck, ema=None, feature_adapter=feature_adapter)
                
                # Evaluate model (no EMA) with very low confidence threshold
                ap50 = evaluate_yolo(head, fp_bottleneck, val_loader, adapter.model.device, conf=0.25, ema=None)
                
                ap_history.append(ap50)
                if ap50 > best_ap:
                    best_ap = ap50
                # Log epoch
                epoch_log = {
                    "epoch": ep + 1,
                    "yolo_loss": losses['loss'],
                    "ap50": ap50,
                }
                feature_log["epochs"].append(epoch_log)

                progress_file = os.path.join(log_dir, f"{tap}_progress.txt")
                with open(progress_file, 'a') as f:
                    f.write(f"Epoch {ep+1:02d} | YOLO: {losses['loss']:.3f} | AP@0.5: {ap50:.5f}\n")

                print(f"Epoch {ep+1:02d} | YOLO: {losses['loss']:.3f} | AP@0.5: {ap50:.5f}")

            # Calculate stability metrics
            ap_std = torch.tensor(ap_history).std().item() if len(ap_history) > 1 else 0.0
            final_ap = ap50
            
            # Add stability metrics to feature log
            feature_log["stability"] = {
                "best_ap": best_ap,
                "final_ap": final_ap,
                "ap_std": ap_std,
                "ap_history": ap_history
            }
        else:
            # Center head for detection
            head = CenterHead(in_ch, hidden=args.hidden).to(adapter.model.device)
            
            # Classification head for additional signal testing
            cls_head = ClassificationHead(in_ch, num_classes=20).to(adapter.model.device)
            
            # Training loop - train both heads simultaneously
            best_ap = 0.0
            best_acc = 0.0
            for ep in range(args.epochs):
                # Train both heads simultaneously
                losses, cls_losses = train_center_and_classification_epoch(
                    head, cls_head, fp_fn, train_loader, adapter.model.device, 
                    lr=args.lr
                )
                
                # Evaluate both heads simultaneously
                ap50, acc = evaluate_center_and_classification(head, cls_head, fp_fn, val_loader, adapter.model.device)
                
                if ap50 > best_ap:
                    best_ap = ap50
                    # Save best checkpoint
                    checkpoint_path = os.path.join(checkpoint_dir, f"{tap}_best_center_head.pth")
                    torch.save({
                        'epoch': ep + 1,
                        'model_state_dict': head.state_dict(),
                        'cls_state_dict': cls_head.state_dict(),
                        'ap50': ap50,
                        'accuracy': acc,
                        'losses': losses,
                        'cls_losses': cls_losses
                    }, checkpoint_path)
                    print(f"    Saved best checkpoint: {checkpoint_path}")
                
                if acc > best_acc:
                    best_acc = acc
                
                # Log epoch results
                if args.head == "detr":
                    epoch_log = {
                        "epoch": ep + 1,
                        "detr_loss": losses['loss'],
                        "cls_loss": cls_losses['loss'],
                        "objects": losses['objects'],
                        "ap50": ap50,
                        "accuracy": acc
                    }
                    feature_log["epochs"].append(epoch_log)
                    
                    # Save progress after each epoch
                    progress_file = os.path.join(log_dir, f"{tap}_progress.txt")
                    with open(progress_file, 'a') as f:
                        f.write(f"Epoch {ep+1:02d} | DETR: {losses['loss']:.5f} | CLS: {cls_losses['loss']:.5f} | AP@0.5: {ap50:.5f} | ACC: {acc:.5f}\n")
                    
                    print(f"Epoch {ep+1:02d} | "
                          f"DETR: {losses['loss']:.5f} | "
                          f"CLS: {cls_losses['loss']:.5f} | "
                          f"AP@0.5: {ap50:.5f} | "
                          f"ACC: {acc:.5f}")
                else:  # Center head
                    epoch_log = {
                        "epoch": ep + 1,
                        "center_loss": losses['loss'],
                        "cls_loss": cls_losses['loss'],
                        "objects": losses['objects'],
                        "ap50": ap50,
                        "accuracy": acc
                    }
                    feature_log["epochs"].append(epoch_log)
                    
                    # Save progress after each epoch
                    progress_file = os.path.join(log_dir, f"{tap}_progress.txt")
                    with open(progress_file, 'a') as f:
                        f.write(f"Epoch {ep+1:02d} | Center: {losses['loss']:.3f} | CLS: {cls_losses['loss']:.3f} | AP@0.5: {ap50:.3f} | ACC: {acc:.3f}\n")
                    
                    print(f"Epoch {ep+1:02d} | "
                          f"Center: {losses['loss']:.3f} | "
                          f"CLS: {cls_losses['loss']:.3f} | "
                          f"AP@0.5: {ap50:.3f} | "
                          f"ACC: {acc:.3f}")
            
            # Final evaluation (same as last epoch)
            final_ap = ap50
            final_acc = acc
        
        results[tap] = final_ap
        results_log["features"][tap] = feature_log
        
        if args.head == "center":
            print(f"\n{tap.upper()} final AP@0.5: {final_ap:.3f}, ACC: {final_acc:.3f}")
        else:
            print(f"\n{tap.upper()} final AP@0.5: {final_ap:.3f}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON - {args.model.upper()}")
    print(f"{'='*60}")
    
    if args.head == "center":
        print(f"{'Feature Source':<15} {'AP@0.5':<10} {'ACC':<10} {'Improvement':<15}")
        print(f"{'-'*60}")
    else:
        print(f"{'Feature Source':<15} {'AP@0.5':<10} {'Improvement':<15}")
        print(f"{'-'*50}")
    
    # Get results for trained taps
    tap_results = {}
    for tap in taps_to_train:
        tap_results[tap] = results.get(tap, 0.0)
    
    # Print results with comparisons
    first_tap = True
    baseline_ap = 0
    for tap in taps_to_train:
        ap = tap_results[tap]
        
        if args.head == "center":
            acc = results.get(f"{tap}_acc", 0.0)
            if first_tap:
                print(f"{tap.upper():<15} {ap:<10.3f} {acc:<10.3f} {'-'}")
                baseline_ap = ap
                first_tap = False
            else:
                improvement = ap - baseline_ap
                print(f"{tap.upper():<15} {ap:<10.3f} {acc:<10.3f} {'+' if improvement > 0 else ''}{improvement:+.3f}")
        else:
            if first_tap:
                print(f"{tap.upper():<15} {ap:<10.3f} {'-'}")
                baseline_ap = ap
                first_tap = False
            else:
                improvement = ap - baseline_ap
                print(f"{tap.upper():<15} {ap:<10.3f} {'+' if improvement > 0 else ''}{improvement:+.3f}")
    
    # Find best
    if tap_results:
        best_tap = max(tap_results.keys(), key=lambda k: tap_results[k])
        best_ap = tap_results[best_tap]
        head_type = args.head.upper()
        print(f"\n✓ {best_tap.upper()} features are BEST for detection with {head_type} head ({best_ap:.3f} AP@0.5)")
    
    # Save detailed results to JSON
    results_log["final_results"] = results
    results_log["best_feature"] = max(results.keys(), key=lambda k: results[k])
    
    # Save JSON log
    json_file = os.path.join(log_dir, "results.json")
    with open(json_file, 'w') as f:
        json.dump(results_log, f, indent=2)
    
    # Save human-readable text summary
    txt_file = os.path.join(log_dir, "summary.txt")
    with open(txt_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"VLM FEATURE COMPARISON RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Model: {args.model.upper()}\n")
        f.write(f"Detection Head: {args.head.upper()}\n")
        f.write(f"Training Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.bs}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("FINAL AP@0.5 RESULTS:\n")
        f.write("-" * 50 + "\n")
        for feature, ap in results.items():
            f.write(f"{feature.upper():<15} {ap:.4f}\n")
        
        f.write("\nBEST PERFORMING FEATURE:\n")
        f.write("-" * 30 + "\n")
        best_feature = max(results.keys(), key=lambda k: results[k])
        best_ap = results[best_feature]
        f.write(f"{best_feature.upper()} with {best_ap:.4f} AP@0.5\n\n")
        
        f.write("TRAINING PROGRESS BY FEATURE:\n")
        f.write("=" * 80 + "\n")
        
        for feature_name, feature_data in results_log["features"].items():
            f.write(f"\n{feature_name.upper()} FEATURES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Feature Map Shape: {feature_data['feature_map_shape']}\n")
            f.write(f"Stride: {feature_data['stride']}\n\n")
            
            f.write("Epoch-by-Epoch Results:\n")
            if args.head == "detr":
                f.write("Epoch    Loss      Objects   AP@0.5\n")
                f.write("-" * 35 + "\n")
                for epoch_data in feature_data['epochs']:
                    # Handle different loss key names
                    loss_key = 'detr_loss' if 'detr_loss' in epoch_data else 'loss'
                    objects_key = 'objects' if 'objects' in epoch_data else 0
                    f.write(f"{epoch_data['epoch']:5d}   {epoch_data[loss_key]:8.4f}   "
                           f"{objects_key:8.1f}   {epoch_data['ap50']:6.4f}\n")
            elif args.head == "yolo":
                f.write("Epoch    YOLO Loss  AP@0.5\n")
                f.write("-" * 25 + "\n")
                for epoch_data in feature_data['epochs']:
                    loss_key = 'yolo_loss' if 'yolo_loss' in epoch_data else 'loss'
                    f.write(f"{epoch_data['epoch']:5d}   {epoch_data[loss_key]:8.4f}   "
                           f"{epoch_data['ap50']:6.4f}\n")
            else:
                f.write("Epoch    Center Loss  CLS Loss   Objects   AP@0.5    ACC\n")
                f.write("-" * 60 + "\n")
                for epoch_data in feature_data['epochs']:
                    f.write(f"{epoch_data['epoch']:5d}   {epoch_data['center_loss']:10.4f}   "
                           f"{epoch_data['cls_loss']:8.4f}   {epoch_data['objects']:8.1f}   "
                           f"{epoch_data['ap50']:6.4f}   {epoch_data['accuracy']:6.1f}\n")
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_file}")
    print(f"  Text: {txt_file}")
    print(f"Training completed successfully!")


if __name__ == "__main__":
    main()

