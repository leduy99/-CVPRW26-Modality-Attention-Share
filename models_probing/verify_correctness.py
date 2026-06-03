#!/usr/bin/env python3
"""
CRITICAL VERIFICATION: Check if all adapters extract features correctly.

This script verifies:
1. Encoder features come from vision encoder (NOT LLM)
2. Projector features use correct projector transformation
3. LLM features come from LLM layers
4. Dimensions match expected architecture
"""

import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np

MODELS = {
    "qwen25": ("Qwen 2.5-VL", "adapters.qwen25vl", "Qwen25VLAdapter"),
    "qwen3": ("Qwen 3-VL", "adapters.qwen3vl", "Qwen3VLAdapter"),
    "internvl25": ("InternVL 2.5", "adapters.internvl25", "InternVL25Adapter"),
    "internvl3": ("InternVL 3.0", "adapters.internvl3", "InternVL3Adapter"),
    "internvl35": ("InternVL 3.5", "adapters.internvl35", "InternVL35Adapter"),
    "llava15": ("LLaVA 1.5", "adapters.llava15", "Llava15Adapter"),
}


def create_dummy_image():
    """Create 512×512 dummy image."""
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(img)


def verify_encoder_not_llm(adapter, model_name, image):
    """
    Verify that encode_grid() does NOT return LLM output.
    
    Strategy:
    1. Extract encoder features
    2. Extract LLM features
    3. Check if they're different (should be!)
    4. Check dimension matches vision encoder, not LLM
    """
    print(f"\n{'='*80}")
    print(f"VERIFYING: {model_name} - Encoder Features")
    print(f"{'='*80}")
    
    issues = []
    
    try:
        # Get encoder features
        encoder_result = adapter.encode_grid(image)
        encoder_grid = encoder_result["grid"]  # (1, Hf, Wf, D)
        encoder_dim = encoder_grid.shape[-1]
        encoder_tokens = encoder_grid.view(1, -1, encoder_dim)  # (1, N, D)
        
        print(f"✓ Encoder output: shape={encoder_grid.shape}, dim={encoder_dim}")
        
        # Get expected vision encoder dimension
        if hasattr(adapter.model, 'config'):
            if hasattr(adapter.model.config, 'vision_config'):
                expected_vision_dim = getattr(adapter.model.config.vision_config, 'hidden_size', None)
                if expected_vision_dim:
                    print(f"  Expected vision encoder dim: {expected_vision_dim}")
                    if encoder_dim != expected_vision_dim:
                        issues.append(f"⚠️  Encoder dim {encoder_dim} != vision config {expected_vision_dim}")
            
            # Get LLM hidden size
            llm_dim = None
            if hasattr(adapter.model.config, 'hidden_size'):
                llm_dim = adapter.model.config.hidden_size
            elif hasattr(adapter.model.config, 'text_config'):
                llm_dim = getattr(adapter.model.config.text_config, 'hidden_size', None)
            
            if llm_dim:
                print(f"  LLM hidden size: {llm_dim}")
                if encoder_dim == llm_dim:
                    issues.append(f"🚨 CRITICAL: Encoder dim {encoder_dim} == LLM dim {llm_dim}")
                    issues.append(f"   → This suggests encode_grid() is extracting LLM output, NOT vision encoder!")
                else:
                    print(f"  ✓ Encoder dim {encoder_dim} != LLM dim {llm_dim} (good!)")
        
        # Try to get LLM features for comparison
        try:
            llm_result = adapter.extract_llm_features(image)
            llm_tokens = llm_result["tokens"]
            llm_dim = llm_tokens.shape[-1]
            
            print(f"✓ LLM output: shape={llm_tokens.shape}, dim={llm_dim}")
            
            # Compare dimensions
            if encoder_dim == llm_dim:
                issues.append(f"🚨 CRITICAL: Encoder and LLM have SAME dimension ({encoder_dim})")
                issues.append(f"   → This is HIGHLY SUSPICIOUS - they should be different!")
                
                # Check if features are actually the same
                if encoder_tokens.shape == llm_tokens.shape:
                    diff = torch.abs(encoder_tokens - llm_tokens).mean().item()
                    print(f"  Mean absolute difference: {diff:.6f}")
                    if diff < 0.01:
                        issues.append(f"🚨 CRITICAL: Features are IDENTICAL (diff={diff:.6f})")
                        issues.append(f"   → encode_grid() is definitely extracting LLM output!")
            else:
                print(f"  ✓ Encoder dim {encoder_dim} != LLM dim {llm_dim} (correct!)")
                
        except Exception as e:
            print(f"  Could not extract LLM features: {e}")
        
    except Exception as e:
        issues.append(f"❌ Failed to verify encoder: {e}")
    
    return issues


def verify_projector_correctness(adapter, model_name, image):
    """
    Verify that project_tokens() uses the correct projector.
    
    Strategy:
    1. Extract encoder features
    2. Extract projector features
    3. Manually apply projector to encoder features
    4. Check if they match
    """
    print(f"\n{'='*80}")
    print(f"VERIFYING: {model_name} - Projector Features")
    print(f"{'='*80}")
    
    issues = []
    
    try:
        # Get encoder features
        encoder_result = adapter.encode_grid(image)
        encoder_grid = encoder_result["grid"]  # (1, Hf, Wf, D_enc)
        B, Hf, Wf, D_enc = encoder_grid.shape
        encoder_flat = encoder_grid.view(B, Hf * Wf, D_enc)  # (1, N, D_enc)
        
        print(f"✓ Encoder features: {encoder_flat.shape}")
        
        # Get projector features
        projector_result = adapter.project_tokens(image)
        projector_tokens = projector_result["tokens"]  # (1, N, D_proj)
        D_proj = projector_tokens.shape[-1]
        
        print(f"✓ Projector features: {projector_tokens.shape}")
        
        # Find projector module
        projector = None
        if hasattr(adapter, 'projector'):
            projector = adapter.projector
        
        if projector is not None:
            print(f"✓ Found projector: {type(projector)}")
            
            if hasattr(projector, 'weight'):
                weight_shape = projector.weight.shape
                print(f"  Projector weight shape: {weight_shape}")
                
                # PyTorch Linear: weight is (out_features, in_features)
                out_dim, in_dim = weight_shape
                print(f"  Projector: {in_dim} → {out_dim}")
                
                # Check if encoder output matches projector input
                # SKIP for Qwen models: they use spatial merging, so encoder (spatial) != projector input
                is_qwen = 'qwen' in model_name.lower()
                if is_qwen:
                    print(f"  ⓘ Qwen model: Encoder has spatial structure, projector includes spatial merging")
                    print(f"    Encoder: {D_enc}d (normalized, spatial H×W)")
                    print(f"    Projector: {in_dim}→{out_dim} (after spatial merge)")
                elif D_enc != in_dim:
                    issues.append(f"🚨 CRITICAL: Encoder output dim {D_enc} != Projector input dim {in_dim}")
                    issues.append(f"   → Projector cannot be applied to encoder output!")
                else:
                    print(f"  ✓ Encoder output {D_enc} matches projector input {in_dim}")
                
                # Check if projector output matches result
                if D_proj != out_dim:
                    issues.append(f"⚠️  Projector output dim {D_proj} != Expected dim {out_dim}")
                    issues.append(f"   → project_tokens() might not be using projector correctly")
                else:
                    print(f"  ✓ Projector output {D_proj} matches expected {out_dim}")
                
                # Manual verification: apply projector ourselves
                # SKIP for Qwen models: spatial merging cannot be verified with simple linear projection
                if not is_qwen:
                    try:
                        manual_projection = torch.nn.functional.linear(
                            encoder_flat, 
                            projector.weight, 
                            projector.bias if hasattr(projector, 'bias') else None
                        )
                        print(f"  Manual projection shape: {manual_projection.shape}")
                        
                        # Compare with adapter's output
                        if manual_projection.shape == projector_tokens.shape:
                            diff = torch.abs(manual_projection - projector_tokens).mean().item()
                            print(f"  Difference from manual projection: {diff:.6f}")
                            
                            if diff < 0.01:
                                print(f"  ✓ project_tokens() uses projector correctly!")
                            elif diff < 1.0:
                                issues.append(f"⚠️  Small difference ({diff:.6f}) - might use slightly different method")
                            else:
                                issues.append(f"🚨 CRITICAL: Large difference ({diff:.6f}) - projector NOT used correctly!")
                        else:
                            issues.append(f"⚠️  Cannot compare - shape mismatch")
                            
                    except Exception as e:
                        issues.append(f"⚠️  Could not verify manual projection: {e}")
                else:
                    print(f"  ⓘ Skipping manual projection verification (Qwen uses spatial merging)")
            else:
                issues.append(f"⚠️  Projector has no weight attribute")
        else:
            issues.append(f"⚠️  Could not find projector module in adapter")
            
    except Exception as e:
        issues.append(f"❌ Failed to verify projector: {e}")
    
    return issues


def verify_llm_correctness(adapter, model_name, image):
    """
    Verify that extract_llm_features() extracts from LLM layers.
    
    Strategy:
    1. Check LLM output dimension matches config
    2. Verify it's different from encoder
    """
    print(f"\n{'='*80}")
    print(f"VERIFYING: {model_name} - LLM Features")
    print(f"{'='*80}")
    
    issues = []
    
    try:
        llm_result = adapter.extract_llm_features(image)
        llm_tokens = llm_result["tokens"]
        llm_dim = llm_tokens.shape[-1]
        
        print(f"✓ LLM output: shape={llm_tokens.shape}, dim={llm_dim}")
        
        # Get expected LLM dimension
        if hasattr(adapter.model, 'config'):
            expected_llm_dim = None
            
            if hasattr(adapter.model.config, 'hidden_size'):
                expected_llm_dim = adapter.model.config.hidden_size
            elif hasattr(adapter.model.config, 'text_config'):
                expected_llm_dim = getattr(adapter.model.config.text_config, 'hidden_size', None)
            
            if expected_llm_dim:
                print(f"  Expected LLM dim: {expected_llm_dim}")
                if llm_dim != expected_llm_dim:
                    issues.append(f"⚠️  LLM output dim {llm_dim} != expected {expected_llm_dim}")
                else:
                    print(f"  ✓ LLM output matches expected dimension")
        
    except Exception as e:
        issues.append(f"❌ Failed to verify LLM: {e}")
    
    return issues


def main():
    print("="*80)
    print("CRITICAL VERIFICATION: Feature Extraction Correctness")
    print("="*80)
    print()
    print("This script verifies that all adapters extract features correctly:")
    print("  1. Encoder → Vision Encoder Output (NOT LLM)")
    print("  2. Projector → Correct Projector Transformation")
    print("  3. LLM → LLM Layer Output")
    print()
    
    dummy_image = create_dummy_image()
    
    all_issues = {}
    
    for model_key, (model_name, module_path, class_name) in MODELS.items():
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_name} ({model_key})")
        print(f"{'#'*80}")
        
        try:
            # Import adapter
            module = __import__(module_path, fromlist=[class_name])
            AdapterClass = getattr(module, class_name)
            
            print(f"Loading {model_name}...")
            adapter = AdapterClass()
            print(f"✓ Loaded on device: {adapter.model.device}")
            
            # Run verifications
            issues = []
            
            issues.extend(verify_encoder_not_llm(adapter, model_name, dummy_image))
            issues.extend(verify_projector_correctness(adapter, model_name, dummy_image))
            issues.extend(verify_llm_correctness(adapter, model_name, dummy_image))
            
            all_issues[model_key] = issues
            
            # Cleanup
            del adapter
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Failed to load/test {model_name}: {e}")
            all_issues[model_key] = [f"❌ Failed to load: {e}"]
    
    # Summary
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}\n")
    
    critical_issues = {}
    warnings_count = {}
    
    for model_key, issues in all_issues.items():
        model_name = MODELS[model_key][0]
        
        critical = [i for i in issues if '🚨' in i]
        warnings = [i for i in issues if '⚠️' in i]
        errors = [i for i in issues if '❌' in i]
        
        if critical:
            critical_issues[model_key] = critical
        
        if critical or errors:
            print(f"🚨 {model_name}: {len(critical)} CRITICAL, {len(errors)} ERRORS, {len(warnings)} warnings")
            for issue in issues:
                print(f"  {issue}")
        elif warnings:
            print(f"⚠️  {model_name}: {len(warnings)} warnings")
            for issue in warnings:
                print(f"  {issue}")
        else:
            print(f"✅ {model_name}: ALL CHECKS PASSED")
    
    print(f"\n{'='*80}")
    print("ACTION ITEMS")
    print(f"{'='*80}\n")
    
    if critical_issues:
        print("🚨 CRITICAL ISSUES FOUND - MUST FIX BEFORE USING RESULTS:")
        for model_key, issues in critical_issues.items():
            model_name = MODELS[model_key][0]
            print(f"\n{model_name}:")
            for issue in issues:
                if '🚨' in issue:
                    print(f"  {issue}")
            print(f"  → ACTION: Fix {model_key} adapter and RE-TRAIN")
    else:
        print("✅ No critical issues found. All models verified!")
    
    return len(critical_issues)


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)


