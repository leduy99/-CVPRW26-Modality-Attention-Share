import torch
from transformers import AutoModel, AutoProcessor


def main():
    model_id = "OpenGVLab/InternVL2_5-2B"
    print(f"Loading {model_id} ...")
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # List top-level modules
    print("\nTop-level modules:")
    for name, mod in model.named_children():
        print(f"- {name}: {mod.__class__.__name__}")

    # Find vision model, projector, and llm layers
    vision = None
    projector = None
    llm_layers = None

    for name, mod in model.named_modules():
        if vision is None and name == "vision_model":
            vision = mod
        if projector is None and ("visual_projector" in name or "projector" in name):
            projector = mod
        if llm_layers is None and name.endswith("language_model.model") and hasattr(mod, "layers"):
            llm_layers = mod.layers

    print("\nDetected components:")
    print(f"- vision_model: {'FOUND' if vision else 'MISSING'}")
    print(f"- projector: {'FOUND' if projector else 'MISSING'}")
    print(f"- llm_layers: {len(llm_layers) if llm_layers is not None else 'MISSING'}")

    # Print first 3 layer types for LLM
    if llm_layers is not None:
        print("\nFirst 3 LLM layers:")
        for i, layer in enumerate(llm_layers[:3]):
            print(f"  [{i}] {layer.__class__.__name__}")

    # Quick dry run to confirm processor IO schema
    from PIL import Image
    import numpy as np
    img = Image.fromarray((np.random.rand(448, 448, 3) * 255).astype('uint8'))
    prompt = "Describe the objects in this image."
    messages = [f"<image>\n{prompt}"]
    inputs = processor(text=messages, images=img, return_tensors="pt").to(model.device)
    print("\nProcessor keys:", list(inputs.keys()))
    for k, v in inputs.items():
        try:
            print(f"  - {k}: {tuple(v.shape)} {v.dtype}")
        except Exception:
            print(f"  - {k}: {type(v)}")

    # Hook one LLM layer output
    target_layer_idx = max(0, (len(llm_layers) - 2) if llm_layers is not None else 0)
    llm_output = {}
    if llm_layers is not None:
        handle = llm_layers[target_layer_idx].register_forward_hook(
            lambda m, i, o: llm_output.setdefault("hidden", (o[0] if isinstance(o, tuple) else o).detach())
        )
        _ = model(**inputs, return_dict=True)
        handle.remove()
        if "hidden" in llm_output:
            hs = llm_output["hidden"]
            print(f"\nCaptured LLM layer {target_layer_idx} hidden: {tuple(hs.shape)}")
        else:
            print("\nFailed to capture LLM hidden state.")


if __name__ == "__main__":
    main()



