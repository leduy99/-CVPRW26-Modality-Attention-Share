import os
import json
import re
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from collections import defaultdict, Counter
from transformers import AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = f'0'
WORD_TO_NUM = {
    'zero': '0', 'no': '0',
    'one': '1', 'two': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6',
    'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12',
    'thirteen': '13', 'fourteen': '14', 'fifteen': '15'
}
word_pattern = '|'.join(WORD_TO_NUM.keys())
number_pattern = rf'(\d+|{word_pattern})'

def extract_number(text):
    m = re.search(number_pattern, text.lower())
    if not m:
        return None
    t = m.group(1)
    return int(WORD_TO_NUM.get(t, t))

def load_gold_label(label_dir, fname):
    try:
        with open(os.path.join(label_dir, f"{fname.split('.png')[0]}.json"), "r", encoding="utf-8") as f:
            return json.load(f).get("count", -1)
    except:
        return -1

def prepare_prompt(question):
    return f"<image>\n{question}"

def load_model_and_tokenizers(model_name):
    # First, modify the config
    import json
    import os
    from huggingface_hub import snapshot_download
    
    # Download the model files
    cache_dir = snapshot_download(model_name)
    
    # Update config.json
    config_path = os.path.join(cache_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Disable flash attention in config
    config["use_flash_attn"] = False
    config["llm_attn_implementation"] = "eager"
    config["attn_implementation"] = "eager"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Now load the model with modified config
    model = AutoModelForCausalLM.from_pretrained(
        cache_dir,
        torch_dtype=torch.bfloat16,
        multimodal_max_length=32768,
        trust_remote_code=True,
        use_cache=True,
        attn_implementation="eager",
        use_flash_attention_2=False,
        low_cpu_mem_usage=True
    ).cuda()
    
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    return model, text_tokenizer, visual_tokenizer

def process_case(case, model, text_tokenizer, visual_tokenizer, data_dir, label_dir, results_dir, question, batch_size, acc_data):
    img_dir = os.path.join(data_dir, case)
    label_subdir = os.path.join(label_dir, case)
    img_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
    if not img_files:
        return

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{case}.jsonl")

    case_correct = 0
    case_total = 0
    count_wise = defaultdict(lambda: {"correct": 0, "total": 0})
    abs_diff_stats = Counter()

    with open(out_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(img_files), batch_size), desc=f"Batching {case}", leave=True):
            batch = img_files[i:i+batch_size]
            
            for fname in batch:
                try:
                    # Load and process image
                    image = Image.open(os.path.join(img_dir, fname))
                    images = [image]
                    query = prepare_prompt(question)
                    
                    # Format conversation
                    prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=9)
                    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
                    input_ids = input_ids.unsqueeze(0).to(device=model.device)
                    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
                    if pixel_values is not None:
                        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
                    pixel_values = [pixel_values]

                    # Generate output
                    with torch.inference_mode():
                        gen_kwargs = dict(
                            max_new_tokens=100,
                            do_sample=False,
                            temperature=0.0,
                            top_p=None,
                            top_k=None,
                            repetition_penalty=None,
                            eos_token_id=model.generation_config.eos_token_id,
                            pad_token_id=text_tokenizer.pad_token_id,
                            use_cache=True
                        )
                        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                        raw = text_tokenizer.decode(output_ids, skip_special_tokens=True)
                    
                    gold = load_gold_label(label_subdir, fname)
                    parsed = extract_number(raw)
                    parsed = -1 if parsed is None else parsed

                except Exception as e:
                    raw = f"[ERROR] {str(e)}"
                    parsed = -1
                    gold = -1

                result = {
                    "img_id": fname,
                    "raw": raw,
                    "parsed": parsed,
                    "gold": gold
                }
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")

                if gold != -1 and parsed != -1:  # Only process valid predictions
                    case_total += 1
                    count_wise[gold]["total"] += 1
                    if parsed == gold:
                        case_correct += 1
                        count_wise[gold]["correct"] += 1
                    abs_diff_stats[gold] += abs(parsed - gold)

    # Calculate accuracy only if we have valid samples
    accuracy = case_correct / case_total if case_total > 0 else 0.0
    
    acc_data[case] = {
        "overall_accuracy": accuracy,
        "total": case_total,
        "correct": case_correct,
        "per_count_accuracy": {
            str(k): {
                "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0,
                "total": v["total"],
                "correct": v["correct"]
            } for k, v in count_wise.items()
        },
        "per_count_abs_diff": {str(k): v for k, v in abs_diff_stats.items()}
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="AIDC-AI/Ovis2-8B")
    parser.add_argument("--data_dir", type=str, default="data/images")
    parser.add_argument("--label_dir", type=str, default="data/labels")
    parser.add_argument("--results_dir", type=str, default="ovis-8b")
    parser.add_argument("--question", type=str, default="How many red circle dots in the image?")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accuracy_file", type=str, default="accuracy.jsonl")
    parser.add_argument("--device", type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    model, text_tokenizer, visual_tokenizer = load_model_and_tokenizers(args.model)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.device}'
    os.environ["CUDA_VISIBLE_DEVICES"] = f'0'
    acc_data = {}
    cases = sorted(d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)))
    for case in cases:
        process_case(case, model, text_tokenizer, visual_tokenizer, args.data_dir, args.label_dir, args.results_dir, args.question, args.batch_size, acc_data)

    with open(os.path.join(args.results_dir, args.accuracy_file), "w", encoding="utf-8") as f:
        f.write(json.dumps(acc_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main() 