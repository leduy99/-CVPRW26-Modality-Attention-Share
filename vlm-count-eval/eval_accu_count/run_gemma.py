import os
import json
import re
import argparse
from transformers import pipeline
from tqdm import tqdm
import torch
from collections import defaultdict, Counter

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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

def prepare_messages(batch, img_dir, question):
    return [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant that counts objects in images accurately."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": os.path.join(img_dir, fname)},
                    {"type": "text", "text": question}
                ]
            }
        ]
        for fname in batch
    ]

def process_case(case, pipe, data_dir, label_dir, results_dir, question, batch_size, acc_data):
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
    
    print("HELLOW")

    with open(out_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(img_files), batch_size), desc=f"Batching {case}", leave=True):
            batch = img_files[i:i+batch_size]
            messages = prepare_messages(batch, img_dir, question)

            try:
                outputs = pipe(text=messages)
            except Exception as e:
                outputs = [e] * len(batch)
            
            for fname, out in zip(batch, outputs):
                try:
                    gold = load_gold_label(label_subdir, fname)

                    if isinstance(out, Exception):
                        raise out

                    raw = out[0]["generated_text"][-1]["content"].strip()
                    parsed = extract_number(raw)
                    parsed = -1 if parsed is None else parsed
                except Exception as e:
                    raw = f"[ERROR] {str(e)}"
                    parsed = -1
                    gold = -1

                fout.write(json.dumps({
                    "img_id": fname,
                    "raw": raw,
                    "parsed": parsed,
                    "gold": gold
                }, ensure_ascii=False) + "\n")

                if gold != -1:
                    case_total += 1
                    count_wise[gold]["total"] += 1
                    if parsed == gold:
                        case_correct += 1
                        count_wise[gold]["correct"] += 1
                    elif isinstance(parsed, int):
                        try:
                            abs_diff = abs(parsed - gold)
                            abs_diff_stats[gold] += abs_diff
                        except:
                            pass

    acc_data[case] = {
        "overall_accuracy": case_correct / case_total if case_total > 0 else 0.0,
        "total": case_total,
        "correct": case_correct,
        "per_count_accuracy": {
            str(k): {
                "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
                "total": v["total"],
                "correct": v["correct"]
            } for k, v in count_wise.items()
        },
        "per_count_abs_diff": {
            str(k): diff for k, diff in abs_diff_stats.items()
        }
    }

def load_pipeline(model):
    torch._dynamo.config.disable = True
    return pipeline(
        "image-text-to-text",
        model=model,
        device="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--data_dir", type=str, default="data/images")
    parser.add_argument("--label_dir", type=str, default="data/labels")
    parser.add_argument("--results_dir", type=str, default="gemma-3-4b")
    parser.add_argument("--question", type=str, default="How many red circle dots in the image?")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accuracy_file", type=str, default="accuracy.jsonl")
    parser.add_argument("--device", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    pipe = load_pipeline(args.model)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.device}'

    acc_data = {}
    cases = sorted(d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)))
    for case in cases:
        process_case(case, pipe, args.data_dir, args.label_dir, args.results_dir, args.question, args.batch_size, acc_data)

    with open(os.path.join(args.results_dir, args.accuracy_file), "w", encoding="utf-8") as f:
        f.write(json.dumps(acc_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()