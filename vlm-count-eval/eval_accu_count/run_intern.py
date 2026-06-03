import os
import json
import re
import argparse
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
from collections import defaultdict, Counter
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

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

# InternVL3 image preprocessing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

def build_transform(input_size=448):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = min(target_ratios, key=lambda x: abs(aspect_ratio - x[0] / x[1]))

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def process_case(case, model, tokenizer, data_dir, label_dir, results_dir, question, batch_size, acc_data):
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

    generation_config = dict(max_new_tokens=100, do_sample=False)

    with open(out_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(img_files), batch_size), desc=f"Processing {case}", leave=True):
            batch = img_files[i:i+batch_size]

            for fname in batch:
                try:
                    image_path = os.path.join(img_dir, fname)
                    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
                    gold = load_gold_label(label_subdir, fname)
                    
                    # Use InternVL3's chat interface with preprocessed pixel values
                    raw = model.chat(tokenizer, pixel_values, question, generation_config)
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

def load_model_and_tokenizer(model_name):
    torch._dynamo.config.disable = True
    
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).cuda().eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        use_fast=False
    )
    
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3-8B")
    parser.add_argument("--data_dir", type=str, default="data/images")
    parser.add_argument("--label_dir", type=str, default="data/labels")
    parser.add_argument("--results_dir", type=str, default="internvl3-8b")
    parser.add_argument("--question", type=str, default="How many red circle dots in the image?")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accuracy_file", type=str, default="accuracy.jsonl")
    parser.add_argument("--device", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.device}'
    acc_data = {}
    cases = sorted(d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)))
    for case in cases:
        process_case(case, model, tokenizer, args.data_dir, args.label_dir, args.results_dir, args.question, args.batch_size, acc_data)

    with open(os.path.join(args.results_dir, args.accuracy_file), "w", encoding="utf-8") as f:
        f.write(json.dumps(acc_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()