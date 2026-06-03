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

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


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

# SAIL-VL specific image preprocessing
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

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

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=10, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

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

def load_image(image_file, input_size=448, max_num=10):
    image = Image.open(image_file).convert('RGB')
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

    generation_config = dict(max_new_tokens=1024, do_sample=False)

    with open(out_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(img_files), batch_size), desc=f"Processing {case}", leave=True):
            batch = img_files[i:i+batch_size]

            for fname in batch:
                try:
                    image_path = os.path.join(img_dir, fname)
                    pixel_values = load_image(image_path, max_num=10).to(torch.bfloat16)
                    
                    # Move to appropriate device
                    if hasattr(model, 'device'):
                        pixel_values = pixel_values.to(model.device)
                    else:
                        pixel_values = pixel_values.cuda()
                    
                    gold = load_gold_label(label_subdir, fname)
                    
                    # Use SAIL-VL's chat interface with image token
                    question_with_image = f"<image> {question}"
                    raw = model.chat(tokenizer, pixel_values, question_with_image, generation_config)
                    parsed = extract_number(raw)
                    parsed = -1 if parsed is None else parsed
                    
                except Exception as e:
                    print(f"Error processing {fname}: {str(e)}")
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
    
    # Pin to a known working revision to avoid configuration conflicts
    revision = "632909f48cca77a2f5d91a3c79ae0c9200f6e602"  # Known working commit
    
    try:
        # Method 1: Direct loading with minimal configuration
        print("Attempting Method 1: Direct loading...")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            revision=revision,
            _commit_hash=revision  # Alternative way to specify revision
        ).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            use_fast=False,
            revision=revision
        )
        print("Method 1 successful!")
        
    except Exception as e1:
        print(f"Method 1 failed: {e1}")
        print("Trying Method 2: Force architecture override...")
        
        try:
            # Method 2: Override architecture configuration
            from transformers import AutoConfig
            
            config = AutoConfig.from_pretrained(
                model_name, 
                trust_remote_code=True,
                revision=revision
            )
            
            # Fix architecture conflicts
            if hasattr(config, 'architectures') and config.architectures:
                print(f"Original architectures: {config.architectures}")
                # Try to find a compatible architecture
                if 'InternLM2ForCausalLM' in str(config.architectures):
                    print("Detected InternLM2 architecture conflict, attempting fix...")
                    # Remove problematic architecture reference
                    config.architectures = ['SAILVLForConditionalGeneration']
            
            model = AutoModel.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                revision=revision
            ).eval()
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                use_fast=False,
                revision=revision
            )
            print("Method 2 successful!")
            
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            print("Trying Method 3: Local configuration override...")
            
            try:
                # Method 3: Load and manually fix configuration
                import tempfile
                import os
                
                # Download model to temp directory and modify config
                from huggingface_hub import snapshot_download
                
                cache_dir = snapshot_download(
                    repo_id=model_name,
                    revision=revision,
                    ignore_patterns=["*.safetensors", "*.bin"]  # Only download config files
                )
                
                # Read and fix config.json
                config_path = os.path.join(cache_dir, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    # Remove problematic architecture entries
                    if 'architectures' in config_data:
                        print(f"Fixing architectures in config: {config_data['architectures']}")
                        config_data['architectures'] = ['SAILVLForConditionalGeneration']
                    
                    # Write back fixed config
                    with open(config_path, 'w') as f:
                        json.dump(config_data, f, indent=2)
                
                # Now load from the fixed local path
                model = AutoModel.from_pretrained(
                    cache_dir,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto",
                    local_files_only=True
                ).eval()
                
                tokenizer = AutoTokenizer.from_pretrained(
                    cache_dir,
                    trust_remote_code=True, 
                    use_fast=False,
                    local_files_only=True
                )
                print("Method 3 successful!")
                
            except Exception as e3:
                print(f"Method 3 failed: {e3}")
                print("All methods failed. Please check:")
                print("1. Internet connection")
                print("2. HuggingFace token permissions") 
                print("3. Model availability")
                print("4. Try updating transformers: pip install transformers --upgrade")
                raise Exception(f"Could not load model after trying all methods. Last error: {e3}")
    
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="BytedanceDouyinContent/SAIL-VL-1d6-8B")
    parser.add_argument("--data_dir", type=str, default="data/images")
    parser.add_argument("--label_dir", type=str, default="data/labels")
    parser.add_argument("--results_dir", type=str, default="sail-vl-1d6-8b")
    parser.add_argument("--question", type=str, default="How many red circle dots in the image?")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accuracy_file", type=str, default="accuracy.jsonl")
    return parser.parse_args()

def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model)

    acc_data = {}
    cases = sorted(d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)))
    for case in cases:
        process_case(case, model, tokenizer, args.data_dir, args.label_dir, args.results_dir, args.question, args.batch_size, acc_data)

    with open(os.path.join(args.results_dir, args.accuracy_file), "w", encoding="utf-8") as f:
        f.write(json.dumps(acc_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main() 