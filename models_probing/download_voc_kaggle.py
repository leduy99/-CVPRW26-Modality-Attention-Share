"""
Download Pascal VOC 2007 dataset using Kaggle API.
"""
import kagglehub
import os
import shutil
from pathlib import Path

def download_voc_kaggle():
    """Download VOC 2007 dataset from Kaggle."""
    print("📥 Downloading Pascal VOC 2007 from Kaggle...")
    
    try:
        # Download dataset
        path = kagglehub.dataset_download("zaraks/pascal-voc-2007")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Create target directory
        target_dir = "./vocdata"
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy files to target directory
        print("📁 Copying files to ./vocdata...")
        
        # Find the VOC2007 directory in the downloaded path
        voc_dir = None
        for root, dirs, files in os.walk(path):
            if "VOC2007" in dirs:
                voc_dir = os.path.join(root, "VOC2007")
                break
        
        if voc_dir and os.path.exists(voc_dir):
            # Copy entire VOC2007 directory
            target_voc = os.path.join(target_dir, "VOC2007")
            if os.path.exists(target_voc):
                shutil.rmtree(target_voc)
            shutil.copytree(voc_dir, target_voc)
            print(f"✅ VOC2007 copied to: {target_voc}")
            
            # List contents to verify
            print("\n📋 Dataset contents:")
            for item in os.listdir(target_voc):
                item_path = os.path.join(target_voc, item)
                if os.path.isdir(item_path):
                    count = len(os.listdir(item_path))
                    print(f"  • {item}/ ({count} items)")
                else:
                    print(f"  • {item}")
            
            return True
        else:
            print("❌ VOC2007 directory not found in downloaded files")
            print(f"Available in {path}:")
            for item in os.listdir(path):
                print(f"  • {item}")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def test_voc_loading():
    """Test loading VOC dataset after download."""
    print("\n🧪 Testing VOC dataset loading...")
    
    try:
        from datasets.voc_center import VOCCenterDataset
        
        # Test trainval set
        print("Loading trainval set...")
        train_ds = VOCCenterDataset(root="./vocdata", year="2007", image_set="trainval")
        print(f"✅ Trainval: {len(train_ds)} samples")
        
        # Test test set
        print("Loading test set...")
        test_ds = VOCCenterDataset(root="./vocdata", year="2007", image_set="test")
        print(f"✅ Test: {len(test_ds)} samples")
        
        # Test loading a sample
        print("Testing sample loading...")
        img, boxes = train_ds[0]
        print(f"✅ Sample: image shape {img.shape}, boxes shape {boxes.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing dataset: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Pascal VOC 2007 Downloader")
    print("=" * 40)
    
    # Download dataset
    success = download_voc_kaggle()
    
    if success:
        # Test loading
        test_success = test_voc_loading()
        
        if test_success:
            print("\n🎉 Dataset ready for training!")
            print("You can now run:")
            print("python train_compare.py --model ovis25 --epochs 5 --bs 2")
        else:
            print("\n⚠️  Dataset downloaded but loading failed")
    else:
        print("\n❌ Dataset download failed")
