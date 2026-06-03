# Synthetic task data generation

This tool generates synthetic datasets of images containing red circles for testing vision models' counting abilities.

## Overview

The dataset generator creates images with red circles placed in various configurations to test a model's ability to:
- Count objects accurately
- Detect objects of varying sizes
- Recognize objects in different positions (grid cells, lines, intersections)
- Handle objects that span multiple attention patches

## Configuration

Configuration is managed through `config.yml`:

- **Global defaults**: Base settings for all models
- **Model-specific settings**: Override defaults for specific models
- **Output settings**: Control file paths and organization

### Key Parameters

- `image_size`: Width/height of generated images (e.g., 384×384)
- `patch_size`: Size of attention patches (e.g., 14×14 or 27×27)
- `images_per_case`: Number of images to generate per case
- `count_range`: Range of circles to include in images [min, max]
- `cases`: Different placement and size configurations

## Cases

The generator creates multiple test cases:

- **1A-4A**: Fixed-size circles in different positions (cells, vertical lines, horizontal lines, intersections)
- **1B-4B**: Same positions as A cases but with varied circle sizes
- **2C-4D**: Random translations from grid positions
- **5A-8A**: Progressively larger circles (2.5×, 3×, 3.5×, and 4× patch diameter)

## Usage

Run the generator with command below after you set your `config.yml`
```
python generate.py
```
