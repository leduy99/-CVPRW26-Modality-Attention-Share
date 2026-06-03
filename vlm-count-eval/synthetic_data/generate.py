import os
import json
import random
import numpy as np
import yaml
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Set

def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def merge_config(defaults: Dict, model_config: Dict) -> Dict:
    """Merge model-specific config with defaults, handling nested dictionaries"""
    import copy
    merged = copy.deepcopy(defaults)
    if model_config:
        for key, value in model_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_config(merged[key], value)
            else:
                merged[key] = value
    return merged

class DatasetGenerator:
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        
        # Extract configuration values
        self.IMAGE_SIZE = config['image_size']
        self.PATCH_SIZE = config['patch_size']
        self.GRID_SIZE = self.IMAGE_SIZE // self.PATCH_SIZE
        self.CELL_SIZE = self.PATCH_SIZE
        self.BASE_RADIUS = self.CELL_SIZE // 2
        self.IMAGES_PER_CASE = config['images_per_case']
        self.COUNT_RANGE = tuple(config['count_range'])
        self.SEED = config['seed']
        self.DRAW_GRID = config['draw_grid']
        self.CASES = config['cases']
        
        # Size variation settings
        self.MIN_RADIUS = self.BASE_RADIUS * config['min_radius_factor']
        self.MAX_RADIUS = self.BASE_RADIUS * config['max_radius_factor']
        
        # Initialize count distribution
        self.count_distribution = self._initialize_count_distribution()
        
        # Output paths
        base_output_path = config.get('base_path', 'data')
        self.output_base = os.path.join(self.model_name, base_output_path)
        self.images_dir = config.get('subdirs', {}).get('images', 'images')
        self.labels_dir = config.get('subdirs', {}).get('labels', 'labels')
        self.patches_dir = config.get('subdirs', {}).get('patches', 'patches')  # New directory for patch data
        self.metadata_filename = config.get('metadata_filename', 'dataset_metadata.json')
        self.stats_filename = config.get('stats_filename', 'dataset_stats.json')
        
        # Set seeds for reproducibility
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        
        print(f"\n=== Configuration for {model_name} ===")
        print(f"  IMAGE_SIZE: {self.IMAGE_SIZE}×{self.IMAGE_SIZE}")
        print(f"  PATCH_SIZE: {self.PATCH_SIZE}×{self.PATCH_SIZE}")
        print(f"  GRID_SIZE: {self.GRID_SIZE}×{self.GRID_SIZE} patches")
        print(f"  Expected attention resolution: {self.GRID_SIZE//2}×{self.GRID_SIZE//2} (after 2×2 compression)")
        print(f"  CELL_SIZE: {self.CELL_SIZE}×{self.CELL_SIZE} pixels per patch")
        print(f"  BASE_RADIUS: {self.BASE_RADIUS} pixels")
        print(f"  Output path: {self.output_base}")
        print(f"  Cases to generate: {len(self.CASES)}")

    def _initialize_count_distribution(self) -> List[int]:
        """Initialize a fair distribution of counts across the range"""
        start, end = self.COUNT_RANGE
        total_counts = end - start + 1
        images_per_count = self.IMAGES_PER_CASE // total_counts
        remainder = self.IMAGES_PER_CASE % total_counts
        
        # Create base distribution
        distribution = []
        for count in range(start, end + 1):
            distribution.extend([count] * images_per_count)
        
        # Add remainder counts randomly
        if remainder > 0:
            additional_counts = random.sample(range(start, end + 1), remainder)
            distribution.extend(additional_counts)
        
        # Shuffle the distribution
        random.shuffle(distribution)
        return distribution

    def get_patch_id(self, x: int, y: int) -> int:
        """Convert pixel coordinates to patch ID (row-major order)"""
        # Calculate which patch this coordinate falls into
        patch_col = min(x // self.PATCH_SIZE, self.GRID_SIZE - 1)
        patch_row = min(y // self.PATCH_SIZE, self.GRID_SIZE - 1)
        
        # Convert to row-major order ID
        patch_id = patch_row * self.GRID_SIZE + patch_col
        return patch_id

    def get_circle_patches(self, x: int, y: int, radius: float) -> Set[int]:
        """Get all patch IDs that a circle overlaps with"""
        patches = set()
        
        # Calculate bounding box of the circle
        min_x = max(0, int(x - radius))
        max_x = min(self.IMAGE_SIZE - 1, int(x + radius))
        min_y = max(0, int(y - radius))
        max_y = min(self.IMAGE_SIZE - 1, int(y + radius))
        
        # Check all patches that the bounding box intersects
        min_patch_col = min_x // self.PATCH_SIZE
        max_patch_col = min(max_x // self.PATCH_SIZE, self.GRID_SIZE - 1)
        min_patch_row = min_y // self.PATCH_SIZE
        max_patch_row = min(max_y // self.PATCH_SIZE, self.GRID_SIZE - 1)
        
        for patch_row in range(min_patch_row, max_patch_row + 1):
            for patch_col in range(min_patch_col, max_patch_col + 1):
                # Check if circle actually intersects with this patch
                patch_left = patch_col * self.PATCH_SIZE
                patch_right = (patch_col + 1) * self.PATCH_SIZE
                patch_top = patch_row * self.PATCH_SIZE
                patch_bottom = (patch_row + 1) * self.PATCH_SIZE
                
                # Find closest point in patch to circle center
                closest_x = max(patch_left, min(x, patch_right))
                closest_y = max(patch_top, min(y, patch_bottom))
                
                # Calculate distance from circle center to closest point
                distance = ((x - closest_x) ** 2 + (y - closest_y) ** 2) ** 0.5
                
                # If distance <= radius, circle intersects with patch
                if distance <= radius:
                    patch_id = patch_row * self.GRID_SIZE + patch_col
                    patches.add(patch_id)
        
        return patches

    def is_circle_within_bounds(self, x: int, y: int, radius: float) -> bool:
        """Check if a circle with given center and radius fits within image bounds with margin"""
        margin = self.CELL_SIZE // 2  # Small margin from edges for final check
        return (x - radius >= margin and 
                y - radius >= margin and 
                x + radius <= self.IMAGE_SIZE - margin and 
                y + radius <= self.IMAGE_SIZE - margin)

    def draw_grid_lines(self, draw: ImageDraw.Draw):
        """Draw grid lines on the image"""
        # Draw vertical lines
        for x in range(0, self.IMAGE_SIZE + 1, self.CELL_SIZE):
            draw.line([(x, 0), (x, self.IMAGE_SIZE)], fill='lightgray', width=1)
        
        # Draw horizontal lines  
        for y in range(0, self.IMAGE_SIZE + 1, self.CELL_SIZE):
            draw.line([(0, y), (self.IMAGE_SIZE, y)], fill='lightgray', width=1)

    def get_cell_center_coordinates(self) -> List[Tuple[int, int]]:
        """Get coordinates at CENTER of each grid cell, excluding edge cells"""
        coords = []
        
        # Add margin of 1 cell from each edge to prevent circles near frame edges
        margin_cells = 1
        
        for i in range(margin_cells, self.GRID_SIZE - margin_cells):
            for j in range(margin_cells, self.GRID_SIZE - margin_cells):
                # Calculate center of each cell
                x = i * self.CELL_SIZE + self.CELL_SIZE // 2
                y = j * self.CELL_SIZE + self.CELL_SIZE // 2
                coords.append((x, y))
        
        return coords

    def is_adjacent_position(self, pos1: Tuple[int, int], pos2: Tuple[int, int], min_gap: int = 2) -> bool:
        """Check if two positions are less than min_gap cells apart"""
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Calculate grid cell indices
        cell_x1, cell_y1 = x1 // self.CELL_SIZE, y1 // self.CELL_SIZE
        cell_x2, cell_y2 = x2 // self.CELL_SIZE, y2 // self.CELL_SIZE
        
        # Check if they're too close (need at least min_gap cells gap in either direction)
        dx = abs(cell_x1 - cell_x2)
        dy = abs(cell_y1 - cell_y2)
        # Return true if they are too close (less than min_gap+1 cells apart in BOTH directions)
        return not (dx >= min_gap + 1 or dy >= min_gap + 1)

    def generate_base_positions(self, count: int, min_gap: int = 2) -> List[Tuple[int, int]]:
        """Generate base circle positions at CENTER of grid cells (Case 1A)"""
        available_coords = self.get_cell_center_coordinates()
        selected_coords = []
        available_coords_copy = available_coords.copy()
        attempts = 0
        max_attempts = 5000
        
        while len(selected_coords) < count and len(available_coords_copy) > 0 and attempts < max_attempts:
            coord = random.choice(available_coords_copy)
            
            # Ensure min_gap block gap
            valid_position = True
            for existing_coord in selected_coords:
                if self.is_adjacent_position(coord, existing_coord, min_gap):
                    valid_position = False
                    break
            
            if valid_position:
                selected_coords.append(coord)
            
            available_coords_copy.remove(coord)
            attempts += 1
        
        return selected_coords

    def transform_positions_for_case(self, base_positions: List[Tuple[int, int]], case_id: str) -> List[Tuple[int, int]]:
        """Transform base positions (1A) to positions for other cases"""
        if case_id in ["1A", "1B", "5A", "6A", "7A", "8A"]:
            # No transformation needed - already at cell centers
            # 5A and 6A use same positioning as 1A but with larger circles
            return base_positions
        
        elif case_id in ["2A", "2B"]:
            # Move to vertical grid lines - shift horizontally by ±CELL_SIZE//2
            transformed = []
            for x, y in base_positions:
                # Move to nearest vertical grid line
                # Find the closest vertical line (multiples of CELL_SIZE)
                left_line = (x // self.CELL_SIZE) * self.CELL_SIZE
                right_line = left_line + self.CELL_SIZE
                
                # Choose the line that keeps us within bounds with margin
                margin = self.CELL_SIZE  # 1 cell margin from edges
                if left_line >= margin:  # Avoid boundary
                    new_x = left_line
                else:
                    new_x = right_line
                
                # Make sure we don't go too close to the right boundary either
                if new_x >= self.IMAGE_SIZE - margin:
                    new_x = left_line
                    
                transformed.append((new_x, y))
            return transformed
        
        elif case_id in ["3A", "3B"]:
            # Move to horizontal grid lines - shift vertically by ±CELL_SIZE//2  
            transformed = []
            for x, y in base_positions:
                # Move to nearest horizontal grid line
                top_line = (y // self.CELL_SIZE) * self.CELL_SIZE
                bottom_line = top_line + self.CELL_SIZE
                
                # Choose the line that keeps us within bounds with margin
                margin = self.CELL_SIZE  # 1 cell margin from edges
                if top_line >= margin:  # Avoid boundary
                    new_y = top_line
                else:
                    new_y = bottom_line
                    
                # Make sure we don't go too close to the bottom boundary either
                if new_y >= self.IMAGE_SIZE - margin:
                    new_y = top_line
                    
                transformed.append((x, new_y))
            return transformed
        
        elif case_id in ["4A", "4B"]:
            # Move to grid intersections - move to nearest intersection
            transformed = []
            for x, y in base_positions:
                # Find nearest grid intersection
                grid_x = round(x / self.CELL_SIZE) * self.CELL_SIZE
                grid_y = round(y / self.CELL_SIZE) * self.CELL_SIZE
                
                # Avoid boundaries with margin
                margin = self.CELL_SIZE  # 1 cell margin from edges
                if grid_x <= margin:
                    grid_x = margin
                elif grid_x >= self.IMAGE_SIZE - margin:
                    grid_x = self.IMAGE_SIZE - margin
                    
                if grid_y <= margin:
                    grid_y = margin
                elif grid_y >= self.IMAGE_SIZE - margin:
                    grid_y = self.IMAGE_SIZE - margin
                    
                transformed.append((grid_x, grid_y))
            return transformed
        
        elif case_id in ["2C", "2D"]:
            # Vertical lines with random horizontal translation (left/right)
            transformed = []
            for x, y in base_positions:
                # First move to vertical grid line (same as 2A/2B)
                left_line = (x // self.CELL_SIZE) * self.CELL_SIZE
                right_line = left_line + self.CELL_SIZE
                
                margin = self.CELL_SIZE  # 1 cell margin from edges
                if left_line >= margin:
                    new_x = left_line
                else:
                    new_x = right_line
                
                if new_x >= self.IMAGE_SIZE - margin:
                    new_x = left_line
                
                # Apply random horizontal translation (25%-75% of circle radius)
                translation_range = self.BASE_RADIUS * 0.5  # 50% range (25% to 75%)
                translation_offset = self.BASE_RADIUS * 0.25  # Start at 25%
                translation = random.uniform(-translation_range, translation_range) + random.choice([-translation_offset, translation_offset])
                new_x = new_x + translation
                
                # Ensure we stay within bounds with margin
                margin = self.CELL_SIZE + self.BASE_RADIUS  # Cell margin plus circle radius
                new_x = max(margin, min(self.IMAGE_SIZE - margin, new_x))
                
                transformed.append((int(new_x), int(y)))
            return transformed
        
        elif case_id in ["3C", "3D"]:
            # Horizontal lines with random vertical translation (up/down)
            transformed = []
            for x, y in base_positions:
                # First move to horizontal grid line (same as 3A/3B)
                top_line = (y // self.CELL_SIZE) * self.CELL_SIZE
                bottom_line = top_line + self.CELL_SIZE
                
                margin = self.CELL_SIZE  # 1 cell margin from edges
                if top_line >= margin:
                    new_y = top_line
                else:
                    new_y = bottom_line
                    
                if new_y >= self.IMAGE_SIZE - margin:
                    new_y = top_line
                
                # Apply random vertical translation (25%-75% of circle radius)
                translation_range = self.BASE_RADIUS * 0.5  # 50% range (25% to 75%)
                translation_offset = self.BASE_RADIUS * 0.25  # Start at 25%
                translation = random.uniform(-translation_range, translation_range) + random.choice([-translation_offset, translation_offset])
                new_y = new_y + translation
                
                # Ensure we stay within bounds with margin
                margin = self.CELL_SIZE + self.BASE_RADIUS  # Cell margin plus circle radius
                new_y = max(margin, min(self.IMAGE_SIZE - margin, new_y))
                
                transformed.append((int(x), int(new_y)))
            return transformed
        
        elif case_id in ["4C", "4D"]:
            # Grid intersections with random translation (both directions)
            transformed = []
            for x, y in base_positions:
                # First move to grid intersection (same as 4A/4B)
                grid_x = round(x / self.CELL_SIZE) * self.CELL_SIZE
                grid_y = round(y / self.CELL_SIZE) * self.CELL_SIZE
                
                # Avoid boundaries with margin
                margin = self.CELL_SIZE  # 1 cell margin from edges
                if grid_x <= margin:
                    grid_x = margin
                elif grid_x >= self.IMAGE_SIZE - margin:
                    grid_x = self.IMAGE_SIZE - margin
                    
                if grid_y <= margin:
                    grid_y = margin
                elif grid_y >= self.IMAGE_SIZE - margin:
                    grid_y = self.IMAGE_SIZE - margin
                
                # Apply random translation in both directions (25%-75% of circle radius)
                translation_range = self.BASE_RADIUS * 0.5  # 50% range (25% to 75%)
                translation_offset = self.BASE_RADIUS * 0.25  # Start at 25%
                
                # Random horizontal translation
                x_translation = random.uniform(-translation_range, translation_range) + random.choice([-translation_offset, translation_offset])
                new_x = grid_x + x_translation
                
                # Random vertical translation
                y_translation = random.uniform(-translation_range, translation_range) + random.choice([-translation_offset, translation_offset])
                new_y = grid_y + y_translation
                
                # Ensure we stay within bounds with margin
                margin = self.CELL_SIZE + self.BASE_RADIUS  # Cell margin plus circle radius
                new_x = max(margin, min(self.IMAGE_SIZE - margin, new_x))
                new_y = max(margin, min(self.IMAGE_SIZE - margin, new_y))
                
                transformed.append((int(new_x), int(new_y)))
            return transformed
        
        return base_positions

    def generate_circle_sizes(self, case_id: str, count: int) -> List[float]:
        """Generate circle sizes based on case size rules"""
        case_data = self.CASES[case_id]
        case_size = case_data["size"]
        
        if case_size == "fixed":
            return [self.BASE_RADIUS] * count
        elif case_size == "varied":
            return [random.uniform(self.MIN_RADIUS, self.MAX_RADIUS) for _ in range(count)]
        elif case_size == "large_25":
            # 2.5 patch diameter = 2.5 * 14 = 35 pixels diameter, so 17.5 pixels radius
            large_radius = 2.5 * self.PATCH_SIZE / 2  # 17.5 pixels
            return [large_radius] * count
        elif case_size == "large_30":
            # 3 patch diameter = 3 * 14 = 42 pixels diameter, so 21 pixels radius
            large_radius = 3.0 * self.PATCH_SIZE / 2  # 21 pixels
            return [large_radius] * count
        elif case_size == "large_35":
            # 3.5 patch diameter = 3.5 * 14 = 49 pixels diameter, so 24.5 pixels radius
            large_radius = 3.5 * self.PATCH_SIZE / 2  # 24.5 pixels
            return [large_radius] * count
        elif case_size == "large_40":
            # 4 patch diameter = 4 * 14 = 56 pixels diameter, so 28 pixels radius
            large_radius = 4.0 * self.PATCH_SIZE / 2  # 28 pixels
            return [large_radius] * count
        else:
            raise ValueError(f"Unknown size type: {case_size}")

    def generate_image_set(self, image_id: int) -> Dict:
        """Generate a complete set of images (all cases) for the same image_id"""
        # Get count from fair distribution
        count = self.count_distribution[image_id]
        
        results = {}
        
        # Generate base positions for different gap requirements
        # Most cases use 2-cell gap
        base_positions_2gap = self.generate_base_positions(count, min_gap=3)
        # Case 6A uses 3-cell gap
        base_positions_3gap = self.generate_base_positions(count, min_gap=4)
        # Case 8A uses 4-cell gap
        base_positions_4gap = self.generate_base_positions(count, min_gap=5)
        
        # Select appropriate base positions for each case
        position_sets = {
            "2gap": base_positions_2gap,
            "3gap": base_positions_3gap, 
            "4gap": base_positions_4gap
        }
        
        # Generate all cases
        for case_id in self.CASES.keys():
            # Determine which position set to use
            if case_id == "6A":
                base_positions = position_sets["3gap"]
                actual_count = len(base_positions)
            elif case_id == "8A":
                base_positions = position_sets["4gap"]
                actual_count = len(base_positions)
            else:
                base_positions = position_sets["2gap"]
                actual_count = len(base_positions)
            
            # If we couldn't generate enough valid circles, proceed with what we have
            if actual_count < count:
                print(f"Warning: Could only generate {actual_count} valid positions out of {count} requested for case {case_id} image {image_id}")
            
            # Transform positions for this case
            positions = self.transform_positions_for_case(base_positions, case_id)
            
            # Generate sizes for this case  
            sizes = self.generate_circle_sizes(case_id, actual_count)
            
            # Create image
            img = Image.new('RGB', (self.IMAGE_SIZE, self.IMAGE_SIZE), 'white')
            draw = ImageDraw.Draw(img)
            
            # Draw grid lines if enabled
            if self.DRAW_GRID:
                self.draw_grid_lines(draw)
            
            # Draw circles and collect data
            circles_data = []
            valid_circles = []
            all_patch_ids = set()
            
            for (x, y), radius in zip(positions, sizes):
                if self.is_circle_within_bounds(x, y, radius):
                    # Draw red circle
                    bbox = [x - radius, y - radius, x + radius, y + radius]
                    draw.ellipse(bbox, fill='red')
                    
                    # Get patch IDs that this circle overlaps with
                    circle_patches = self.get_circle_patches(x, y, radius)
                    all_patch_ids.update(circle_patches)
                    
                    circles_data.append({
                        "x": int(x),
                        "y": int(y), 
                        "radius": float(radius),
                        "patch_ids": sorted(list(circle_patches))
                    })
                    valid_circles.append(((x, y), radius))
            
            # Save image
            case_dir = os.path.join(self.output_base, self.images_dir, case_id)
            os.makedirs(case_dir, exist_ok=True)
            img_path = os.path.join(case_dir, f"img_{image_id}.png")
            img.save(img_path)
            
            results[case_id] = {
                "case": case_id,
                "image_id": image_id,
                "count": len(circles_data),
                "circles": circles_data,
                "patches_with_circles": sorted(list(all_patch_ids)),
                "patch_count": len(all_patch_ids)
            }
        
        return results

    def generate_dataset(self):
        """Generate the complete dataset for this model"""
        print(f"\nStarting Circle Counting Dataset Generation for {self.model_name}")
        print(f"Total images: {len(self.CASES)} cases × {self.IMAGES_PER_CASE} images = {len(self.CASES) * self.IMAGES_PER_CASE} images")
        print(f"Image size: {self.IMAGE_SIZE}×{self.IMAGE_SIZE}")
        print(f"Grid size: {self.GRID_SIZE}×{self.GRID_SIZE}")
        print(f"Cell size: {self.CELL_SIZE}×{self.CELL_SIZE}")
        print()
        
        # Create output directories
        os.makedirs(os.path.join(self.output_base, self.images_dir), exist_ok=True)
        os.makedirs(os.path.join(self.output_base, self.labels_dir), exist_ok=True)
        os.makedirs(os.path.join(self.output_base, self.patches_dir), exist_ok=True)
        
        all_metadata = []
        all_labels = {}  # Dictionary to store all labels by case_id
        all_patches = {}  # Dictionary to store all patch data by case_id
        
        # Initialize label and patch dictionaries for each case
        for case_id in self.CASES.keys():
            all_labels[case_id] = []
            all_patches[case_id] = []
        
        # Generate images with progress bar
        print(f"Generating image sets for {self.model_name}...")
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.generate_image_set, i) 
                for i in range(self.IMAGES_PER_CASE)
            ]
            
            for future in tqdm(as_completed(futures), total=self.IMAGES_PER_CASE, 
                              desc=f"Generating {self.model_name}"):
                image_set_results = future.result()
                
                # Add all cases from this image set to metadata
                for case_id, case_data in image_set_results.items():
                    all_metadata.append(case_data)
        
        # Save complete metadata
        metadata_path = os.path.join(self.output_base, self.metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        # Save merged labels for each case
        for case_id in self.CASES.keys():
            case_data = [item for item in all_metadata if item['case'] == case_id]
            case_labels = []
            case_patches = []
            
            for item in case_data:
                # Extract label data
                label_data = {
                    "image_id": f"img_{item['image_id']}.png",
                    "count": item['count'],
                    "circles": item['circles']
                }
                case_labels.append(label_data)
                
                # Extract patch data
                patch_data = {
                    "image_id": f"img_{item['image_id']}.png",
                    "grid_size": self.GRID_SIZE,
                    "patch_size": self.PATCH_SIZE,
                    "total_patches": self.GRID_SIZE * self.GRID_SIZE,
                    "patches_with_circles": item['patches_with_circles'],
                    "patch_count": item['patch_count']
                }
                case_patches.append(patch_data)
            
            # Save merged labels for this case
            labels_path = os.path.join(self.output_base, self.labels_dir, f"{case_id}_labels.json")
            with open(labels_path, 'w') as f:
                json.dump(case_labels, f, indent=2)
            
            # Save merged patches for this case
            patches_path = os.path.join(self.output_base, self.patches_dir, f"{case_id}_patches.json")
            with open(patches_path, 'w') as f:
                json.dump(case_patches, f, indent=2)
        
        # Generate summary statistics
        case_stats = {}
        for case_id in self.CASES.keys():
            case_data = [item for item in all_metadata if item['case'] == case_id]
            counts = [item['count'] for item in case_data]
            case_stats[case_id] = {
                "images": len(case_data),
                "avg_count": np.mean(counts),
                "min_count": min(counts),
                "max_count": max(counts),
                "std_count": np.std(counts)
            }
        
        stats_path = os.path.join(self.output_base, self.stats_filename)
        with open(stats_path, 'w') as f:
            json.dump(case_stats, f, indent=2)
        
        print(f"\nDataset generation complete for {self.model_name}!")
        print(f"Total images generated: {len(all_metadata)}")
        print("Files saved:")
        print(f"- {self.output_base}/{self.images_dir}/{{case_id}}/img_{{i}}.png - Generated images")
        print(f"- {self.output_base}/{self.labels_dir}/{{case_id}}_labels.json - Merged circle annotations for each case")
        print(f"- {self.output_base}/{self.patches_dir}/{{case_id}}_patches.json - Merged patch ID annotations for each case")
        print(f"- {metadata_path} - Complete metadata")
        print(f"- {stats_path} - Summary statistics")
        
        return len(all_metadata)

def main():
    """Generate datasets for all configured models"""
    print("Circle Counting Dataset Generation")
    print("=" * 50)
    
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError:
        print("Error: config.yml not found. Please create the configuration file.")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return
    
    defaults = config.get('defaults', {})
    models = config.get('models', {})
    output_config = config.get('output', {})
    
    if not models:
        print("No models configured in YAML file.")
        return
    
    print(f"Found {len(models)} model(s) to generate datasets for:")
    for model_name in models.keys():
        description = models[model_name].get('description', 'No description')
        print(f"  - {model_name}: {description}")
    
    total_images_generated = 0
    
    # Generate dataset for each model
    for model_name, model_config in models.items():
        # Merge model-specific config with defaults
        merged_config = merge_config(defaults, model_config)
        merged_config.update(output_config)  # Add output configuration
        
        # Create generator and run
        generator = DatasetGenerator(model_name, merged_config)
        images_count = generator.generate_dataset()
        total_images_generated += images_count
    
    print("\n" + "=" * 50)
    print("ALL DATASETS GENERATION COMPLETE!")
    print(f"Total images generated across all models: {total_images_generated}")
    print(f"Models processed: {', '.join(models.keys())}")
    print()
    print("Key features:")
    print("- Cases 1A-5A/7A use 2-cell gap for standard separation")
    print("- Case 6A uses 3-cell gap for better separation of 3-patch diameter circles")
    print("- Case 8A uses 4-cell gap for optimal separation of 4-patch diameter circles")
    print("- Cases 2A/3A/4A are transformations of 1A positions") 
    print("- B cases use the same positions as A cases but with varied sizes")
    print("- C cases add random translations while keeping circles within grid patches")
    print("- D cases combine C positioning with varied sizes like B cases")
    print("- 5A/6A/7A/8A test multi-patch circles (2.5x, 3x, 3.5x, and 4x patch diameter)")
    print("- Dynamic patch sizing: larger images = more patches = higher attention resolution")
    print("- Patch ID extraction: identifies which patches contain red circles (row-major order: 0,1,2... then 4,5,6...)")
    print("- Merged JSON files for circle coordinates and patch IDs for attention analysis (one file per case)")

if __name__ == "__main__":
    main() 