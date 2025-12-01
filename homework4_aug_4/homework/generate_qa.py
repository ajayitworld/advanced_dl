# used coilot and chatgpt and online help
import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size
    print(img_width, img_height)

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    with open(info_path, 'r') as f:
        info = json.load(f)

    karts = info.get('karts', [])
    detections = info.get('detections', [])
    if view_index >= len(detections):
        return []
    kart_detections = [det for det in detections[view_index] if det[0] == 1]

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Compute centers and filter out-of-bounds
    kart_objs = []
    for det in kart_detections:
        _, track_id, x1, y1, x2, y2 = det

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        center_x = ((x1_scaled + x2_scaled) // 2) 
        center_y = ((y1_scaled + y2_scaled) // 2) 
        # Filter out-of-bounds
        #if not (0 <= center_x < img_width and 0 <= center_y < img_height):
        #    continue
        kart_name = karts[track_id] if 0 <= track_id < len(karts) else str(track_id)
        kart_objs.append({
            'instance_id': track_id,
            'kart_name': kart_name,
            'center': (center_x, center_y),
            'is_center_kart': False
          
        })
    # Find the kart closest to image center
    if kart_objs:
        img_center = (img_width / 2, img_height / 2)
        dists = [((obj['center'][0] - img_center[0]) ** 2 + (obj['center'][1] - img_center[1]) ** 2) for obj in kart_objs]
        
        min_idx = dists.index(min(dists))
        kart_objs[min_idx]['is_center_kart'] = True
        #print(f"Dists to center: {dists}")
        #print(f"Kart objs: {kart_objs}")
    return kart_objs


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path, 'r') as f:
        info = json.load(f)
    return info.get('track', '')


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    file_path = Path(info_path)
    #print(file_path)
    base_name = file_path.stem.replace("_info", "") 

    #image_file_name = f"{base_name}_{view_index:02d}_im.jpg"
    image_files = list(file_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))

    if not image_files:
        raise FileNotFoundError(f"No image found for pattern: {base_name}_{view_index:02d}_im.jpg in {info_path.parent}")
    image_file_name = image_files[0].parent.name + "/" + image_files[0].name
    #print('image_file=',image_file_name)
    #print(f"Using image file: {info_path.parent}/{image_file}")
    
    


    qa_pairs = []
    kart_objs = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    if not kart_objs:
        return qa_pairs
    # Ego car is the center kart
    ego_kart = next((k for k in kart_objs if k['is_center_kart']), None)
    if not ego_kart:
        return qa_pairs
    ego_center = ego_kart['center']
    # 1. Ego car question
    qa_pairs.append({
        'question': 'What kart is the ego car?',
        'answer': ego_kart['kart_name'],
        'image_file': image_file_name
    })
    # 2. Total karts question
    qa_pairs.append({
        'question': 'How many karts are there in the scenario?',
        'answer': str(len(kart_objs)),
        'image_file': image_file_name
    })
    # 3. Track information question
    qa_pairs.append({
        'question': 'What track is this?',
        'answer': track_name,
        'image_file': image_file_name
    })
    # 4. Relative position questions for each kart (not ego)
    left_count = right_count = front_count = behind_count = 0
    for kart in kart_objs:
        if kart['is_center_kart']:
            continue
        dx = kart['center'][0] - ego_center[0]
        dy = kart['center'][1] - ego_center[1]
        # Left/Right
        lr = 'left' if dx < 0 else 'right'
        if dx < 0:
            left_count += 1
        else:
            right_count += 1
        # Front/Behind (y axis: top is 0, so smaller y is in front)
        fb = 'front' if dy < 0 else 'back'
        if dy < 0:
            front_count += 1
        else:
            behind_count += 1
        # Q: Is {kart_name} to the left or right of the ego car?
        qa_pairs.append({
            'question': f'Is {kart["kart_name"]} to the left or right of the ego car?',
            'answer': lr,
            'image_file': image_file_name
        })
        # Q: Is {kart_name} in front of or behind the ego car?
        qa_pairs.append({
            'question': f'Is {kart["kart_name"]} in front of or behind the ego car?',
            'answer': fb,
            'image_file': image_file_name
        })
        # Q: Where is {kart_name} relative to the ego car?
        rel = []
        if abs(dx) > 1e-2:
            rel.append(lr)
        if abs(dy) > 1e-2:
            rel.append(fb)
        rel_str = ' and '.join(rel) if rel else 'same position'
        qa_pairs.append({
            'question': f'Where is {kart["kart_name"]} relative to the ego car?',
            'answer': rel_str,
            'image_file': image_file_name
        })
    # 5. Counting questions
    qa_pairs.append({
        'question': 'How many karts are to the left of the ego car?',
        'answer': str(left_count),
        'image_file': image_file_name
    })
    qa_pairs.append({
        'question': 'How many karts are to the right of the ego car?',
        'answer': str(right_count),
        'image_file': image_file_name
    })
    qa_pairs.append({
        'question': 'How many karts are in front of the ego car?',
        'answer': str(front_count),
        'image_file': image_file_name
    })
    qa_pairs.append({
        'question': 'How many karts are behind the ego car?',
        'answer': str(behind_count),
        'image_file': image_file_name
    })
    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    print(info_path)
    base_name = info_path.stem.replace("_info", "")
    #image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    image_files = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))
    print(f"{base_name}_{view_index:02d}_im.jpg")

    if not image_files:
        raise FileNotFoundError(f"No image found for pattern: {base_name}_{view_index:02d}_im.jpg in {info_path.parent}")
    image_file = image_files[0]
    print(f"Using image file: {image_file}")

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()
    
    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print(f"A: {qa['image_file']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   cd

You probably need to add additional commands to Fire below.
"""
def create(train_dir: str = "../data/train", output_file: str = "../data/train/train_qa_pairs.json"):
    """
    Generate QA pairs for all info.json files in the train folder and save to a single JSON file.

    Args:
        train_dir: Path to the train directory containing *_info.json files
        output_file: Path to the output JSON file to save all QA pairs
    """
    train_path = Path(train_dir).resolve()
    output_path = Path(output_file).resolve()
    
    if not train_path.exists():
        print(f"Error: Train directory {train_path} does not exist")
        return
    
    # Get all info.json files and sort them
    info_files = sorted(list(train_path.glob("*_info.json")))
    print(f"Found {len(info_files)} info.json files in {train_path}")
    
    
    all_qa_data = []
    for idx, info_file in enumerate(info_files, 1):        
        try:            
            # Load info to determine number of views (detections)
            with open(info_file, 'r') as f:
                info = json.load(f)
            
            num_views = len(info.get('detections', []))
            frame_name = info_file.stem.replace('_info', '')
            track_name = info.get('track', 'unknown')            
            
            print(f"[{idx}/{len(info_files)}] Processing {frame_name} (track: {track_name}, views: {num_views})")
            
            # Generate QA pairs for each view
            for view_idx in range(num_views):
                try:
                    qa_pairs = generate_qa_pairs(str(info_file), view_idx)
                    if qa_pairs:
                        #all_qa_data.append({
                        #    'frame': frame_name,
                        #    'track': track_name,
                        #    'view': '0'+str(view_idx),
                        #    'qa_pairs': qa_pairs
                        #})
                        all_qa_data.extend(qa_pairs)
                except Exception as e:
                    print(f"  Warning: Failed to generate QA for view {view_idx}: {e}")
                    continue
                
        except Exception as e:
            print(f"  Error processing {info_file}: {e}")
            continue
        #print(all_qa_data)            
        
    # Save all QA pairs to output file
    print(f"\nSaving {len(all_qa_data)} QA records to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_qa_data, f, indent=2)
        
    print(f"Successfully created {output_path}")


def test():
    train_dir = Path(__file__).parent.joinpath("..", "data", "valid").resolve()
    print(train_dir)
    info_files = sorted(list(train_dir.glob("*_info.json")))
    if not info_files:
        print("No info.json files found in data/train for test()")
        return
    info_file = str(info_files[0])
    info_file = "../data/valid/00040_info.json"
    view_index = 0
    print(f"Running test with info file: {info_file}, view_index: {view_index}")
    check_qa_pairs(info_file, view_index)

def main():
    fire.Fire({"check": check_qa_pairs, "create": create})


if __name__ == "__main__":
    #main()
    #test()
    create()
