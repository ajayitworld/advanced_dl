from pathlib import Path
import json
import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate captions for a specific view.
    
    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)
    
    Returns:
        List of caption strings
    """
    file_path = Path(info_path)
    print(file_path)
    base_name = file_path.stem.replace("_info", "") 
    print(file_path.parent)

    #image_file_name = f"{base_name}_{view_index:02d}_im.jpg"
    image_files = list(file_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))
    print(image_files)
    if not image_files:
        raise FileNotFoundError(f"No image found for pattern: {base_name}_{view_index:02d}_im.jpg in {info_path.parent}")
    image_file_name = image_files[0].parent.name + "/" + image_files[0].name
    
    captions = []
    
    # Extract kart objects and track info
    kart_objs = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    
    if not kart_objs:
        return captions
    
    # Find ego kart (center kart)
    ego_kart = next((k for k in kart_objs if k['is_center_kart']), None)
    if not ego_kart:
        return captions
    
    ego_center = ego_kart['center']
    
    # 1. Ego car caption
    #captions.append(f"{ego_kart['kart_name']} is the ego car.")
    captions.append({
        'image_file': image_file_name,
        'caption': f"{ego_kart['kart_name']} is the ego car."     
    })
    
    # 2. Counting caption    
    captions.append({
        'image_file': image_file_name,
        'caption': f"There are {len(kart_objs)} karts in the scene."     
    })
    
    # 3. Track name caption    
    captions.append({
        'image_file': image_file_name,
        'caption': f"The track is {track_name}."     
    })
    
    # 4. Relative position captions for non-ego karts
    for kart in kart_objs:
        if kart['is_center_kart']:
            continue
        
        dx = kart['center'][0] - ego_center[0]
        dy = kart['center'][1] - ego_center[1]
        
        # Determine position
        position_parts = []
        if abs(dx) > 1e-2:
            if dx < 0:
                position_parts.append("left of")
            else:
                position_parts.append("right of")
        
        if position_parts:
            position_str = " and ".join(position_parts)            
            captions.append({
                'image_file': image_file_name,
                'caption': f"{kart['kart_name']} is {position_str} the ego car."     
            })

        position_parts = []
        if abs(dy) > 1e-2:
            if dy < 0:
                position_parts.append("in front of")
            else:
                position_parts.append("behind")
        
        if position_parts:
            position_str = " and ".join(position_parts)            
            captions.append({
                'image_file': image_file_name,
                'caption': f"{kart['kart_name']} is {position_str} the ego car."     
            })
    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def create(train_dir: str = "./data/valid", output_file: str = "./data/valid/valid_captions.json"):
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
                    qa_pairs = generate_caption(str(info_file), view_idx)
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
        
    print(f"Successfully created {output_file}")



"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""

def validate_json():
    import json

    captions_balanced = json.load(open(Path(__file__).parent.parent / "data" / "valid_grader" / "all_mc_qas.json"))
    captions_generated = []
    data_dir = Path(__file__).parent.parent
    captions_files = list(data_dir.glob("data/train/valid_captions.json"))

    for caption_file in captions_files:
        with open(caption_file) as f:
            cap = json.load(f)
            captions_generated.extend(cap)

    print(f"Number of captions golden: {len(captions_balanced)}")
    print(f"Number of captions generated: {len(captions_generated)}")

    generated_by_image = {}
    for cg in captions_generated:
        image_file = cg["image_file"]
        filename = Path(image_file).name
        if filename not in generated_by_image:
            generated_by_image[filename] = []
        generated_by_image[filename].append(cg["caption"])
    
    count_missing = 0
    count_correct = 0

    for idx, cb in enumerate(captions_balanced):
        image_file = cb["image_file"]
        filename = Path(image_file).name
        correct_caption = cb["candidates"][cb["correct_index"]]

        if filename not in generated_by_image:
            print(f"Not found: {cb}")
            count_missing += 1
            continue

        if correct_caption in generated_by_image[filename]:
            count_correct += 1
        else:
            print(f"Wrong answer fpr image {filename}:\nExpected: {correct_caption}\nGenerated: {generated_by_image[filename]}")
        
    print(f"Number of missing images: {count_missing}")
    print(f"Number of correct caption matches: {count_correct} of {len(captions_balanced)}")





def main():
    fire.Fire({"check": check_caption, "create":create, "validate":validate_json})


if __name__ == "__main__":
    main()
    #validate_json()

