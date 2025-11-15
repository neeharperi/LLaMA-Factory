import json
import re
from typing import Dict, List, Tuple

def parse_dataset_file(file_path: str) -> Dict:
    """Parse the dataset text file and extract relevant information."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract overview section to get class names
    overview_match = re.search(r'# Overview\n(.*?)(?=\n# |\Z)', content, re.DOTALL)
    overview_content = overview_match.group(1).strip() if overview_match else ""
    
    # Extract class names from overview links
    class_links = re.findall(r'- \[(.+?)\]\(#(.+?)\)', overview_content)
    class_name_mapping = {}
    for display_name, link_name in class_links:
        if display_name not in ['Introduction', 'Object Classes']:  # Skip non-class entries
            class_name_mapping[display_name] = link_name
    
    # Extract introduction section
    intro_match = re.search(r'# Introduction\n(.*?)(?=\n# Object Classes|\n#|\Z)', content, re.DOTALL)
    introduction = intro_match.group(1).strip() if intro_match else ""
    
    # Extract object classes section - everything after "# Object Classes"
    classes_match = re.search(r'# Object Classes.*?\n(.*)', content, re.DOTALL)
    classes_content = classes_match.group(1).strip() if classes_match else ""
    
    print(f"Classes content length: {len(classes_content)}")
    print(f"Classes content first 500 chars: '{classes_content[:500]}'")  # Debug print
    
    # Parse individual classes - more flexible regex
    class_sections = re.findall(r'##\s+(.+?)\s*\n(.*?)(?=\n##\s+|\Z)', classes_content, re.DOTALL)
    print(f'\n\nClass Sections {class_sections}')
    classes = {}
    for display_class_name, class_content in class_sections:
        display_class_name = display_class_name.strip()
        
        # Get the actual class name from the mapping
        actual_class_name = class_name_mapping.get(display_class_name, display_class_name.lower())
        print(f'Display Class Name {display_class_name}')
        # Extract description
        desc_match = re.search(r'### Description\n(.*?)(?=\n### Instructions|\Z)', class_content, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""
        print(f'Description {description}')
        
        # Extract instructions
        inst_match = re.search(r'### Instructions\n(.*?)(?=\n### |\Z)', class_content, re.DOTALL)
        instructions = inst_match.group(1).strip() if inst_match else ""
        
        classes[actual_class_name] = {
            'description': description,
            'instructions': instructions,
            'display_name': display_class_name
        }
    
    return {
        'introduction': introduction,
        'classes': classes,
        'class_name_mapping': class_name_mapping
    }

def load_coco_classes(json_path: str) -> List[str]:
    """Load class names from COCO format JSON file."""
    with open(json_path, 'r') as file:
        coco_data = json.load(file)
    
    # Extract class names, excluding the parent category
    classes = []
    for category in coco_data['categories']:
        if category['supercategory'] != 'none':  # Skip parent categories
            classes.append(category['name'])
    
    return classes

def extract_class_labels_from_introduction(introduction: str) -> Dict[str, str]:
    """Extract class labels and their descriptions from the introduction."""
    labels = {}
    
    # Find bullet points or lines that describe classes
    lines = introduction.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('- **') and '**:' in line:
            # Extract class name and description
            match = re.match(r'- \*\*(.+?)\*\*:\s*(.+)', line)
            if match:
                class_name = match.group(1).strip().lower()
                description = match.group(2).strip()
                labels[class_name] = description
    
    return labels

def generate_prompts(dataset_info: Dict, coco_classes: List[str]) -> Dict:
    """Generate prompts for each class and overall."""
    intro_labels = extract_class_labels_from_introduction(dataset_info['introduction'])
    
    output_format = '\n\nThe format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "CLASS_LABEL"}'
    
    prompts = {}
    
    # Generate prompts for each class
    for class_name in coco_classes:
        class_key = class_name.lower()
        
        # Label only prompt (from introduction)
        label_only = ""
        if class_key in intro_labels:
            label_only = f"Identify and locate {class_name} in the image. {intro_labels[class_key]}"
        else:
            label_only = f"Identify and locate {class_name} in the image."
        
        label_only += output_format
        
        # With description prompt (from object classes section)
        with_description = label_only  # Start with label_only as fallback
        if class_key in dataset_info['classes']:
            class_info = dataset_info['classes'][class_key]
            description = class_info['description']
            instructions = class_info['instructions']
            display_name = class_info['display_name']
            
            # Build the full detailed prompt
            with_description = f"Identify and locate {class_name} in the image.\n\n"
            
            # Add the introduction label if available
            if class_key in intro_labels:
                with_description += f"{intro_labels[class_key]}\n\n"
            
            # Add the detailed class information using the display name
            with_description += f"## {display_name}\n"
            with_description += f"### Description\n{description}\n\n"
            with_description += f"### Instructions\n{instructions}"
            with_description += output_format
        
        prompts[class_name] = {
            "label_only": label_only,
            "with_description": with_description
        }
    
    # Generate overall prompts
    all_labels = []
    all_descriptions = []
    
    for class_name in coco_classes:
        class_key = class_name.lower()
        if class_key in intro_labels:
            all_labels.append(f"- **{class_name}**: {intro_labels[class_key]}")
        
        if class_key in dataset_info['classes']:
            class_info = dataset_info['classes'][class_key]
            all_descriptions.append(f"**{class_name}**: {class_info['description']}")
    
    # Overall label only
    overall_label_only = "Identify and locate objects in the image from the following classes:\n"
    overall_label_only += "\n".join(all_labels)
    overall_label_only += output_format
    
    # Overall with description
    overall_with_description = "Identify and locate objects in the image from the following classes:\n\n"
    
    # Add brief intro descriptions
    if all_labels:
        overall_with_description += "\n".join(all_labels) + "\n\n"
    
    # Add detailed class information
    for class_name in coco_classes:
        class_key = class_name.lower()
        if class_key in dataset_info['classes']:
            class_info = dataset_info['classes'][class_key]
            display_name = class_info['display_name']
            overall_with_description += f"## {display_name}\n"
            overall_with_description += f"### Description\n{class_info['description']}\n\n"
            overall_with_description += f"### Instructions\n{class_info['instructions']}\n\n"
    
    overall_with_description += output_format
    
    prompts["overall"] = {
        "label_only": overall_label_only,
        "with_description": overall_with_description
    }
    
    return prompts

def main(text_file_path: str, json_file_path: str, output_file_path: str = None):
    """Main function to parse files and generate prompts."""
    # Parse the dataset file
    dataset_info = parse_dataset_file(text_file_path)
    
    # Load COCO classes
    coco_classes = load_coco_classes(json_file_path)
    
    # Generate prompts
    prompts = generate_prompts(dataset_info, coco_classes)
    
    # Save or return results
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(prompts, file, indent=2, ensure_ascii=False)
        print(f"Prompts saved to {output_file_path}")
    else:
        print(json.dumps(prompts, indent=2, ensure_ascii=False))
    
    return prompts


if __name__ == "__main__":
    import os
    root_dir = "/data3/nperi/rf20vl"
    
    for dataset in os.listdir(root_dir):
        # Example usage
        text_file = "{}/{}/README.dataset.txt".format(root_dir, dataset)  # Your text file path
        json_file = "{}/{}/train/_annotations.coco.json".format(root_dir, dataset)  # Your COCO JSON file path
        output_file = "{}/{}/{}_prompts.json".format(root_dir, dataset, dataset)  # Output file path
        
        # Run the parser
        prompts = main(text_file, json_file, output_file)
