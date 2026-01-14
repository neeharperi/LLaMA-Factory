import os
import cv2
import numpy as np
import albumentations as A
import argparse
import json
import random
import copy
from tqdm import tqdm
from pathlib import Path
import shutil

def get_albumentations_pipeline(width=640, height=640):
    """
    Maps the user's requested list to Albumentations classes.
    Updated for Albumentations >= 1.4.0 compatibility.
    """
    return A.Compose([
        A.RandomResizedCrop(size=(height, width), scale=(0.5, 1.0), p=0.5),
        A.Resize(height=height, width=width, p=0.8),
        A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10, 
            sat_shift_limit=30, 
            val_shift_limit=30, 
            p=0.5
        ),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.1))
    
def apply_mosaic(image, bboxes, category_ids, buffer, output_size=(640, 640)):
    """
    Simulates CachedMosaic by combining the current image with 3 random images from the buffer.
    """
    if len(buffer) < 3:
        return image, bboxes, category_ids

    indices = random.sample(range(len(buffer)), 3)
    mosaic_imgs = [image] + [buffer[i]['image'] for i in indices]
    mosaic_anns = [{'bboxes': bboxes, 'category_ids': category_ids}] + \
                  [{'bboxes': buffer[i]['bboxes'], 'category_ids': buffer[i]['category_ids']} for i in indices]

    h, w = output_size[0], output_size[1]
    c = 3
    mosaic_img = np.full((h * 2, w * 2, c), 114, dtype=np.uint8) # YOLOX grey fill
    
    yc = int(random.uniform(0.5 * h, 1.5 * h))
    xc = int(random.uniform(0.5 * w, 1.5 * w))

    new_bboxes = []
    new_categories = []

    for i, (img_part, ann_part) in enumerate(zip(mosaic_imgs, mosaic_anns)):
        oh, ow, _ = img_part.shape
        
        img_part = cv2.resize(img_part, (w, h))
        h_part, w_part, _ = img_part.shape
        
        scale_x = w / ow
        scale_y = h / oh
        
        if i == 0:  # Top-left
            x1a, y1a, x2a, y2a = max(xc - w_part, 0), max(yc - h_part, 0), xc, yc
            x1b, y1b, x2b, y2b = w_part - (x2a - x1a), h_part - (y2a - y1a), w_part, h_part
        elif i == 1:  # Top-right
            x1a, y1a, x2a, y2a = xc, max(yc - h_part, 0), min(xc + w_part, w * 2), yc
            x1b, y1b, x2b, y2b = 0, h_part - (y2a - y1a), min(w_part, x2a - x1a), h_part
        elif i == 2:  # Bottom-left
            x1a, y1a, x2a, y2a = max(xc - w_part, 0), yc, xc, min(yc + h_part, h * 2)
            x1b, y1b, x2b, y2b = w_part - (x2a - x1a), 0, w_part, min(h_part, y2a - y1a)
        elif i == 3:  # Bottom-right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w_part, w * 2), min(yc + h_part, h * 2)
            x1b, y1b, x2b, y2b = 0, 0, min(w_part, x2a - x1a), min(h_part, y2a - y1a)

        mosaic_img[y1a:y2a, x1a:x2a] = img_part[y1b:y2b, x1b:x2b]
        pad_w = x1a - x1b
        pad_h = y1a - y1b

        current_bboxes = ann_part['bboxes']
        current_cats = ann_part['category_ids']
        
        for bbox, cat in zip(current_bboxes, current_cats):
            bx, by, bw, bh = bbox
            
            sbx = bx * scale_x
            sby = by * scale_y
            sbw = bw * scale_x
            sbh = bh * scale_y
            
            nbx = sbx + pad_w
            nby = sby + pad_h
            nbw = sbw 
            nbh = sbh 
            
            new_bboxes.append([nbx, nby, nbw, nbh])
            new_categories.append(cat)

    final_img = mosaic_img 
    
    valid_bboxes = []
    valid_cats = []
    
    clip_h, clip_w = final_img.shape[:2]
    
    for bbox, cat in zip(new_bboxes, new_categories):
        x, y, w_b, h_b = bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(clip_w, x + w_b), min(clip_h, y + h_b)
        
        if x2 > x1 and y2 > y1:
            valid_bboxes.append([x1, y1, x2-x1, y2-y1])
            valid_cats.append(cat)
            
    return final_img, valid_bboxes, valid_cats

def apply_mixup(image, bboxes, category_ids, buffer, output_size=(640, 640)):    
    idx = random.randint(0, len(buffer) - 1)
    mix_img = buffer[idx]['image']
    mix_bboxes = buffer[idx]['bboxes']
    mix_cats = buffer[idx]['category_ids']
    
    # Resize both to target
    img1 = cv2.resize(image, output_size)
    img2 = cv2.resize(mix_img, output_size)
    
    # Blend
    alpha = random.uniform(0.5, 0.8) # Mixup ratio
    mixed_img = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    
    final_bboxes = []
    final_cats = []
    
    # Scale function
    def scale_boxes(boxes, cats, orig_shape, target_shape):
        sh, sw = orig_shape[:2]
        th, tw = target_shape[:2]
        res = []
        for b in boxes:
            res.append([b[0]*(tw/sw), b[1]*(th/sh), b[2]*(tw/sw), b[3]*(th/sh)])
        return res
        
    final_bboxes.extend(scale_boxes(bboxes, category_ids, image.shape, output_size))
    final_cats.extend(category_ids)
    
    final_bboxes.extend(scale_boxes(mix_bboxes, mix_cats, mix_img.shape, output_size))
    final_cats.extend(mix_cats)
    
    return mixed_img, final_bboxes, final_cats

def get_annotations(coco_json):
    res = {}
    for annotation in coco_json["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in res:
            res[image_id] = []
        res[image_id].append(annotation)
    return res 

def visualize_dataset(json_path, image_dir, output_vis_dir, num_samples=10):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_vis_dir, exist_ok=True)
    
    cat_map = {cat['id']: cat['name'] for cat in data.get('categories', [])}

    img_to_anns = {}
    for ann in data['annotations']:
        iid = ann['image_id']
        if iid not in img_to_anns: img_to_anns[iid] = []
        img_to_anns[iid].append(ann)
        
    all_images = data['images']
    if len(all_images) > num_samples:
        images = random.sample(all_images, num_samples)
    else:
        images = all_images
    
    for img_info in images:
        file_name = img_info['file_name']
        img_path = os.path.join(image_dir, file_name)
            
        image = cv2.imread(img_path)        
        anns = img_to_anns.get(img_info['id'], [])
        
        for ann in anns:
            x, y, w, h = map(int, ann['bbox'])
            cat_id = ann['category_id']
            
            class_name = cat_map.get(cat_id, str(cat_id))            
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)            
            label = f"{class_name}"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x, y - 20), (x + w_text, y), (0, 255, 0), -1)
            cv2.putText(image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        cv2.imwrite(os.path.join(output_vis_dir, f"vis_{file_name}"), image)

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_epochs = args.num_epochs
    no_aug_epochs = args.no_aug_epochs
    target_size = (640, 640)
    
    os.makedirs(output_dir, exist_ok=True)
    
    transform_pipeline = get_albumentations_pipeline(width=target_size[1], height=target_size[0])
    
    for dataset in tqdm(os.listdir(input_dir), desc="Datasets"):
        shutil.copytree(os.path.join(input_dir, dataset, "valid"), os.path.join(output_dir, dataset, "valid"), dirs_exist_ok=True)
        shutil.copytree(os.path.join(input_dir, dataset, "test"), os.path.join(output_dir, dataset, "test"), dirs_exist_ok=True)
        shutil.copyfile(os.path.join(input_dir, dataset, "README.dataset.txt"), os.path.join(output_dir, dataset, "README.dataset.txt"))
            
        dataset_path = os.path.join(input_dir, dataset)
        if not os.path.isdir(dataset_path): continue

        train_dir = os.path.join(dataset_path, "train")
        anno_file = os.path.join(train_dir, "_annotations.coco.json")
        
        os.makedirs(os.path.join(output_dir, dataset, "train"), exist_ok=True)
        
        with open(anno_file, 'r') as f:
            train_json = json.load(f)

        augmented_json = {
            "info": train_json.get("info", {}),
            "licenses": train_json.get("licenses", []),
            "images": [],
            "annotations": [],
            "categories": train_json["categories"]
        }
        
        annotations_dict = get_annotations(train_json)

        image_buffer = []
        MAX_BUFFER_SIZE = 20
        
        new_img_id = 0
        new_ann_id = 0
        
        for epoch in range(num_epochs):            
            for image_info in train_json["images"]:
                image_id = image_info["id"]
                image_filename = image_info["file_name"]
                image_path = os.path.join(train_dir, image_filename)
                
                image = cv2.imread(image_path)
                
                anns = annotations_dict.get(image_id, [])
                bboxes = [a['bbox'] for a in anns]
                category_ids = [a['category_id'] for a in anns]

                image_buffer.append({'image': image, 'bboxes': bboxes, 'category_ids': category_ids})
                if len(image_buffer) > MAX_BUFFER_SIZE:
                    image_buffer.pop(0)
                    
                final_image = image.copy()
                final_bboxes = copy.deepcopy(bboxes)
                final_cats = copy.deepcopy(category_ids)
                
                if epoch > no_aug_epochs:
                    # We need to fix mosaic and mixup for few-shot detection
                    #if random.random() < 0.5 and len(image_buffer) > 4:
                    #    final_image, final_bboxes, final_cats = apply_mosaic(
                    #        final_image, final_bboxes, final_cats, image_buffer, output_size=target_size
                    #        )
                    #elif random.random() < 0.5 and len(image_buffer) > 1:
                    #    final_image, final_bboxes, final_cats = apply_mixup(
                    #        final_image, final_bboxes, final_cats, image_buffer, output_size=target_size
                    #        )
                        
                    transformed = transform_pipeline(
                        image=final_image, 
                        bboxes=final_bboxes, 
                        category_ids=final_cats
                    )
                    final_image = transformed['image']
                    final_bboxes = transformed['bboxes']
                    final_cats = transformed['category_ids']
                  
                new_filename = f"{os.path.splitext(image_filename)[0]}_ep{epoch}.jpg"
                output_path = os.path.join(output_dir, dataset, "train", new_filename)
                cv2.imwrite(output_path, final_image)
                
                augmented_json["images"].append({
                    "id": new_img_id,
                    "file_name": new_filename,
                    "height": final_image.shape[0],
                    "width": final_image.shape[1]
                })
                
                for bbox, cat_id in zip(final_bboxes, final_cats):
                    if isinstance(bbox, np.ndarray):
                        bbox = bbox.tolist()
                    
                    bbox = [float(x) for x in bbox]
                    area = bbox[2] * bbox[3]
                    
                    augmented_json["annotations"].append({
                        "id": int(new_ann_id),           
                        "image_id": int(new_img_id),     
                        "category_id": int(cat_id),      
                        "bbox": bbox,                    
                        "area": float(area),             
                        "iscrowd": 0
                    })
                    new_ann_id += 1
                
                new_img_id += 1
        
        output_json_path = os.path.join(output_dir, dataset, "train", "_annotations.coco.json")
        with open(output_json_path, "w") as f:
            json.dump(augmented_json, f)
        
        if args.visualize:
            visualize_dataset(output_json_path, os.path.join(output_dir, dataset, "train"), 
                              os.path.join(output_dir, dataset, "visualize"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Augmentation for Image Dataset")
    parser.add_argument("--input_dir", default="/home/nperi/Workspace/LLaMA-Factory/data/rf20vl")
    parser.add_argument("--output_dir", default="/home/nperi/Workspace/LLaMA-Factory/data/rf20vl_augmented")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--no_aug_epochs", type=int, default=1)
    parser.add_argument("--visualize", action='store_true', help="Visualize output samples")
    
    args = parser.parse_args()
    main(args)
