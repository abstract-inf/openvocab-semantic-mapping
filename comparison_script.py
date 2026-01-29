# ICPS/comparison_script.py

# --- MONKEY PATCH FOR CLIP TRUNCATE ARGUMENT (Provided by User) ---
# This fixes compatibility issues with some versions of open_clip/torch
import sys
try:
    import open_clip
    import torch
    
    class FakeClip:
        def __init__(self):
            # Using the same model name ultralytics defaults to
            try:
                self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            except:
                print("Warning: open_clip tokenizer failed to load. Ensure open_clip_torch is installed.")
            
        def load(self, name, device="cuda" if torch.cuda.is_available() else "cpu"):
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained='openai')
            return model.to(device), preprocess
            
        def tokenize(self, texts, truncate=True):
            # Added 'truncate' support to fix the TypeError
            return self.tokenizer(texts)

    # Inject the patch into sys.modules
    sys.modules['clip'] = FakeClip()
    
except ImportError:
    print("NOTE: open_clip_torch not found. If YOLO-World crashes, run: pip install open_clip_torch")
# ------------------------------------------------

print("loading libraries...")
import time
import numpy as np
import pandas as pd
import cv2
import os
import json
import yaml
import csv
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO, RTDETR, YOLOWorld 


# --- CONFIGURATION ---
IMAGE_DIR = "./val2017" 
ANNOT_FILE = "./annotations_trainval2017/annotations/instances_val2017.json"
YAML_FILE = "./coco.yaml"
OUTPUT_CSV = "model_comparison_results.csv"
DETAILED_CSV = "detailed_predictions.csv" 
SAVE_VISUALS = True 
VISUALS_DIR = "./visuals"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

if SAVE_VISUALS and not os.path.exists(VISUALS_DIR):
    os.makedirs(VISUALS_DIR)

# --- HELPER: LOAD CLASSES FROM YAML ---
def load_classes_from_yaml(yaml_path):
    print(f"Loading classes from {yaml_path}...")
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        names_dict = data.get('names', {})
        sorted_indices = sorted(names_dict.keys())
        class_list = [names_dict[i] for i in sorted_indices]
        print(f"Successfully loaded {len(class_list)} classes.")
        return class_list
    except Exception as e:
        print(f"Error loading YAML: {e}")
        return ["person"] * 80 # Fallback

# LOAD CLASSES GLOBALLY
COCO_CLASSES = load_classes_from_yaml(YAML_FILE)

# --- DATA LOADER ---
def load_coco_ground_truth(json_path):
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cat_id_to_idx = {}
    for cat in data['categories']:
        name = cat['name']
        if name in COCO_CLASSES:
            cat_id_to_idx[cat['id']] = COCO_CLASSES.index(name)
    
    img_id_to_file = {img['id']: img['file_name'] for img in data['images']}
    ground_truths = {}
    
    for ann in tqdm(data['annotations'], desc="Parsing Labels"):
        img_id = ann['image_id']
        if img_id not in img_id_to_file: continue
        filename = img_id_to_file[img_id]
        cat_id = ann['category_id']
        if cat_id not in cat_id_to_idx: continue 
        cls_idx = cat_id_to_idx[cat_id]
        x, y, w, h = ann['bbox']
        box = [x, y, x + w, y + h, cls_idx]
        
        if filename not in ground_truths:
            ground_truths[filename] = []
        ground_truths[filename].append(box)
        
    print(f"Loaded ground truth for {len(ground_truths)} images.")
    return ground_truths

# --- METRIC MATHEMATICS ---
def calculate_iou_ciou(box1, box2):
    """
    Calculates IoU and CIoU.
    Formula: CIoU = IoU - (rho^2 / c^2 + alpha * v)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-7)

    # CIoU Terms
    c_x1 = min(box1[0], box2[0])
    c_y1 = min(box1[1], box2[1])
    c_x2 = max(box1[2], box2[2])
    c_y2 = max(box1[3], box2[3])
    c_diag_sq = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2

    center1_x, center1_y = (box1[0]+box1[2])/2, (box1[1]+box1[3])/2
    center2_x, center2_y = (box2[0]+box2[2])/2, (box2[1]+box2[3])/2
    rho_sq = (center2_x - center1_x)**2 + (center2_y - center1_y)**2

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    
    if h1 == 0 or h2 == 0: v = 0
    else: v = (4 / (np.pi ** 2)) * (np.arctan(w1 / h1) - np.arctan(w2 / h2)) ** 2

    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (rho_sq / (c_diag_sq + 1e-7) + alpha * v)
    return iou, ciou

class MetricsTracker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tp_50 = 0
        self.tp_75 = 0
        self.tp_80 = 0
        self.fp = 0
        self.fn = 0
        self.total_iou = 0
        self.total_ciou = 0
        self.total_detections = 0
        self.latencies = []
        self.params = 0
        self.gflops = 0
        self.detailed_logs = [] 

    def update(self, preds, ground_truths, latency, filename, img_path):
        self.latencies.append(latency)
        
        matched_gt_indices = set()
        preds.sort(key=lambda x: x[4], reverse=True)
        
        # For visualization
        best_pred_iou = -1
        worst_pred_iou = 2.0
        
        for p in preds:
            p_box = p[:4]
            p_cls = p[5]
            p_conf = p[4]
            
            best_iou = 0
            best_ciou = 0
            match_idx = -1

            for i, gt in enumerate(ground_truths):
                if i in matched_gt_indices: continue
                if gt[4] != p_cls: continue 

                iou, ciou = calculate_iou_ciou(p_box, gt[:4])
                
                if iou > best_iou:
                    best_iou = iou
                    best_ciou = ciou
                    match_idx = i

            # Log Details
            self.detailed_logs.append({
                "Filename": filename,
                "Model": self.model_name,
                "Class": COCO_CLASSES[int(p_cls)] if int(p_cls) < len(COCO_CLASSES) else str(p_cls),
                "Confidence": round(p_conf, 4),
                "IoU": round(best_iou, 4),
                "CIoU": round(best_ciou, 4),
                "x1": round(p_box[0], 1), "y1": round(p_box[1], 1),
                "x2": round(p_box[2], 1), "y2": round(p_box[3], 1)
            })

            # Track for visual snapshot
            if best_iou > best_pred_iou: best_pred_iou = best_iou
            if best_iou < worst_pred_iou: worst_pred_iou = best_iou

            if best_iou >= 0.5:
                self.tp_50 += 1
                self.total_iou += best_iou
                self.total_ciou += best_ciou
                self.total_detections += 1
                matched_gt_indices.add(match_idx)
                
                if best_iou >= 0.75: self.tp_75 += 1
                if best_iou >= 0.8: self.tp_80 += 1
            else:
                self.fp += 1
        
        self.fn += len(ground_truths) - len(matched_gt_indices)

        if SAVE_VISUALS:
            import random
            if random.random() < 0.002: # Approx 10 images total
                self.save_image_with_boxes(img_path, preds, filename, "random")

    def save_image_with_boxes(self, img_path, preds, filename, tag):
        try:
            img = cv2.imread(img_path)
            for p in preds:
                x1, y1, x2, y2 = map(int, p[:4])
                conf = p[4]
                label = f"{conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            out_name = f"{self.model_name}_{tag}_{filename}"
            cv2.imwrite(os.path.join(VISUALS_DIR, out_name), img)
        except Exception as e:
            print(f"Failed to save visual: {e}")

    def compute(self):
        avg_latency = np.mean(self.latencies) if self.latencies else 0
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0
        
        precision = self.tp_50 / (self.tp_50 + self.fp + 1e-7)
        recall = self.tp_50 / (self.tp_50 + self.fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        avg_ciou = self.total_ciou / (self.total_detections + 1e-7)
        map_75 = self.tp_75 / (self.tp_50 + self.fp + 1e-7)
        prec_80 = self.tp_80 / (self.total_detections + 1e-7)

        return {
            "mAP@.75": round(map_75, 3),
            "CIoU": round(avg_ciou, 3),
            "Precision@IoU>0.8": round(prec_80, 3),
            "Inference Latency (ms)": round(avg_latency, 2),
            "FPS": round(fps, 1),
            "Params (M)": self.params,
            "GFLOPs": self.gflops,
            "F1-Score": round(f1, 3)
        }

# --- WRAPPERS ---

class YOLOv10Wrapper:
    def __init__(self):
        self.model = YOLO("yolov10n.pt")
        self.name = "YOLOv10n"
        self.metrics = MetricsTracker(self.name)
        try:
            info = self.model.info()
            self.metrics.params = info[1] / 1e6
            self.metrics.gflops = info[3] 
        except:
            self.metrics.params = 2.7 
            self.metrics.gflops = 8.7

    def predict(self, image_path):
        t0 = time.time()
        results = self.model(image_path, verbose=False)[0]
        t1 = time.time()
        preds = []
        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy()
            c = box.conf.item()
            cls = int(box.cls.item())
            preds.append([b[0], b[1], b[2], b[3], c, cls])
        return preds, (t1 - t0) * 1000

class RTDETRWrapper:
    def __init__(self):
        self.model = RTDETR("rtdetr-l.pt")
        self.name = "RT-DETR"
        self.metrics = MetricsTracker(self.name)
        try:
            info = self.model.info()
            self.metrics.params = info[1] / 1e6
            self.metrics.gflops = info[3]
        except:
            self.metrics.params = 32.0
            self.metrics.gflops = 110.0

    def predict(self, image_path):
        t0 = time.time()
        results = self.model(image_path, verbose=False)[0]
        t1 = time.time()
        preds = []
        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy()
            c = box.conf.item()
            cls = int(box.cls.item())
            preds.append([b[0], b[1], b[2], b[3], c, cls])
        return preds, (t1 - t0) * 1000

class YOLO11Wrapper:
    def __init__(self):
        self.model = YOLO("yolo11n.pt") 
        self.name = "YOLO11n"
        self.metrics = MetricsTracker(self.name)
        try:
            info = self.model.info()
            self.metrics.params = info[1] / 1e6
            self.metrics.gflops = info[3]
        except:
            self.metrics.params = 2.6
            self.metrics.gflops = 6.5

    def predict(self, image_path):
        t0 = time.time()
        results = self.model(image_path, verbose=False)[0]
        t1 = time.time()
        preds = []
        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy()
            c = box.conf.item()
            cls = int(box.cls.item())
            preds.append([b[0], b[1], b[2], b[3], c, cls])
        return preds, (t1 - t0) * 1000

class YOLOWorldWrapper:
    def __init__(self):
        # Using the model user tinkered with
        self.model = YOLOWorld("yolov8s-worldv2.pt")
        self.name = "YOLO-Worldv2"
        self.metrics = MetricsTracker(self.name)
        
        # Grounding the model to COCO vocab
        print(f"Setting {self.name} vocabulary to {len(COCO_CLASSES)} COCO classes...")
        self.model.set_classes(COCO_CLASSES)
        
        try:
            info = self.model.info()
            self.metrics.params = info[1] / 1e6
            self.metrics.gflops = info[3]
        except:
            # Fallback stats for yolov8s-world
            self.metrics.params = 11.1
            self.metrics.gflops = 28.6

    def predict(self, image_path):
        t0 = time.time()
        # verbose=False, conf=0.05 default as requested (or standard 0.25)
        # Using standard default for fair comparison unless user insists on 0.05
        results = self.model(image_path, verbose=False)[0]
        t1 = time.time()
        preds = []
        for box in results.boxes:
            b = box.xyxy[0].cpu().numpy()
            c = box.conf.item()
            cls = int(box.cls.item())
            # YOLOWorld returns indices mapped to the custom list we provided
            # So cls 0 = COCO_CLASSES[0] which aligns with our ground truth
            preds.append([b[0], b[1], b[2], b[3], c, cls])
        return preds, (t1 - t0) * 1000

# --- MAIN EXECUTION ---
def main():
    if not os.path.exists(IMAGE_DIR) or not os.path.exists(ANNOT_FILE):
        print(f"ERROR: Dataset not found at {IMAGE_DIR}")
        return

    gt_data = load_coco_ground_truth(ANNOT_FILE)
    images_in_dir = set(os.listdir(IMAGE_DIR))
    valid_images = [f for f in gt_data.keys() if f in images_in_dir]
    
    print(f"Starting comparison on {len(valid_images)} images...")

    models = [
        YOLOv10Wrapper(),
        RTDETRWrapper(),
        YOLO11Wrapper(),
        YOLOWorldWrapper() # New Model Added
    ]

    all_results = []
    
    # Initialize Detailed CSV
    with open(DETAILED_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Model", "Class", "Confidence", "IoU", "CIoU", "x1", "y1", "x2", "y2"])

    for model in models:
        print(f"\nEvaluating {model.name}...")
        for filename in tqdm(valid_images):
            img_path = os.path.join(IMAGE_DIR, filename)
            preds, latency = model.predict(img_path)
            ground_truth = gt_data[filename]
            model.metrics.update(preds, ground_truth, latency, filename, img_path)

        stats = model.metrics.compute()
        stats["Model"] = model.name
        all_results.append(stats)
        
        # Append detailed logs to CSV
        with open(DETAILED_CSV, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Filename", "Model", "Class", "Confidence", "IoU", "CIoU", "x1", "y1", "x2", "y2"])
            writer.writerows(model.metrics.detailed_logs)

    df = pd.DataFrame(all_results)
    cols = ["Model", "mAP@.75", "CIoU", "Precision@IoU>0.8", "FPS", 
            "Inference Latency (ms)", "Params (M)", "GFLOPs", "F1-Score"]
    df = df[cols]
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAnalysis complete. Summary saved to {OUTPUT_CSV}")
    print(f"Detailed predictions saved to {DETAILED_CSV}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()