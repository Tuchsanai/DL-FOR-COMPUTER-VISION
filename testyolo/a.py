# %%
# =============================================================================
# YOLO26 Advanced Lab: Complete Guide to Real-Time Object Detection
# =============================================================================
# Course: Deep Learning / Computer Vision / MLOps
# Author: [Instructor Name]
# Version: 1.0 (2026)
# 
# เนื้อหา Lab นี้ครอบคลุมทุก Features และ Techniques ของ YOLO26
# ออกแบบมาเพื่อให้นักศึกษาเข้าใจตั้งแต่พื้นฐานจนถึงระดับ Advanced
# =============================================================================

# %% [markdown]
# # 🚀 YOLO26 Advanced Lab
#
# ## Learning Objectives (วัตถุประสงค์การเรียนรู้)
#
# หลังจากทำ Lab นี้เสร็จสิ้น นักศึกษาจะสามารถ:
#
# 1. **เข้าใจสถาปัตยกรรม YOLO26** - NMS-Free, DFL Removal, ProgLoss, STAL, MuSGD
# 2. **ใช้งาน YOLO26 สำหรับทุก Tasks** - Detection, Segmentation, Classification, Pose, OBB
# 3. **Train โมเดล Custom Dataset** - ตั้งแต่เตรียมข้อมูลจนถึง Fine-tuning
# 4. **Export และ Deploy** - ONNX, TensorRT, CoreML, TFLite
# 5. **ใช้ Open-Vocabulary Detection** - YOLOE-26 Text/Visual Prompts
# 6. **Object Tracking** - Real-time Multi-Object Tracking
# 7. **Optimization และ Quantization** - INT8, FP16 สำหรับ Edge Devices
#
# ## Prerequisites (ความรู้พื้นฐาน)
# - Python Programming
# - Deep Learning Fundamentals
# - Computer Vision Basics
# - PyTorch Basics

# %% [markdown]
# ---
# # Part 1: Introduction to YOLO26 Architecture
# ---
#
# ## 1.1 ความแตกต่างของ YOLO26 จาก Version ก่อนหน้า
#
# | Feature | Previous YOLO | YOLO26 |
# |---------|--------------|--------|
# | Post-processing | NMS Required | **NMS-Free (End-to-End)** |
# | Box Regression | DFL (Distribution Focal Loss) | **DFL Removed** |
# | Loss Balancing | Fixed | **ProgLoss (Progressive)** |
# | Small Objects | Standard | **STAL (Small-Target-Aware)** |
# | Optimizer | SGD/Adam | **MuSGD (Hybrid)** |
# | CPU Speed | Baseline | **43% Faster** |
#
# ## 1.2 Three Core Principles of YOLO26
#
# 1. **Simplicity** - Native end-to-end, ไม่ต้องมี NMS post-processing
# 2. **Deployment Efficiency** - Export cleanly to ONNX, TensorRT, CoreML, TFLite
# 3. **Training Innovation** - MuSGD optimizer จาก LLM training techniques

# %% [markdown]
# ---
# # Part 2: Environment Setup
# ---

# %%
# =============================================================================
# 2.1 Installation
# =============================================================================

import IPython
import sys

def clean_notebook():
    IPython.display.clear_output(wait=True)
    print("Notebook cleaned.")

# Run the installation commands
# ติดตั้ง Ultralytics package
# !uv pip install ultralytics>=8.4.0 --upgrade

# ติดตั้ง dependencies เพิ่มเติม
# #!uv pip install supervision roboflow opencv-python-headless
# Clean up the notebook

clean_notebook()





# %%
# =============================================================================
# 2.2 Import Libraries
# =============================================================================

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Ultralytics YOLO
from ultralytics import YOLO

# แสดง version และ GPU info
print("=" * 60)
print("YOLO26 Advanced Lab - Environment Check")
print("=" * 60)
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 60)

# %%
# =============================================================================
# 2.3 Helper Functions
# =============================================================================

def display_results(results, title="Detection Results"):
    """แสดงผลลัพธ์การ detect พร้อม annotations"""
    for r in results:
        # Plot results
        im_array = r.plot()
        # Convert BGR to RGB
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(im_rgb)
        plt.title(title)
        plt.axis('off')
        plt.show()

def print_detection_info(results):
    """แสดงข้อมูลรายละเอียดของ detection results"""
    for r in results:
        boxes = r.boxes
        print(f"Number of detections: {len(boxes)}")
        print("-" * 40)
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"[{i+1}] Class: {cls_name:15s} | Conf: {conf:.3f} | Box: {xyxy}")

# %% [markdown]
# ---
# # Part 3: YOLO26 Model Variants
# ---
#
# ## 3.1 Available Model Sizes
#
# | Model | Parameters | mAP@50-95 | CPU Speed | GPU Speed | Use Case |
# |-------|------------|-----------|-----------|-----------|----------|
# | yolo26n | ~3M | 39.8% | 38.9ms | 1.5ms | Mobile, IoT |
# | yolo26s | ~11M | 47.2% | 87.2ms | 2.1ms | Edge devices |
# | yolo26m | ~26M | 51.5% | 220ms | 4.7ms | Balanced |
# | yolo26l | ~49M | 53.4% | 286ms | 6.1ms | High accuracy |
# | yolo26x | ~97M | 55.2% | 450ms | 9.8ms | Maximum accuracy |

# %%
# =============================================================================
# 3.2 Load Different Model Variants
# =============================================================================

# โหลด YOLO26 Nano (เหมาะสำหรับ edge devices)
model_nano = YOLO("yolo26n.pt")
print("✅ Loaded YOLO26 Nano")

# โหลด YOLO26 Small (balanced)
model_small = YOLO("yolo26s.pt")
print("✅ Loaded YOLO26 Small")

# สำหรับ training จาก scratch ใช้ .yaml
# model_from_scratch = YOLO("yolo26n.yaml")

# %%
# =============================================================================
# 3.3 Model Information
# =============================================================================

# แสดงข้อมูล model
model = YOLO("yolo26n.pt")
print("\n" + "=" * 60)
print("Model Information")
print("=" * 60)
print(f"Model Type: {model.type}")
print(f"Task: {model.task}")
print(f"Number of Classes: {len(model.names)}")
print(f"Class Names: {list(model.names.values())[:10]}...")  # แสดง 10 classes แรก
print("=" * 60)

# %% [markdown]
# ---
# # Part 4: Object Detection (การตรวจจับวัตถุ)
# ---
#
# ## 4.1 Basic Inference
#
# YOLO26 รองรับ input หลายรูปแบบ:
# - Single image (path, URL, numpy array, PIL Image)
# - Multiple images (list)
# - Video file
# - Webcam stream

# %%
# =============================================================================
# 4.1 Basic Detection - Single Image
# =============================================================================

# Load model
model = YOLO("yolo26n.pt")

# ทำ inference บน image
# ใช้ sample image จาก Ultralytics
results = model("https://ultralytics.com/images/bus.jpg")

# แสดงผลลัพธ์
print("\n📊 Detection Results:")
print_detection_info(results)

# Visualize
display_results(results, "YOLO26 Object Detection")

# %%
# =============================================================================
# 4.2 Detection with Custom Parameters
# =============================================================================

# กำหนด parameters เพิ่มเติม
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.25,           # Confidence threshold (default: 0.25)
    iou=0.45,            # IoU threshold สำหรับ NMS (ถ้าใช้)
    imgsz=640,           # Input image size
    max_det=300,         # Maximum detections per image
    classes=[0, 2],      # Filter เฉพาะ class 0 (person) และ 2 (car)
    save=True,           # Save results
    save_txt=True,       # Save results as .txt
    save_conf=True,      # Include confidence in txt
    project="runs/detect",
    name="custom_inference"
)

print(f"\n✅ Results saved to: {results[0].save_dir}")

# %%
# =============================================================================
# 4.3 Batch Inference - Multiple Images
# =============================================================================

# รายการ images
image_urls = [
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg"
]

# Batch inference
results = model(image_urls)

# แสดงผลลัพธ์แต่ละ image
for i, r in enumerate(results):
    print(f"\n🖼️ Image {i+1}:")
    print(f"   Detections: {len(r.boxes)}")
    display_results([r], f"Image {i+1} Results")

# %%
# =============================================================================
# 4.4 Video Inference
# =============================================================================

# สำหรับ video inference
# model.predict(
#     source="path/to/video.mp4",
#     stream=True,  # ใช้ streaming mode สำหรับ video ยาว
#     save=True,
#     show=True  # แสดง real-time
# )

# สำหรับ webcam
# model.predict(source=0, show=True)

print("💡 Video inference code ready - uncomment to use")

# %%
# =============================================================================
# 4.5 Access Detection Results Programmatically
# =============================================================================

# ทำ inference
results = model("https://ultralytics.com/images/bus.jpg")

# เข้าถึง results
for r in results:
    # Bounding boxes
    boxes = r.boxes
    
    print("\n📦 Bounding Box Details:")
    print("-" * 60)
    
    for box in boxes:
        # Coordinates
        xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        xywh = box.xywh[0].cpu().numpy()  # [x_center, y_center, width, height]
        xyxyn = box.xyxyn[0].cpu().numpy()  # normalized xyxy
        
        # Class and confidence
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = r.names[cls]
        
        print(f"Class: {name}")
        print(f"  Confidence: {conf:.4f}")
        print(f"  Box (xyxy): {xyxy}")
        print(f"  Box (xywh): {xywh}")
        print(f"  Box (normalized): {xyxyn}")
        print()

# %% [markdown]
# ---
# # Part 5: Instance Segmentation (การแบ่งส่วนภาพ)
# ---
#
# ## 5.1 Segmentation Overview
#
# Instance Segmentation ทำได้ทั้ง:
# - Object Detection (bounding boxes)
# - Pixel-level Masks (contours ของแต่ละ object)

# %%
# =============================================================================
# 5.1 Load Segmentation Model
# =============================================================================

# โหลด segmentation model
seg_model = YOLO("yolo26n-seg.pt")
print("✅ Loaded YOLO26 Segmentation Model")

# %%
# =============================================================================
# 5.2 Basic Segmentation Inference
# =============================================================================

# ทำ segmentation
seg_results = seg_model("https://ultralytics.com/images/bus.jpg")

# แสดงผลลัพธ์
display_results(seg_results, "YOLO26 Instance Segmentation")

# %%
# =============================================================================
# 5.3 Access Segmentation Masks
# =============================================================================

for r in seg_results:
    if r.masks is not None:
        masks = r.masks
        
        print("\n🎭 Segmentation Masks:")
        print("-" * 60)
        print(f"Number of masks: {len(masks)}")
        print(f"Mask shape (original): {masks.orig_shape}")
        print(f"Mask data shape: {masks.data.shape}")
        
        # เข้าถึง mask แต่ละ instance
        for i, mask in enumerate(masks.data):
            mask_np = mask.cpu().numpy()
            print(f"\nMask {i+1}:")
            print(f"  Shape: {mask_np.shape}")
            print(f"  Non-zero pixels: {np.count_nonzero(mask_np)}")
            print(f"  Coverage: {np.count_nonzero(mask_np) / mask_np.size * 100:.2f}%")

# %%
# =============================================================================
# 5.4 Extract Individual Masks
# =============================================================================

# สร้าง visualization ของ masks แต่ละ instance
for r in seg_results:
    if r.masks is not None:
        fig, axes = plt.subplots(1, min(4, len(r.masks)), figsize=(16, 4))
        if len(r.masks) == 1:
            axes = [axes]
        
        for i, (mask, ax) in enumerate(zip(r.masks.data[:4], axes)):
            ax.imshow(mask.cpu().numpy(), cmap='viridis')
            cls_id = int(r.boxes.cls[i])
            ax.set_title(f"{r.names[cls_id]}")
            ax.axis('off')
        
        plt.suptitle("Individual Instance Masks")
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ---
# # Part 6: Image Classification
# ---
#
# ## 6.1 Classification Task
#
# YOLO26 Classification ทำการจำแนกทั้งภาพเป็น 1 class
# โมเดล pretrained บน ImageNet มี 1000 classes

# %%
# =============================================================================
# 6.1 Load Classification Model
# =============================================================================

# โหลด classification model
cls_model = YOLO("yolo26n-cls.pt")
print("✅ Loaded YOLO26 Classification Model")

# %%
# =============================================================================
# 6.2 Classification Inference
# =============================================================================

# ทำ classification
cls_results = cls_model("https://ultralytics.com/images/bus.jpg")

# แสดงผลลัพธ์
for r in cls_results:
    probs = r.probs
    
    print("\n📊 Classification Results:")
    print("-" * 60)
    
    # Top 5 predictions
    top5_indices = probs.top5
    top5_confs = probs.top5conf
    
    print("Top 5 Predictions:")
    for i, (idx, conf) in enumerate(zip(top5_indices, top5_confs)):
        class_name = r.names[idx]
        print(f"  {i+1}. {class_name}: {float(conf):.4f}")

# %% [markdown]
# ---
# # Part 7: Pose Estimation (การประมาณท่าทาง)
# ---
#
# ## 7.1 Pose Estimation Overview
#
# YOLO26-Pose ตรวจจับ 17 keypoints ของร่างกายมนุษย์:
# - Nose, Eyes, Ears
# - Shoulders, Elbows, Wrists
# - Hips, Knees, Ankles

# %%
# =============================================================================
# 7.1 Load Pose Model
# =============================================================================

# โหลด pose estimation model
pose_model = YOLO("yolo26n-pose.pt")
print("✅ Loaded YOLO26 Pose Model")

# %%
# =============================================================================
# 7.2 Pose Estimation Inference
# =============================================================================

# ทำ pose estimation (ใช้ image ที่มีคน)
pose_results = pose_model("https://ultralytics.com/images/zidane.jpg")

# แสดงผลลัพธ์
display_results(pose_results, "YOLO26 Pose Estimation")

# %%
# =============================================================================
# 7.3 Access Keypoints Data
# =============================================================================

# Keypoint mapping
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

for r in pose_results:
    if r.keypoints is not None:
        keypoints = r.keypoints
        
        print("\n🦴 Keypoints Data:")
        print("-" * 60)
        print(f"Number of persons detected: {len(keypoints)}")
        
        for person_idx, kpts in enumerate(keypoints.data):
            print(f"\nPerson {person_idx + 1}:")
            kpts_np = kpts.cpu().numpy()
            
            for kp_idx, kp in enumerate(kpts_np):
                x, y, conf = kp
                if conf > 0.5:  # แสดงเฉพาะ keypoints ที่มี confidence สูง
                    print(f"  {KEYPOINT_NAMES[kp_idx]:15s}: ({x:6.1f}, {y:6.1f}) conf={conf:.3f}")

# %% [markdown]
# ---
# # Part 8: Oriented Bounding Boxes (OBB)
# ---
#
# ## 8.1 OBB Overview
#
# OBB ตรวจจับวัตถุพร้อม rotation angle
# เหมาะสำหรับ:
# - Aerial/Drone imagery
# - Satellite imagery
# - Document analysis

# %%
# =============================================================================
# 8.1 Load OBB Model
# =============================================================================

# โหลด OBB model (trained on DOTA dataset)
obb_model = YOLO("yolo26n-obb.pt")
print("✅ Loaded YOLO26 OBB Model")

# %%
# =============================================================================
# 8.2 OBB Inference
# =============================================================================

# OBB inference (ต้องใช้ aerial/satellite image)
# obb_results = obb_model("path/to/aerial_image.jpg")

# สำหรับ demo ใช้ sample
print("💡 OBB model loaded - use with aerial/satellite imagery")
print("   Example: obb_model.predict('aerial_image.jpg')")

# เข้าถึง OBB data
# for r in obb_results:
#     obbs = r.obb
#     for obb in obbs:
#         # xywhr format: center_x, center_y, width, height, rotation
#         xywhr = obb.xywhr[0].cpu().numpy()
#         print(f"OBB: center=({xywhr[0]:.1f}, {xywhr[1]:.1f}), size=({xywhr[2]:.1f}, {xywhr[3]:.1f}), angle={xywhr[4]:.2f}rad")

# %% [markdown]
# ---
# # Part 9: Object Tracking
# ---
#
# ## 9.1 Multi-Object Tracking (MOT)
#
# YOLO26 รองรับ tracking algorithms:
# - BoT-SORT (default)
# - ByteTrack

# %%
# =============================================================================
# 9.1 Object Tracking on Video
# =============================================================================

# Load model
model = YOLO("yolo26n.pt")

# Tracking บน video
# track_results = model.track(
#     source="path/to/video.mp4",
#     tracker="botsort.yaml",  # หรือ "bytetrack.yaml"
#     persist=True,  # เก็บ track IDs ระหว่าง frames
#     conf=0.3,
#     iou=0.5,
#     show=True,
#     save=True
# )

print("💡 Tracking code ready - uncomment to use with video")

# %%
# =============================================================================
# 9.2 Streaming Tracking with Custom Processing
# =============================================================================

def process_tracking_frame(results):
    """Process each tracking frame"""
    for r in results:
        boxes = r.boxes
        if boxes.id is not None:  # มี track IDs
            track_ids = boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = r.names[cls]
                
                print(f"Track ID: {track_id} | Class: {name} | Conf: {conf:.3f}")

# Streaming tracking
# for results in model.track(source="video.mp4", stream=True, persist=True):
#     process_tracking_frame(results)

print("💡 Custom tracking processing ready")

# %% [markdown]
# ---
# # Part 10: YOLOE-26 Open-Vocabulary Detection
# ---
#
# ## 10.1 Open-Vocabulary Overview
#
# YOLOE-26 ทำ zero-shot detection โดยใช้:
# - **Text Prompts** - กำหนด class ด้วยข้อความ
# - **Visual Prompts** - ให้ตัวอย่าง visual ของ class
# - **Prompt-Free** - ใช้ built-in vocabulary (4,585 classes)

# %%
# =============================================================================
# 10.1 Text Prompts Detection
# =============================================================================

# Load YOLOE-26 model
# yoloe_model = YOLO("yoloe-26l-seg.pt")

# กำหนด classes ที่ต้องการ detect ด้วย text
# names = ["person", "bus", "traffic light", "backpack"]
# yoloe_model.set_classes(names, yoloe_model.get_text_pe(names))

# ทำ detection
# results = yoloe_model.predict("https://ultralytics.com/images/bus.jpg")

print("💡 YOLOE-26 Text Prompts Example:")
print("""
from ultralytics import YOLO

# Load model
model = YOLO("yoloe-26l-seg.pt")

# Set custom classes
names = ["person", "bus", "traffic light"]
model.set_classes(names, model.get_text_pe(names))

# Detect
results = model.predict("image.jpg")
""")

# %%
# =============================================================================
# 10.2 Visual Prompts Detection
# =============================================================================

print("💡 YOLOE-26 Visual Prompts Example:")
print("""
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# Load model
model = YOLO("yoloe-26l-seg.pt")

# Define visual prompts (bounding boxes as examples)
# Format: [[x1, y1, x2, y2], ...]
visual_prompts = {
    0: [[100, 100, 200, 200]],  # Class 0 examples
    1: [[300, 100, 400, 200]]   # Class 1 examples
}

# Predict with visual prompts
results = model.predict(
    "image.jpg",
    predictor=YOLOEVPSegPredictor,
    prompts=visual_prompts
)
""")

# %%
# =============================================================================
# 10.3 Prompt-Free Detection
# =============================================================================

print("💡 YOLOE-26 Prompt-Free Example:")
print("""
from ultralytics import YOLO

# Load prompt-free model (4,585 classes)
model = YOLO("yoloe-26l-seg-pf.pt")

# Detect without any prompts
results = model.predict("image.jpg")
results[0].show()
""")

# %% [markdown]
# ---
# # Part 11: Training on Custom Dataset
# ---
#
# ## 11.1 Dataset Preparation
#
# ### YOLO Format Structure:
# ```
# dataset/
# ├── train/
# │   ├── images/
# │   │   ├── img1.jpg
# │   │   └── img2.jpg
# │   └── labels/
# │       ├── img1.txt
# │       └── img2.txt
# ├── val/
# │   ├── images/
# │   └── labels/
# └── data.yaml
# ```
#
# ### Label Format (per line):
# ```
# class_id x_center y_center width height
# ```
# (ค่าทั้งหมด normalized 0-1)

# %%
# =============================================================================
# 11.1 Create Sample Dataset Structure
# =============================================================================

def create_dataset_yaml(path, classes, train_path, val_path, test_path=None):
    """สร้างไฟล์ data.yaml สำหรับ training"""
    yaml_content = f"""# YOLO26 Dataset Configuration
# Created for training custom model

path: {path}  # dataset root dir
train: {train_path}  # train images (relative to 'path')
val: {val_path}  # val images (relative to 'path')
"""
    if test_path:
        yaml_content += f"test: {test_path}  # test images (optional)\n"
    
    yaml_content += f"\n# Classes\nnames:\n"
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\n"
    
    return yaml_content

# Example dataset configuration
example_yaml = create_dataset_yaml(
    path="/path/to/dataset",
    classes=["person", "car", "motorcycle", "bus", "truck"],
    train_path="train/images",
    val_path="val/images",
    test_path="test/images"
)

print("📄 Example data.yaml:")
print("-" * 40)
print(example_yaml)

# %%
# =============================================================================
# 11.2 Training from Pretrained Model (Recommended)
# =============================================================================

# โหลด pretrained model
model = YOLO("yolo26n.pt")

# Training configuration
training_config = {
    "data": "path/to/data.yaml",  # Path to dataset yaml
    "epochs": 100,                # Number of epochs
    "imgsz": 640,                 # Input image size
    "batch": 16,                  # Batch size (use -1 for autobatch)
    "patience": 50,               # Early stopping patience
    "save": True,                 # Save checkpoints
    "save_period": 10,            # Save every N epochs
    "cache": True,                # Cache images for faster training
    "device": 0,                  # GPU device (0, 1, 2... or 'cpu')
    "workers": 8,                 # DataLoader workers
    "project": "runs/train",      # Save directory
    "name": "yolo26_custom",      # Experiment name
    "exist_ok": False,            # Overwrite existing
    "pretrained": True,           # Use pretrained weights
    "optimizer": "auto",          # auto uses MuSGD
    "verbose": True,              # Verbose output
    "seed": 42,                   # Random seed
    "deterministic": True,        # Deterministic training
    "single_cls": False,          # Single class training
    "rect": False,                # Rectangular training
    "cos_lr": False,              # Cosine LR scheduler
    "close_mosaic": 10,           # Disable mosaic last N epochs
    "resume": False,              # Resume from checkpoint
    "amp": True,                  # Automatic Mixed Precision
    "fraction": 1.0,              # Dataset fraction to use
    "profile": False,             # Profile ONNX/TensorRT
    "freeze": None,               # Freeze layers (e.g., freeze=10)
    "lr0": 0.01,                  # Initial learning rate
    "lrf": 0.01,                  # Final learning rate factor
    "momentum": 0.937,            # SGD momentum
    "weight_decay": 0.0005,       # Weight decay
    "warmup_epochs": 3.0,         # Warmup epochs
    "warmup_momentum": 0.8,       # Warmup momentum
    "warmup_bias_lr": 0.1,        # Warmup bias lr
    "box": 7.5,                   # Box loss gain
    "cls": 0.5,                   # Class loss gain
    "dfl": 0.0,                   # DFL loss gain (0 in YOLO26)
    "pose": 12.0,                 # Pose loss gain
    "kobj": 1.0,                  # Keypoint obj loss gain
    "label_smoothing": 0.0,       # Label smoothing
    "nbs": 64,                    # Nominal batch size
    "overlap_mask": True,         # Overlapping masks
    "mask_ratio": 4,              # Mask downsample ratio
    "dropout": 0.0,               # Dropout
    "val": True,                  # Validate during training
}

# Start training (uncomment to run)
# results = model.train(**training_config)

print("💡 Training configuration ready")
print("   Uncomment model.train() to start training")

# %%
# =============================================================================
# 11.3 Training from Scratch
# =============================================================================

# สร้าง model จาก YAML (training from scratch)
# model = YOLO("yolo26n.yaml")
# results = model.train(data="data.yaml", epochs=300)

print("💡 Training from scratch:")
print('   model = YOLO("yolo26n.yaml")')
print('   model.train(data="data.yaml", epochs=300)')

# %%
# =============================================================================
# 11.4 Resume Training
# =============================================================================

# Resume จาก last checkpoint
# model = YOLO("runs/train/yolo26_custom/weights/last.pt")
# results = model.train(resume=True)

print("💡 Resume training:")
print('   model = YOLO("path/to/last.pt")')
print('   model.train(resume=True)')

# %%
# =============================================================================
# 11.5 Data Augmentation Configuration
# =============================================================================

# Augmentation parameters (default values)
augmentation_config = {
    "hsv_h": 0.015,       # HSV-Hue augmentation
    "hsv_s": 0.7,         # HSV-Saturation augmentation
    "hsv_v": 0.4,         # HSV-Value augmentation
    "degrees": 0.0,       # Rotation (+/- deg)
    "translate": 0.1,     # Translation (+/- fraction)
    "scale": 0.5,         # Scale (+/- gain)
    "shear": 0.0,         # Shear (+/- deg)
    "perspective": 0.0,   # Perspective (+/- fraction)
    "flipud": 0.0,        # Flip up-down probability
    "fliplr": 0.5,        # Flip left-right probability
    "bgr": 0.0,           # BGR probability
    "mosaic": 1.0,        # Mosaic augmentation probability
    "mixup": 0.0,         # Mixup augmentation probability
    "copy_paste": 0.0,    # Copy-paste augmentation probability
    "copy_paste_mode": "flip",
    "auto_augment": "randaugment",  # Auto augmentation policy
    "erasing": 0.4,       # Random erasing probability
    "crop_fraction": 1.0, # Image crop fraction
}

print("📊 Data Augmentation Parameters:")
print("-" * 40)
for key, value in augmentation_config.items():
    print(f"  {key}: {value}")

# %% [markdown]
# ---
# # Part 12: Model Validation
# ---

# %%
# =============================================================================
# 12.1 Validate Model Performance
# =============================================================================

# Load trained model
model = YOLO("yolo26n.pt")

# Validate on COCO8 (small test dataset)
metrics = model.val(
    data="coco8.yaml",    # Dataset
    imgsz=640,            # Image size
    batch=16,             # Batch size
    conf=0.001,           # Confidence threshold
    iou=0.6,              # IoU threshold
    max_det=300,          # Max detections
    half=False,           # FP16 inference
    device=0,             # Device
    split="val",          # Dataset split
    save_json=False,      # Save results to JSON
    save_hybrid=False,    # Save hybrid labels
    plots=True,           # Save validation plots
    rect=False,           # Rectangular validation
    verbose=True          # Verbose output
)

# %%
# =============================================================================
# 12.2 Access Validation Metrics
# =============================================================================

print("\n📊 Validation Metrics:")
print("=" * 60)
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print("=" * 60)

# Per-class metrics
print("\nPer-class AP50:")
for i, ap in enumerate(metrics.box.ap50):
    print(f"  Class {i}: {ap:.4f}")

# %% [markdown]
# ---
# # Part 13: Model Export (Deployment)
# ---
#
# ## 13.1 Export Formats
#
# | Format | Argument | Use Case |
# |--------|----------|----------|
# | PyTorch | - | Training, Research |
# | ONNX | onnx | Cross-platform |
# | TensorRT | engine | NVIDIA GPU |
# | CoreML | coreml | Apple devices |
# | TFLite | tflite | Mobile, Edge |
# | OpenVINO | openvino | Intel hardware |
# | TorchScript | torchscript | Production Python-free |
# | Paddle | paddle | Baidu ecosystem |

# %%
# =============================================================================
# 13.1 Export to ONNX
# =============================================================================

model = YOLO("yolo26n.pt")

# Export to ONNX
onnx_path = model.export(
    format="onnx",
    imgsz=640,           # Input size
    half=False,          # FP16 export
    int8=False,          # INT8 quantization
    dynamic=True,        # Dynamic input size
    simplify=True,       # Simplify ONNX graph
    opset=17,            # ONNX opset version
    batch=1,             # Batch size
    workspace=4,         # TensorRT workspace (GB)
)

print(f"✅ Exported to: {onnx_path}")

# %%
# =============================================================================
# 13.2 Export to TensorRT
# =============================================================================

# Export to TensorRT (requires NVIDIA GPU)
# engine_path = model.export(
#     format="engine",
#     imgsz=640,
#     half=True,       # FP16 for better speed
#     int8=False,      # INT8 quantization
#     dynamic=False,   # Fixed input size
#     batch=1,
#     workspace=4,     # GPU memory (GB)
#     device=0,        # GPU device
# )

print("💡 TensorRT export:")
print('   model.export(format="engine", half=True)')

# %%
# =============================================================================
# 13.3 Export with INT8 Quantization
# =============================================================================

# INT8 quantization สำหรับ edge devices
# int8_path = model.export(
#     format="engine",
#     imgsz=640,
#     int8=True,       # Enable INT8
#     data="coco.yaml" # Calibration dataset
# )

print("💡 INT8 Quantization export:")
print('   model.export(format="engine", int8=True, data="coco.yaml")')

# %%
# =============================================================================
# 13.4 Export to Multiple Formats
# =============================================================================

export_formats = {
    "onnx": {"half": False, "dynamic": True, "simplify": True},
    # "engine": {"half": True, "int8": False},  # TensorRT
    # "coreml": {"half": True},                 # Apple CoreML
    # "tflite": {"int8": True},                 # TensorFlow Lite
    # "openvino": {"half": True},               # Intel OpenVINO
}

print("📦 Export to multiple formats:")
for fmt, kwargs in export_formats.items():
    print(f"   model.export(format='{fmt}', **{kwargs})")

# %%
# =============================================================================
# 13.5 Run Inference with Exported Model
# =============================================================================

# Load and run ONNX model
onnx_model = YOLO("yolo26n.onnx")
results = onnx_model("https://ultralytics.com/images/bus.jpg")

print(f"✅ ONNX inference successful")
print(f"   Detections: {len(results[0].boxes)}")

# Load and run TensorRT model
# trt_model = YOLO("yolo26n.engine")
# results = trt_model("image.jpg")

# %% [markdown]
# ---
# # Part 14: Benchmarking
# ---

# %%
# =============================================================================
# 14.1 Benchmark Different Formats
# =============================================================================

from ultralytics.utils.benchmarks import benchmark

# Benchmark model on different formats
# benchmark_results = benchmark(
#     model="yolo26n.pt",
#     data="coco8.yaml",
#     imgsz=640,
#     half=False,
#     int8=False,
#     device=0,
#     verbose=True
# )

print("💡 Benchmarking:")
print("""
from ultralytics.utils.benchmarks import benchmark

results = benchmark(
    model="yolo26n.pt",
    data="coco8.yaml",
    imgsz=640,
    device=0
)
""")

# %%
# =============================================================================
# 14.2 Speed Test
# =============================================================================

import time

model = YOLO("yolo26n.pt")

# Warmup
for _ in range(10):
    _ = model("https://ultralytics.com/images/bus.jpg", verbose=False)

# Benchmark
n_iterations = 100
start_time = time.time()
for _ in range(n_iterations):
    _ = model("https://ultralytics.com/images/bus.jpg", verbose=False)
end_time = time.time()

avg_time = (end_time - start_time) / n_iterations * 1000
fps = 1000 / avg_time

print("\n⚡ Speed Benchmark:")
print("=" * 40)
print(f"Average inference time: {avg_time:.2f} ms")
print(f"FPS: {fps:.1f}")
print("=" * 40)

# %% [markdown]
# ---
# # Part 15: Advanced Techniques
# ---

# %%
# =============================================================================
# 15.1 Multi-GPU Training
# =============================================================================

print("💡 Multi-GPU Training:")
print("""
# ใช้ DDP (Distributed Data Parallel)
model = YOLO("yolo26n.pt")
model.train(
    data="data.yaml",
    epochs=100,
    device=[0, 1, 2, 3]  # Use 4 GPUs
)
""")

# %%
# =============================================================================
# 15.2 Hyperparameter Tuning
# =============================================================================

print("💡 Hyperparameter Tuning:")
print("""
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.tune(
    data="coco8.yaml",
    epochs=30,
    iterations=300,
    optimizer="AdamW",
    plots=True,
    save=True,
    val=True
)
""")

# %%
# =============================================================================
# 15.3 Knowledge Distillation
# =============================================================================

print("💡 Knowledge Distillation:")
print("""
# Train small model with large teacher
model = YOLO("yolo26n.yaml")
model.train(
    data="data.yaml",
    epochs=100,
    # Teacher model for distillation
    teacher="yolo26x.pt"  # Large model as teacher
)
""")

# %%
# =============================================================================
# 15.4 Freeze Layers for Transfer Learning
# =============================================================================

print("💡 Freeze Layers:")
print("""
model = YOLO("yolo26n.pt")
model.train(
    data="data.yaml",
    epochs=100,
    freeze=10  # Freeze first 10 layers
)
""")

# %%
# =============================================================================
# 15.5 Custom Callbacks
# =============================================================================

from ultralytics import YOLO
from ultralytics.utils.callbacks import add_integration_callbacks

def on_train_epoch_end(trainer):
    """Custom callback หลังจบแต่ละ epoch"""
    epoch = trainer.epoch
    metrics = trainer.metrics
    print(f"Epoch {epoch} completed. mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}")

def on_val_end(validator):
    """Custom callback หลัง validation"""
    print(f"Validation completed.")

# Register callbacks
# model = YOLO("yolo26n.pt")
# model.add_callback("on_train_epoch_end", on_train_epoch_end)
# model.add_callback("on_val_end", on_val_end)

print("💡 Custom Callbacks registered")

# %% [markdown]
# ---
# # Part 16: Real-World Applications
# ---

# %%
# =============================================================================
# 16.1 Real-time Webcam Detection
# =============================================================================

def realtime_webcam_detection():
    """Real-time detection บน webcam"""
    model = YOLO("yolo26n.pt")
    
    # เปิด webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detection
        results = model(frame, verbose=False)
        
        # Annotate frame
        annotated_frame = results[0].plot()
        
        # แสดงผล
        cv2.imshow("YOLO26 Real-time Detection", annotated_frame)
        
        # กด 'q' เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Uncomment to run:
# realtime_webcam_detection()

print("💡 Webcam detection ready - uncomment realtime_webcam_detection() to run")

# %%
# =============================================================================
# 16.2 Batch Processing Images
# =============================================================================

def batch_process_folder(input_folder, output_folder, model_path="yolo26n.pt"):
    """Process ทุก images ใน folder"""
    import glob
    from pathlib import Path
    
    model = YOLO(model_path)
    
    # สร้าง output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # หา images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(f"{input_folder}/{ext}"))
    
    print(f"Found {len(images)} images")
    
    # Process แต่ละ image
    for img_path in images:
        results = model(img_path, verbose=False)
        
        # Save annotated image
        output_path = Path(output_folder) / Path(img_path).name
        results[0].save(str(output_path))
        
        print(f"Processed: {Path(img_path).name} -> {len(results[0].boxes)} detections")
    
    print(f"\n✅ All results saved to: {output_folder}")

# Example usage:
# batch_process_folder("input_images", "output_results")

print("💡 Batch processing function ready")

# %%
# =============================================================================
# 16.3 Count Objects in ROI (Region of Interest)
# =============================================================================

def count_objects_in_roi(image_path, roi_points, class_filter=None):
    """นับ objects ภายใน ROI ที่กำหนด"""
    import cv2
    import numpy as np
    
    model = YOLO("yolo26n.pt")
    results = model(image_path, verbose=False)
    
    # สร้าง ROI mask
    img = cv2.imread(image_path)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    roi_array = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(mask, [roi_array], 255)
    
    # นับ objects ใน ROI
    count = 0
    for box in results[0].boxes:
        # หา center ของ box
        xyxy = box.xyxy[0].cpu().numpy()
        center_x = int((xyxy[0] + xyxy[2]) / 2)
        center_y = int((xyxy[1] + xyxy[3]) / 2)
        
        # Check ว่าอยู่ใน ROI หรือไม่
        if mask[center_y, center_x] == 255:
            cls = int(box.cls[0])
            if class_filter is None or cls in class_filter:
                count += 1
    
    return count

print("💡 ROI counting function ready")

# %% [markdown]
# ---
# # Part 17: Lab Exercises
# ---
#
# ## Exercise 1: Basic Detection (20 points)
# 1. โหลด YOLO26s model
# 2. ทำ detection บน image ที่กำหนด
# 3. กรองเฉพาะ class "person" และ "car"
# 4. แสดง bounding boxes และ confidence scores
#
# ## Exercise 2: Instance Segmentation (20 points)
# 1. โหลด YOLO26-seg model
# 2. ทำ segmentation บน image
# 3. Extract masks ของแต่ละ instance
# 4. คำนวณพื้นที่ (pixels) ของแต่ละ mask
#
# ## Exercise 3: Custom Training (30 points)
# 1. เตรียม dataset ในรูปแบบ YOLO format
# 2. สร้าง data.yaml configuration
# 3. Train model ด้วย hyperparameters ที่เหมาะสม
# 4. Validate และรายงาน mAP metrics
#
# ## Exercise 4: Model Export & Deployment (30 points)
# 1. Export trained model เป็น ONNX format
# 2. ทำ inference ด้วย ONNX model
# 3. เปรียบเทียบ speed ระหว่าง PyTorch และ ONNX
# 4. (Bonus) Export เป็น TensorRT และเปรียบเทียบ

# %%
# =============================================================================
# Exercise 1: Basic Detection
# =============================================================================

def exercise_1():
    """
    Exercise 1: Basic Detection
    
    Tasks:
    1. โหลด YOLO26s model
    2. ทำ detection บน image
    3. กรองเฉพาะ class "person" (0) และ "car" (2)
    4. แสดงผลลัพธ์
    """
    # TODO: เขียน code ของคุณที่นี่
    
    # Step 1: โหลด model
    # model = YOLO(???)
    
    # Step 2: ทำ detection พร้อม filter classes
    # results = model.predict(???)
    
    # Step 3: แสดงข้อมูล detections
    # for r in results:
    #     ...
    
    pass

# exercise_1()

# %%
# =============================================================================
# Exercise 2: Instance Segmentation
# =============================================================================

def exercise_2():
    """
    Exercise 2: Instance Segmentation
    
    Tasks:
    1. โหลด segmentation model
    2. ทำ segmentation
    3. Extract และคำนวณพื้นที่ masks
    """
    # TODO: เขียน code ของคุณที่นี่
    
    pass

# exercise_2()

# %%
# =============================================================================
# Exercise 3: Custom Training
# =============================================================================

def exercise_3():
    """
    Exercise 3: Custom Training
    
    Tasks:
    1. สร้าง dataset structure
    2. สร้าง data.yaml
    3. Train model
    4. Validate และรายงาน metrics
    """
    # TODO: เขียน code ของคุณที่นี่
    
    pass

# exercise_3()

# %%
# =============================================================================
# Exercise 4: Export & Deployment
# =============================================================================

def exercise_4():
    """
    Exercise 4: Model Export & Deployment
    
    Tasks:
    1. Export to ONNX
    2. Compare inference speed
    3. (Bonus) TensorRT export
    """
    # TODO: เขียน code ของคุณที่นี่
    
    pass

# exercise_4()

# %% [markdown]
# ---
# # Part 18: Summary & Best Practices
# ---
#
# ## 18.1 YOLO26 Key Takeaways
#
# 1. **NMS-Free Architecture** - ลด latency และทำให้ deployment ง่ายขึ้น
# 2. **DFL Removal** - export ง่ายขึ้น, รองรับ hardware หลากหลาย
# 3. **MuSGD Optimizer** - training stable และ converge เร็ว
# 4. **ProgLoss + STAL** - ปรับปรุง small object detection
# 5. **Multi-Task Support** - Detection, Segmentation, Pose, OBB, Classification
#
# ## 18.2 Best Practices
#
# ### Training:
# - ใช้ pretrained weights เสมอ (transfer learning)
# - ใช้ data augmentation อย่างเหมาะสม
# - Monitor training ด้วย TensorBoard/Weights & Biases
# - ใช้ early stopping เพื่อป้องกัน overfitting
#
# ### Inference:
# - ใช้ batch inference เมื่อเป็นไปได้
# - Export เป็น TensorRT สำหรับ NVIDIA GPU
# - ใช้ INT8 quantization สำหรับ edge devices
# - Set confidence threshold ตามความเหมาะสมของ application
#
# ### Deployment:
# - Test thoroughly ก่อน deploy
# - Monitor performance ใน production
# - Version control สำหรับ models
# - Document model limitations

# %%
# =============================================================================
# Final: Cleanup and Summary
# =============================================================================

print("\n" + "=" * 60)
print("🎉 YOLO26 Advanced Lab Complete!")
print("=" * 60)
print("""
สิ่งที่ได้เรียนรู้ในวันนี้:

✅ YOLO26 Architecture (NMS-Free, DFL Removal)
✅ Object Detection with custom parameters
✅ Instance Segmentation and mask extraction
✅ Image Classification
✅ Pose Estimation with keypoints
✅ Oriented Bounding Boxes (OBB)
✅ Object Tracking (BoT-SORT, ByteTrack)
✅ YOLOE-26 Open-Vocabulary Detection
✅ Training on Custom Datasets
✅ Model Validation and Metrics
✅ Export to ONNX, TensorRT, CoreML, TFLite
✅ Benchmarking and Optimization
✅ Real-world Applications

📚 References:
- Ultralytics Docs: https://docs.ultralytics.com
- YOLO26 Paper: https://arxiv.org/abs/2509.25164
- GitHub: https://github.com/ultralytics/ultralytics

Happy Learning! 🚀
""")
print("=" * 60)
