# %% [markdown]
# # 🔬 Lab: YOLO26 + Depth Estimation
# ## Combining Object Detection with Monocular Depth Estimation
#
# **Objectives:**
# 1. Understand Depth Estimation principles (Monocular Depth Estimation)
# 2. Use YOLO26 for Object Detection
# 3. Use MiDaS for Depth Estimation
# 4. Combine YOLO26 + MiDaS to estimate object distance from camera
# 5. Use Object Size-based Distance Estimation (YOLO BBox heuristics)
# 6. Create 3D-aware Visualizations
#
# **Tools:**
# - Ultralytics YOLO26 (Object Detection)
# - Intel MiDaS (Monocular Depth Estimation)
# - OpenCV, Matplotlib, NumPy
#
# ---

# %% [markdown]
# ## Part 1: Depth Estimation Theory
#
# ### What is Depth Estimation?
#
# **Depth Estimation** is the process of estimating the distance of each pixel in an image from the camera.
# The result is a **Depth Map** where pixel intensity represents distance.
#
# ### Types of Depth Estimation
#
# | Type | Description | Pros | Cons |
# |------|------------|------|------|
# | **Stereo Vision** | Uses 2 cameras, computes from disparity | Accurate | Requires 2 cameras |
# | **LiDAR / ToF** | Uses laser light to measure distance | Very accurate | Expensive, specialized hardware |
# | **Monocular Depth** | Uses single image + Deep Learning | Works with normal camera | Gives relative depth only |
#
# ### MiDaS (Multiple Depth from a Single Image)
#
# MiDaS is a model from Intel Labs using Encoder-Decoder architecture:
# - **Encoder**: ResNet / DPT (Dense Prediction Transformer) for feature extraction
# - **Decoder**: Upsampling + Feature Fusion to create depth map
# - Trained on 12+ datasets → works well with diverse image types
#
# ### Concept: YOLO + Depth Estimation
#
# ```
# Input Image
#    ├── YOLO26 → Bounding Boxes + Class Labels (2D Detection)
#    ├── MiDaS  → Depth Map (distance per pixel)
#    └── BBox Size Heuristic → Estimated distance from known object sizes
#         ↓
#    Combined → Each object + estimated distance (Pseudo-3D)
# ```
#
# ### Object Distance Estimation using BBox Size
#
# In addition to MiDaS depth maps, we can estimate distance using a simple
# **pinhole camera model** heuristic:
#
# ```
# Distance ≈ (Known Real Height × Focal Length) / BBox Height in Pixels
# ```
#
# This approach uses the fact that objects appear smaller when farther away.
# By assuming typical real-world sizes for known COCO classes (e.g., a person
# is ~1.7m tall), we can convert bounding box pixel height into an approximate
# metric distance (in meters). This complements MiDaS relative depth with
# an absolute distance estimate.

# %% [markdown]
# ## Part 2: Install Libraries

# %%
# Install required libraries
# !pip install ultralytics opencv-python matplotlib numpy torch torchvision timm --quiet

# %%
import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Part 3: YOLO26 Object Detection
#
# YOLO26 is the latest model from Ultralytics (2025) with key features:
# - **NMS-Free**: No Non-Maximum Suppression needed → faster inference
# - **DFL Removed**: Removed Distribution Focal Loss → easier deployment
# - **Edge Optimized**: 43% faster CPU inference
# - Supports: Detection, Segmentation, Pose, OBB, Classification

# %%
from ultralytics import YOLO

# Load YOLO26 nano model (pretrained on COCO dataset — 80 classes)
model = YOLO("yolo26n.pt")

# Run inference on image
IMAGE_PATH = ".././images/football_teamplay.jpeg"
results = model(IMAGE_PATH, imgsz=640)

# %%
# ============================================================
# Assign persistent Object IDs at the FIRST detection
# These IDs will be used consistently throughout the entire lab
# ============================================================
result = results[0]

# Build a master object registry with stable IDs
OBJECT_REGISTRY = []
for i, box in enumerate(result.boxes):
    cls_id = int(box.cls[0])
    cls_name = result.names[cls_id]
    conf = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    OBJECT_REGISTRY.append({
        "obj_id": i,           # ← stable ID used everywhere
        "class": cls_name,
        "class_id": cls_id,
        "confidence": conf,
        "bbox": (int(x1), int(y1), int(x2), int(y2)),
        "bbox_area": int((x2 - x1) * (y2 - y1)),
        "center_x": int((x1 + x2) / 2),
        "center_y": int((y1 + y2) / 2),
        "bbox_width": int(x2 - x1),
        "bbox_height": int(y2 - y1),
    })

print("=" * 70)
print("  YOLO26 Detection Results — Master Object Registry")
print("=" * 70)
print(f"  Total objects detected : {len(OBJECT_REGISTRY)}")
print(f"  Image size             : {result.orig_shape}")
print(f"  Unique classes         : {result.boxes.cls.unique().tolist()}")
print()
print(f"  {'ID':>4} {'Class':>12} {'Conf':>6} {'BBox (x1,y1,x2,y2)':>28} {'Area':>8}")
print("  " + "-" * 66)
for obj in OBJECT_REGISTRY:
    x1, y1, x2, y2 = obj["bbox"]
    print(f"  {obj['obj_id']:>4} {obj['class']:>12} {obj['confidence']:>6.2f} "
          f"  ({x1:>4},{y1:>4},{x2:>4},{y2:>4}) {obj['bbox_area']:>8}")

print()
print("  These Object IDs will be referenced throughout the lab.")

# %%
# Visualize YOLO26 detection
annotated_img = result.plot()
annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(annotated_img_rgb)
plt.title("YOLO26 Object Detection", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 4: Object Distance Estimation using YOLO BBox Size
#
# ### Pinhole Camera Model Heuristic
#
# Before using a depth model, we can get a rough **metric distance** estimate
# by exploiting a simple geometric relationship:
#
# ```
# Estimated Distance (m) = (Real World Height × Focal Length) / BBox Height (px)
# ```
#
# We use typical real-world heights for common COCO classes and an assumed
# focal length. This gives **absolute distance in meters** (unlike MiDaS
# which only gives relative depth).
#
# **Limitations:**
# - Assumes the object is upright and fully visible
# - Focal length is estimated (not calibrated)
# - Works best for objects with known/consistent real-world sizes
#
# **Advantages:**
# - No extra model needed — uses only YOLO detection output
# - Gives metric distance (meters) instead of relative values
# - Very fast — just arithmetic on bounding box dimensions

# %%
# ==============================================================
# Known real-world heights (meters) for common COCO classes
# Used for pinhole-model distance estimation
# ==============================================================
KNOWN_HEIGHTS_M = {
    "person": 1.70,
    "bicycle": 1.10,
    "car": 1.50,
    "motorcycle": 1.20,
    "bus": 3.00,
    "truck": 3.50,
    "dog": 0.50,
    "cat": 0.30,
    "chair": 0.90,
    "bottle": 0.25,
    "cup": 0.12,
    "tv": 0.60,
    "laptop": 0.30,
    "cell phone": 0.14,
    "sports ball": 0.22,
    "backpack": 0.50,
    "umbrella": 1.00,
    "handbag": 0.35,
    "suitcase": 0.70,
    "bench": 0.85,
    "bird": 0.20,
    "horse": 1.60,
    "sheep": 0.80,
    "cow": 1.40,
    "elephant": 3.00,
    "bear": 1.50,
    "zebra": 1.40,
    "giraffe": 5.50,
}

# Default height for classes not in the table
DEFAULT_HEIGHT_M = 0.50

# Assumed focal length in pixels (rough estimate for a typical camera)
# For more accurate results, calibrate your camera.
ASSUMED_FOCAL_LENGTH_PX = 800


def estimate_distance_from_bbox(objects, focal_length=ASSUMED_FOCAL_LENGTH_PX):
    """
    Estimate metric distance (meters) from bounding box height
    using the pinhole camera model heuristic.
    
    Parameters:
    -----------
    objects : list of dict — object registry entries
    focal_length : float — estimated focal length in pixels
    
    Returns:
    --------
    list of dict — same objects with 'estimated_distance_m' added
    """
    for obj in objects:
        real_height = KNOWN_HEIGHTS_M.get(obj["class"], DEFAULT_HEIGHT_M)
        bbox_h = obj["bbox_height"]
        
        if bbox_h > 0:
            distance_m = (real_height * focal_length) / bbox_h
        else:
            distance_m = float("inf")
        
        obj["real_height_m"] = real_height
        obj["estimated_distance_m"] = round(distance_m, 2)
    
    return objects


# %%
# Apply distance estimation to all detected objects
OBJECT_REGISTRY = estimate_distance_from_bbox(OBJECT_REGISTRY)

# Sort by estimated distance (nearest first)
objects_by_distance = sorted(OBJECT_REGISTRY, key=lambda x: x["estimated_distance_m"])

print("=" * 80)
print("  Object Distance Estimation (Pinhole Camera Model)")
print("=" * 80)
print(f"  Assumed focal length: {ASSUMED_FOCAL_LENGTH_PX} px")
print()
print(f"  {'ID':>4} {'Class':>12} {'BBox H(px)':>10} {'Real H(m)':>10} {'Est. Dist(m)':>13}")
print("  " + "-" * 55)

for obj in objects_by_distance:
    print(f"  {obj['obj_id']:>4} {obj['class']:>12} {obj['bbox_height']:>10} "
          f"{obj['real_height_m']:>10.2f} {obj['estimated_distance_m']:>13.2f}")

# %%
# Visualize BBox-based distance estimation on the image
def draw_distance_annotated_image(image_rgb, objects):
    """
    Draw bounding boxes with metric distance labels on the image.
    Color: green (near) → red (far)
    """
    img_out = image_rgb.copy()
    
    # Get distance range for normalization
    distances = [o["estimated_distance_m"] for o in objects if o["estimated_distance_m"] < float("inf")]
    if not distances:
        return img_out
    d_min, d_max = min(distances), max(distances)
    d_range = d_max - d_min if d_max > d_min else 1.0
    
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        dist = obj["estimated_distance_m"]
        norm = min(1.0, (dist - d_min) / d_range)
        
        # Color: near=green, far=red
        r, g, b = int(255 * norm), int(255 * (1 - norm)), 0
        color = (r, g, b)
        
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID{obj['obj_id']} {obj['class']} ~{dist:.1f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img_out


img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

dist_annotated = draw_distance_annotated_image(img_rgb, OBJECT_REGISTRY)

plt.figure(figsize=(12, 8))
plt.imshow(dist_annotated)
plt.title("YOLO26 Object Detection — BBox Distance Estimation\n"
          "(Green = Near, Red = Far)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()

# %%
# Bar chart: Estimated distance per object
if OBJECT_REGISTRY:
    sorted_objs = sorted(OBJECT_REGISTRY, key=lambda x: x["estimated_distance_m"])
    labels = [f"ID{o['obj_id']} {o['class']}" for o in sorted_objs]
    dists = [o["estimated_distance_m"] for o in sorted_objs]
    
    d_min, d_max = min(dists), max(dists)
    d_range = d_max - d_min if d_max > d_min else 1.0
    colors = [plt.cm.RdYlGn_r((d - d_min) / d_range) for d in dists]
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.4)))
    ax.barh(labels, dists, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Estimated Distance (meters)", fontsize=12)
    ax.set_title("Object Distance Ranking (BBox Heuristic)\n"
                 "Shorter bar = closer to camera", fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Part 5: MiDaS Depth Estimation
#
# ### MiDaS Model Variants
#
# | Model | Size | Accuracy | Speed |
# |-------|------|----------|-------|
# | **DPT_Large** | Large | Highest | Slowest |
# | **DPT_Hybrid** | Medium | Medium | Medium |
# | **MiDaS_small** | Small | Lowest | Fastest |
#
# > ⚠️ MiDaS gives **relative depth** (not absolute in meters).
# > Higher value = farther from camera, Lower value = closer (inverse depth).

# %%
# Load MiDaS model from PyTorch Hub
# Choose model_type: "DPT_Large", "DPT_Hybrid", "MiDaS_small"

model_type = "MiDaS_small"  # Start with small model (fast, low RAM)

print(f"Loading MiDaS model: {model_type}...")
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()
print(f"MiDaS loaded successfully! (device: {device})")

# Load transform for preprocessing
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

print(f"Transform loaded successfully!")

# %%
# Read image and perform Depth Estimation
img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Preprocessing: transform image
input_batch = transform(img_rgb).to(device)

print(f"Input shape: {input_batch.shape}")

# Inference
with torch.no_grad():
    prediction = midas(input_batch)
    
    # Resize depth map to match original image size
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

print(f"Depth map shape: {depth_map.shape}")
print(f"Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")

# %%
# Visualize Depth Map
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original image
axes[0].imshow(img_rgb)
axes[0].set_title("Original Image", fontsize=14)
axes[0].axis("off")

# Depth Map (Inferno colormap)
im1 = axes[1].imshow(depth_map, cmap="inferno")
axes[1].set_title("Depth Map (Inferno)", fontsize=14)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Relative Depth")

# Depth Map inverted (near = bright)
depth_inv = depth_map.max() - depth_map
im2 = axes[2].imshow(depth_inv, cmap="plasma")
axes[2].set_title("Inverted Depth (Near = Bright)", fontsize=14)
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Inverted Depth")

plt.suptitle("MiDaS Monocular Depth Estimation", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 6: Combine YOLO26 + MiDaS Depth Estimation
#
# ### Concept
# 1. Use YOLO26 to detect objects → get bounding boxes
# 2. Use MiDaS to generate depth map
# 3. For each bounding box → crop depth map region → compute average depth
# 4. Average depth = relative distance of object from camera
# 5. Rank: which objects are nearest / farthest

# %%
def estimate_object_depth(boxes, depth_map, names, method="median"):
    """
    Compute depth of each detected object using MiDaS depth map.
    
    Parameters:
    -----------
    boxes : ultralytics Boxes object
    depth_map : numpy array — depth map from MiDaS
    names : dict — class names mapping
    method : str — depth computation method ("mean", "median", "center")
    
    Returns:
    --------
    list of dict — each object info with depth
    """
    objects_with_depth = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        conf = float(box.conf[0])
        
        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1], x2)
        y2 = min(depth_map.shape[0], y2)
        
        # Crop depth region by bounding box
        depth_region = depth_map[y1:y2, x1:x2]
        
        if depth_region.size == 0:
            continue
        
        # Compute depth using selected method
        if method == "mean":
            obj_depth = np.mean(depth_region)
        elif method == "median":
            obj_depth = np.median(depth_region)
        elif method == "center":
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            obj_depth = depth_map[cy, cx]
        else:
            obj_depth = np.mean(depth_region)
        
        objects_with_depth.append({
            "obj_id": i,            # matches OBJECT_REGISTRY id
            "class": cls_name,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
            "depth_value": obj_depth,
            "depth_std": np.std(depth_region),
            "bbox_area": (x2 - x1) * (y2 - y1),
        })
    
    # Sort by depth (MiDaS: higher value = farther)
    objects_with_depth.sort(key=lambda x: x["depth_value"])
    
    return objects_with_depth

# %%
# Compute depth of each object
objects = estimate_object_depth(result.boxes, depth_map, result.names, method="median")

# Normalize depth to 0-100 for display
depth_values = [obj["depth_value"] for obj in objects]
if depth_values:
    d_min, d_max = min(depth_values), max(depth_values)
    for obj in objects:
        if d_max > d_min:
            obj["depth_normalized"] = ((obj["depth_value"] - d_min) / (d_max - d_min)) * 100
        else:
            obj["depth_normalized"] = 50.0

# Merge BBox-based distance into depth results for cross-reference
for obj in objects:
    registry_match = next((r for r in OBJECT_REGISTRY if r["obj_id"] == obj["obj_id"]), None)
    if registry_match:
        obj["estimated_distance_m"] = registry_match.get("estimated_distance_m", None)

# Display results
print("=" * 95)
print("  YOLO26 + MiDaS Depth Estimation Results (with BBox Distance)")
print("=" * 95)
print(f"  {'ID':>4} {'Class':>12} {'Conf':>6} {'MiDaS Depth':>12} {'Norm%':>8} "
      f"{'BBox Dist(m)':>13} {'Zone':>10}")
print("  " + "-" * 90)

for rank, obj in enumerate(objects, 1):
    zone = "NEAR" if obj["depth_normalized"] < 33 else "MID" if obj["depth_normalized"] < 66 else "FAR"
    zone_icon = {"NEAR": "[NEAR]", "MID": "[MID]", "FAR": "[FAR]"}[zone]
    
    bbox_dist_str = f"{obj['estimated_distance_m']:.2f}" if obj.get("estimated_distance_m") else "N/A"
    
    print(f"  {obj['obj_id']:>4} {obj['class']:>12} {obj['confidence']:>6.2f} "
          f"{obj['depth_value']:>12.2f} {obj['depth_normalized']:>7.1f}% "
          f"{bbox_dist_str:>13} {zone_icon:>10}")

# %% [markdown]
# ## Part 7: Visualization — Annotated Image with Depth
#
# Create image showing both bounding boxes and distance of each object.

# %%
def draw_depth_annotated_image(image_rgb, objects, depth_map):
    """
    Draw bounding boxes with depth annotation on image.
    Box color changes by depth (near = green, far = red).
    """
    img_annotated = image_rgb.copy()
    
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        norm = obj["depth_normalized"] / 100.0  # 0 = near, 1 = far
        
        # Color: near = green (0,255,0), far = red (255,0,0)
        r = int(255 * norm)
        g = int(255 * (1 - norm))
        b = 0
        color = (r, g, b)
        
        # Draw bounding box
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
        
        # Label: ID + class + depth
        dist_str = f"~{obj['estimated_distance_m']:.1f}m" if obj.get("estimated_distance_m") else ""
        label = f"ID{obj['obj_id']} {obj['class']} D:{obj['depth_normalized']:.0f}% {dist_str}"
        
        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img_annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        
        # White text
        cv2.putText(img_annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img_annotated

# %%
# Create annotated image
annotated = draw_depth_annotated_image(img_rgb, objects, depth_map)

# Display
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes[0].imshow(annotated)
axes[0].set_title("YOLO26 Detection + Depth Annotation\n"
                  "(Green = Near,  Red = Far)", fontsize=14)
axes[0].axis("off")

# Depth Map overlay
depth_normalized_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
axes[1].imshow(img_rgb, alpha=0.5)
axes[1].imshow(depth_normalized_map, cmap="inferno", alpha=0.5)
axes[1].set_title("Original + Depth Map Overlay", fontsize=14)
axes[1].axis("off")

plt.suptitle("YOLO26 + MiDaS Depth Estimation", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 8: Depth-based Object Sorting & Visualization
#
# Categorize objects by distance: Near, Mid, Far

# %%
def categorize_by_depth(objects):
    """Split objects into 3 groups by depth"""
    near = [o for o in objects if o["depth_normalized"] < 33]
    mid  = [o for o in objects if 33 <= o["depth_normalized"] < 66]
    far  = [o for o in objects if o["depth_normalized"] >= 66]
    return near, mid, far

near_objects, mid_objects, far_objects = categorize_by_depth(objects)

print("=" * 60)
print("  Object Distance Categorization")
print("=" * 60)

print(f"\n  [NEAR] — {len(near_objects)} objects:")
for o in near_objects:
    dist_str = f", ~{o['estimated_distance_m']:.1f}m" if o.get("estimated_distance_m") else ""
    print(f"    - ID{o['obj_id']} {o['class']} (conf: {o['confidence']:.2f}, "
          f"depth: {o['depth_normalized']:.1f}%{dist_str})")

print(f"\n  [MID] — {len(mid_objects)} objects:")
for o in mid_objects:
    dist_str = f", ~{o['estimated_distance_m']:.1f}m" if o.get("estimated_distance_m") else ""
    print(f"    - ID{o['obj_id']} {o['class']} (conf: {o['confidence']:.2f}, "
          f"depth: {o['depth_normalized']:.1f}%{dist_str})")

print(f"\n  [FAR] — {len(far_objects)} objects:")
for o in far_objects:
    dist_str = f", ~{o['estimated_distance_m']:.1f}m" if o.get("estimated_distance_m") else ""
    print(f"    - ID{o['obj_id']} {o['class']} (conf: {o['confidence']:.2f}, "
          f"depth: {o['depth_normalized']:.1f}%{dist_str})")

# %%
# Charts: Depth ranking + BBox Area vs Depth
if objects:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart — MiDaS relative depth
    labels = [f"ID{o['obj_id']} {o['class']}" for o in objects]
    depths = [o["depth_normalized"] for o in objects]
    colors = [plt.cm.RdYlGn_r(d / 100.0) for d in depths]
    
    bars = axes[0].barh(labels, depths, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Relative Depth (%)", fontsize=12)
    axes[0].set_title("Object Depth Ranking (MiDaS)\n"
                      "(0% = Nearest, 100% = Farthest)", fontsize=14)
    axes[0].axvline(x=33, color="green", linestyle="--", alpha=0.5, label="Near/Mid boundary")
    axes[0].axvline(x=66, color="red", linestyle="--", alpha=0.5, label="Mid/Far boundary")
    axes[0].legend()
    axes[0].set_xlim(0, 105)
    
    # Scatter plot: BBox Area vs Depth
    areas = [o["bbox_area"] for o in objects]
    axes[1].scatter(depths, areas, c=colors, s=100, edgecolors="black", linewidth=0.5)
    for o in objects:
        axes[1].annotate(f"ID{o['obj_id']} {o['class']}", 
                        (o["depth_normalized"], o["bbox_area"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[1].set_xlabel("Relative Depth (%)", fontsize=12)
    axes[1].set_ylabel("Bounding Box Area (pixels sq.)", fontsize=12)
    axes[1].set_title("BBox Area vs Depth\n"
                      "(Closer objects tend to have larger bboxes)", fontsize=14)
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Part 9: Compare MiDaS vs BBox Distance Estimation
#
# How well does the simple BBox heuristic correlate with MiDaS depth?

# %%
if objects:
    # Gather paired data
    midas_depths = []
    bbox_dists = []
    obj_labels = []
    
    for obj in objects:
        if obj.get("estimated_distance_m") is not None:
            midas_depths.append(obj["depth_normalized"])
            bbox_dists.append(obj["estimated_distance_m"])
            obj_labels.append(f"ID{obj['obj_id']} {obj['class']}")
    
    if len(midas_depths) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        scatter = ax.scatter(midas_depths, bbox_dists, s=100, 
                           c=midas_depths, cmap="RdYlGn_r",
                           edgecolors="black", linewidth=0.5)
        
        for lbl, x, y in zip(obj_labels, midas_depths, bbox_dists):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        
        ax.set_xlabel("MiDaS Relative Depth (%)", fontsize=12)
        ax.set_ylabel("BBox Estimated Distance (m)", fontsize=12)
        ax.set_title("MiDaS Depth vs BBox Distance Estimation\n"
                     "Correlation between two methods", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Compute correlation
        corr = np.corrcoef(midas_depths, bbox_dists)[0, 1]
        ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
                fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
        plt.colorbar(scatter, label="MiDaS Depth (%)")
        plt.tight_layout()
        plt.show()
        
        print(f"\n  Pearson correlation between MiDaS depth and BBox distance: {corr:.3f}")
        if corr > 0.7:
            print("  -> Strong positive correlation: both methods agree well.")
        elif corr > 0.3:
            print("  -> Moderate correlation: partial agreement between methods.")
        else:
            print("  -> Weak correlation: methods give different depth ordering.")
    else:
        print("  Not enough data points for correlation analysis.")

# %% [markdown]
# ## Part 10: Video Processing — Frame-by-Frame Detection + Depth
#
# Apply detection + depth estimation frame by frame on video.

# %%
def process_video_with_depth(video_path, yolo_model, midas_model, midas_transform,
                              device, max_frames=30, conf_threshold=0.4):
    """
    Process video: YOLO26 detection + MiDaS depth estimation
    
    Parameters:
    -----------
    video_path : str — path to video file
    yolo_model : YOLO model
    midas_model : MiDaS model
    midas_transform : MiDaS preprocessing transform
    device : torch.device
    max_frames : int — maximum frames to process
    conf_threshold : float — confidence threshold for YOLO
    
    Returns:
    --------
    list of dict — results per frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.1f} FPS")
    print(f"Processing {min(max_frames, total_frames)} frames...")
    
    frame_results = []
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # YOLO26 Detection
        yolo_results = yolo_model(frame_rgb, imgsz=640, conf=conf_threshold, verbose=False)
        
        # MiDaS Depth Estimation
        input_batch = midas_transform(frame_rgb).to(device)
        with torch.no_grad():
            depth_pred = midas_model(input_batch)
            depth_pred = torch.nn.functional.interpolate(
                depth_pred.unsqueeze(1),
                size=frame_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_frame = depth_pred.cpu().numpy()
        
        # Compute depth of each object
        result = yolo_results[0]
        objects = estimate_object_depth(result.boxes, depth_frame, result.names, method="median")
        
        frame_results.append({
            "frame_id": frame_count,
            "num_objects": len(objects),
            "objects": objects,
            "depth_map": depth_frame,
            "frame_rgb": frame_rgb,
        })
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"  Frame {frame_count}/{min(max_frames, total_frames)}")
    
    cap.release()
    print(f"Processing complete: {frame_count} frames")
    return frame_results

# %%
# Example usage with video (uncomment when you have a video file)
# VIDEO_PATH = ".././videos/sample.mp4"
# 
# video_results = process_video_with_depth(
#     VIDEO_PATH, model, midas, transform, device,
#     max_frames=30, conf_threshold=0.4
# )
# 
# # Display sample frame
# if video_results:
#     sample = video_results[0]
#     annotated = draw_depth_annotated_image(
#         sample["frame_rgb"], sample["objects"], sample["depth_map"]
#     )
#     plt.figure(figsize=(12, 8))
#     plt.imshow(annotated)
#     plt.title(f"Frame {sample['frame_id']} — {sample['num_objects']} objects detected")
#     plt.axis("off")
#     plt.show()

print("Video processing function is ready!")
print("Uncomment the code above when you have a video file.")

# %% [markdown]
# ## Part 11: Compare MiDaS Models
#
# Compare 3 MiDaS variants: DPT_Large, DPT_Hybrid, MiDaS_small

# %%
import time

def compare_midas_models(image_path, model_types=None):
    """
    Compare different MiDaS models.
    """
    if model_types is None:
        model_types = ["MiDaS_small"]  # Use small only for speed
        # model_types = ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]  # full comparison
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results_compare = {}
    
    for mt in model_types:
        print(f"\nLoading {mt}...")
        m = torch.hub.load("intel-isl/MiDaS", mt)
        m.to(device)
        m.eval()
        
        t = torch.hub.load("intel-isl/MiDaS", "transforms")
        if mt in ["DPT_Large", "DPT_Hybrid"]:
            tf = t.dpt_transform
        else:
            tf = t.small_transform
        
        input_batch = tf(img_rgb).to(device)
        
        # Warm up
        with torch.no_grad():
            _ = m(input_batch)
        
        # Benchmark
        start = time.time()
        n_runs = 5
        for _ in range(n_runs):
            with torch.no_grad():
                pred = m(input_batch)
        elapsed = (time.time() - start) / n_runs
        
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=img_rgb.shape[:2],
            mode="bicubic", align_corners=False
        ).squeeze()
        
        depth = pred.cpu().numpy()
        results_compare[mt] = {
            "depth_map": depth,
            "time_ms": elapsed * 1000,
        }
        print(f"  {mt}: {elapsed*1000:.1f}ms/frame")
        
        # Free model memory
        del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results_compare

# %%
# Compare (using small model only for speed)
comparison = compare_midas_models(IMAGE_PATH, model_types=["MiDaS_small"])

# Uncomment for full comparison (uses more RAM):
# comparison = compare_midas_models(IMAGE_PATH, model_types=["MiDaS_small", "DPT_Hybrid", "DPT_Large"])

# %%
# Visualize comparison
n_models = len(comparison)
fig, axes = plt.subplots(1, n_models + 1, figsize=(6 * (n_models + 1), 5))

if n_models == 1:
    axes = [axes] if not isinstance(axes, np.ndarray) else axes.tolist()

# Original image
ax0 = axes[0] if isinstance(axes, list) else axes[0]
ax0.imshow(img_rgb)
ax0.set_title("Original", fontsize=12)
ax0.axis("off")

for idx, (mt, data) in enumerate(comparison.items(), 1):
    ax = axes[idx] if isinstance(axes, list) else axes[idx]
    ax.imshow(data["depth_map"], cmap="inferno")
    ax.set_title(f"{mt}\n({data['time_ms']:.1f} ms)", fontsize=12)
    ax.axis("off")

plt.suptitle("MiDaS Model Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 12: Advanced — Depth-Aware Object Priority
#
# ### Use Case: ADAS (Advanced Driver Assistance System)
# Closer objects → higher risk → alert first

# %%
def compute_danger_score(obj, weights=None):
    """
    Compute danger score of an object (for ADAS scenario).
    
    Objects that are close + large + human → very dangerous
    
    Parameters:
    -----------
    obj : dict — object info from estimate_object_depth
    weights : dict — weights for each factor
    
    Returns:
    --------
    float — danger score (0-100)
    """
    if weights is None:
        weights = {"proximity": 0.5, "size": 0.2, "class_risk": 0.3}
    
    # Proximity score (near = more danger) — invert depth
    proximity = 100 - obj["depth_normalized"]
    
    # Size score (larger = more danger)
    max_area = 500 * 500  # max value for normalization
    size_score = min(100, (obj["bbox_area"] / max_area) * 100)
    
    # Class risk (some classes are more dangerous)
    high_risk_classes = {"person": 100, "bicycle": 80, "motorcycle": 80, "car": 70, 
                         "bus": 60, "truck": 60, "dog": 70, "cat": 50}
    class_risk = high_risk_classes.get(obj["class"], 30)
    
    # Weighted sum
    danger = (weights["proximity"] * proximity +
              weights["size"] * size_score +
              weights["class_risk"] * class_risk)
    
    return min(100, danger)

# %%
# Compute danger scores
if objects:
    for obj in objects:
        obj["danger_score"] = compute_danger_score(obj)
    
    # Sort by danger score (high = more dangerous)
    objects_by_danger = sorted(objects, key=lambda x: x["danger_score"], reverse=True)
    
    print("=" * 80)
    print("  DANGER SCORE RANKING (ADAS Scenario)")
    print("=" * 80)
    print(f"  {'Rank':>4} {'ID':>4} {'Class':>12} {'Depth%':>8} {'Area':>8} "
          f"{'Danger':>8} {'Alert':>10}")
    print("  " + "-" * 70)
    
    for rank, obj in enumerate(objects_by_danger, 1):
        danger = obj["danger_score"]
        if danger >= 70:
            alert = "!! HIGH"
        elif danger >= 40:
            alert = "!  MEDIUM"
        else:
            alert = "   LOW"
        
        print(f"  {rank:>4} {obj['obj_id']:>4} {obj['class']:>12} "
              f"{obj['depth_normalized']:>7.1f}% {obj['bbox_area']:>8} "
              f"{danger:>7.1f} {alert:>10}")

# %% [markdown]
# ## Part 13: Top-Down View (Bird's Eye View)
#
# Display object positions from a top-down perspective.

# %%
def create_topdown_view(objects, img_width, img_height, figsize=(8, 10)):
    """
    Create top-down view showing x position (left-right) vs depth (near-far).
    """
    if not objects:
        print("No objects to display")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        cx = (x1 + x2) / 2  # center x
        depth_pct = obj["depth_normalized"]
        danger = obj.get("danger_score", 50)
        
        # Normalize x position (0-100)
        x_norm = (cx / img_width) * 100
        
        # Color by danger score
        color = plt.cm.RdYlGn_r(danger / 100)
        
        # Marker size by bbox area
        marker_size = max(50, min(500, obj["bbox_area"] / 100))
        
        ax.scatter(x_norm, depth_pct, s=marker_size, c=[color], 
                  edgecolors="black", linewidth=1, zorder=5, alpha=0.8)
        
        dist_str = f"\n~{obj['estimated_distance_m']:.1f}m" if obj.get("estimated_distance_m") else ""
        ax.annotate(f"ID{obj['obj_id']} {obj['class']}\n(danger:{danger:.0f}){dist_str}", 
                   (x_norm, depth_pct),
                   textcoords="offset points", xytext=(10, 5),
                   fontsize=8, ha="left")
    
    # Camera position (bottom)
    ax.scatter(50, -5, s=200, c="blue", marker="^", zorder=10, label="Camera")
    ax.annotate("Camera", (50, -5), textcoords="offset points", 
               xytext=(0, -15), fontsize=10, ha="center", fontweight="bold")
    
    # Zone bands
    ax.axhspan(-10, 33, alpha=0.1, color="green", label="Near Zone")
    ax.axhspan(33, 66, alpha=0.1, color="yellow", label="Mid Zone")
    ax.axhspan(66, 110, alpha=0.1, color="red", label="Far Zone")
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(-10, 110)
    ax.set_xlabel("Horizontal Position (Left to Right)", fontsize=12)
    ax.set_ylabel("Relative Depth (Near to Far)", fontsize=12)
    ax.set_title("Top-Down View (Bird's Eye View)\n"
                 "Each point = 1 detected object", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Near at top
    
    plt.tight_layout()
    plt.show()

# %%
if objects:
    h, w = img_rgb.shape[:2]
    create_topdown_view(objects, w, h)

# %% [markdown]
# ## Part 14: Summary and Exercises
#
# ### What We Learned in This Lab
#
# 1. **YOLO26** — Latest model from Ultralytics for real-time object detection
#    - NMS-Free, DFL Removed, Edge Optimized
#    - Supports Detection, Segmentation, Pose, OBB, Classification
#
# 2. **Object Distance Estimation (BBox Heuristic)** — NEW!
#    - Uses pinhole camera model: Distance = (Real Height x Focal Length) / BBox Height
#    - Gives **metric distance (meters)** — no extra model needed
#    - Good complement to MiDaS relative depth
#
# 3. **MiDaS** — Monocular Depth Estimation from Intel Labs
#    - Estimates relative depth from a single image
#    - 3 variants: DPT_Large (accurate), DPT_Hybrid (balanced), MiDaS_small (fast)
#
# 4. **Combining YOLO + Depth** — Pseudo-3D Object Detection
#    - Detect objects (2D) + estimate distance (Z-axis)
#    - Two complementary methods: MiDaS (relative) + BBox heuristic (metric)
#    - Applications: ADAS, Robotics, AR/VR, Surveillance
#
# ### Limitations
#
# - MiDaS gives **relative depth** — not absolute distance (meters)
# - BBox heuristic assumes known object sizes and estimated focal length
# - Accuracy depends on scene complexity and camera calibration
# - Additional calibration needed for precise metric depth
#
# ---
#
# ### Exercises
#
# **Exercise 1:** Change MiDaS model to `DPT_Large` and compare depth map
# with `MiDaS_small` — what differences do you observe?
#
# **Exercise 2:** Use a larger YOLO26 model such as `yolo26s.pt` or `yolo26m.pt`
# and compare number of detected objects.
#
# **Exercise 3:** Try changing method in `estimate_object_depth()` 
# from "median" to "mean" or "center" — how do results differ?
#
# **Exercise 4:** Use a different image with objects at multiple depth levels
# (e.g., street scene, indoor room) and analyze depth estimation quality.
#
# **Exercise 5 (Advanced):** Modify `compute_danger_score()` to include velocity
# estimation by comparing depth between frames in video.
#
# **Exercise 6 (New):** Calibrate the `ASSUMED_FOCAL_LENGTH_PX` value for your
# specific camera. Compare BBox distance estimates before and after calibration.
# How does focal length affect accuracy?
#
# **Exercise 7 (New):** Add more classes to `KNOWN_HEIGHTS_M` dictionary and
# test on images with those objects. Which classes give the most accurate
# BBox distance estimates?
#
# ---
#
# ### References
#
# - [Ultralytics YOLO26 Docs](https://docs.ultralytics.com/models/yolo26/)
# - [MiDaS GitHub](https://github.com/isl-org/MiDaS)
# - [MiDaS Paper: Towards Robust Monocular Depth Estimation (TPAMI 2022)](https://arxiv.org/abs/1907.01341)
# - [Ultralytics Depth Estimation Guide](https://www.ultralytics.com/glossary/depth-estimation)
# - [Pinhole Camera Model — OpenCV](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)