# %% [markdown]
# # 🔬 Lab: YOLO26 + Depth Estimation
# ## การรวม Object Detection กับ Monocular Depth Estimation
#
# **วัตถุประสงค์ (Objectives):**
# 1. เข้าใจหลักการ Depth Estimation (Monocular Depth Estimation)
# 2. ใช้งาน YOLO26 สำหรับ Object Detection
# 3. ใช้งาน MiDaS สำหรับ Depth Estimation
# 4. รวม YOLO26 + MiDaS เพื่อประมาณระยะห่างของวัตถุจากกล้อง
# 5. สร้าง Visualization แบบ 3D-aware
#
# **เครื่องมือที่ใช้ (Tools):**
# - Ultralytics YOLO26 (Object Detection)
# - Intel MiDaS (Monocular Depth Estimation)
# - OpenCV, Matplotlib, NumPy
#
# ---

# %% [markdown]
# ## Part 1: ทฤษฎี Depth Estimation
#
# ### Depth Estimation คืออะไร?
#
# **Depth Estimation** คือกระบวนการประมาณระยะห่างของแต่ละ pixel ในภาพจากกล้อง
# ผลลัพธ์คือ **Depth Map** ที่ค่าความเข้มของ pixel แสดงถึงระยะห่าง
#
# ### ประเภทของ Depth Estimation
#
# | ประเภท | คำอธิบาย | ข้อดี | ข้อเสีย |
# |--------|---------|------|--------|
# | **Stereo Vision** | ใช้กล้อง 2 ตัว คำนวณจาก disparity | แม่นยำ | ต้องใช้กล้อง 2 ตัว |
# | **LiDAR / ToF** | ใช้แสงเลเซอร์วัดระยะ | แม่นยำมาก | แพง, hardware เฉพาะ |
# | **Monocular Depth** | ใช้ภาพเดียว + Deep Learning | ใช้กล้องปกติได้ | เป็น relative depth |
#
# ### MiDaS (Multiple Depth from a Single Image)
#
# MiDaS เป็น model จาก Intel Labs ที่ใช้ Encoder-Decoder architecture
# - **Encoder**: ResNet / DPT (Dense Prediction Transformer) สำหรับ feature extraction
# - **Decoder**: Upsampling + Feature Fusion สร้าง depth map
# - ฝึกจาก 12+ datasets → ทำงานได้ดีกับภาพหลากหลายประเภท
#
# ### แนวคิด: YOLO + Depth Estimation
#
# ```
# ภาพ Input
#    ├── YOLO26 → Bounding Boxes + Class Labels (2D Detection)
#    └── MiDaS  → Depth Map (ระยะห่างแต่ละ pixel)
#         ↓
#    รวมกัน → วัตถุแต่ละชิ้น + ระยะห่างประมาณ (Pseudo-3D)
# ```

# %% [markdown]
# ## Part 2: ติดตั้ง Library

# %%
# ติดตั้ง library ที่จำเป็น
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
# YOLO26 เป็น model ล่าสุดจาก Ultralytics (2025) มีคุณสมบัติหลัก:
# - **NMS-Free**: ไม่ต้องใช้ Non-Maximum Suppression → inference เร็วขึ้น
# - **DFL Removed**: ลบ Distribution Focal Loss → deploy ง่ายขึ้น
# - **Edge Optimized**: CPU inference เร็วขึ้น 43%
# - รองรับ: Detection, Segmentation, Pose, OBB, Classification

# %%
from ultralytics import YOLO

# โหลด YOLO26 nano model (pretrained บน COCO dataset — 80 classes)
model = YOLO("yolo26n.pt")

# ทำ inference บนรูปภาพ
IMAGE_PATH = ".././images/football_teamplay.jpeg"
results = model(IMAGE_PATH, imgsz=640)

# %%
# แสดงผลลัพธ์ detection
result = results[0]

print("=" * 60)
print("📊 YOLO26 Detection Results")
print("=" * 60)
print(f"จำนวนวัตถุที่ตรวจพบ: {len(result.boxes)}")
print(f"ขนาดภาพ: {result.orig_shape}")
print(f"Classes ที่ตรวจพบ: {result.boxes.cls.unique().tolist()}")
print()

# แสดงรายละเอียดแต่ละ detection
for i, box in enumerate(result.boxes):
    cls_id = int(box.cls[0])
    cls_name = result.names[cls_id]
    conf = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"  [{i}] {cls_name:15s} | conf: {conf:.2f} | bbox: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

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
# ## Part 4: MiDaS Depth Estimation
#
# ### MiDaS Model Variants
#
# | Model | ขนาด | ความแม่นยำ | ความเร็ว |
# |-------|------|-----------|---------|
# | **DPT_Large** | ใหญ่ | สูงสุด | ช้าสุด |
# | **DPT_Hybrid** | กลาง | ปานกลาง | ปานกลาง |
# | **MiDaS_small** | เล็ก | ต่ำสุด | เร็วสุด |
#
# > ⚠️ MiDaS ให้ **relative depth** (ค่าสัมพัทธ์) ไม่ใช่ absolute depth (หน่วยเมตร)
# > ค่าสูง = ไกลจากกล้อง, ค่าต่ำ = ใกล้กล้อง (inverse depth)

# %%
# โหลด MiDaS model จาก PyTorch Hub
# เลือก model_type: "DPT_Large", "DPT_Hybrid", "MiDaS_small"

model_type = "MiDaS_small"  # เริ่มจาก small model (เร็ว, ใช้ RAM น้อย)

print(f"🔄 กำลังโหลด MiDaS model: {model_type}...")
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()
print(f"✅ โหลด MiDaS สำเร็จ! (device: {device})")

# โหลด transform สำหรับ preprocessing
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

print(f"✅ โหลด transform สำเร็จ!")

# %%
# อ่านภาพและทำ Depth Estimation
img_bgr = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Preprocessing: transform ภาพ
input_batch = transform(img_rgb).to(device)

print(f"📐 Input shape: {input_batch.shape}")

# Inference
with torch.no_grad():
    prediction = midas(input_batch)
    
    # Resize depth map ให้ตรงกับขนาดภาพต้นฉบับ
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()

print(f"✅ Depth map shape: {depth_map.shape}")
print(f"📊 Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")

# %%
# Visualize Depth Map
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ภาพต้นฉบับ
axes[0].imshow(img_rgb)
axes[0].set_title("Original Image", fontsize=14)
axes[0].axis("off")

# Depth Map (Viridis colormap)
im1 = axes[1].imshow(depth_map, cmap="inferno")
axes[1].set_title("Depth Map (Inferno)", fontsize=14)
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Relative Depth")

# Depth Map (Plasma colormap — inverted)
depth_inv = depth_map.max() - depth_map  # invert: ใกล้ = สว่าง
im2 = axes[2].imshow(depth_inv, cmap="plasma")
axes[2].set_title("Inverted Depth (ใกล้ = สว่าง)", fontsize=14)
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Inverted Depth")

plt.suptitle("MiDaS Monocular Depth Estimation", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 5: รวม YOLO26 + Depth Estimation
#
# ### แนวคิด
# 1. ใช้ YOLO26 ตรวจจับวัตถุ → ได้ bounding box
# 2. ใช้ MiDaS สร้าง depth map
# 3. สำหรับแต่ละ bounding box → crop ส่วนของ depth map → คำนวณค่าเฉลี่ย
# 4. ค่า depth เฉลี่ย = ระยะห่างสัมพัทธ์ของวัตถุจากกล้อง
# 5. จัดอันดับ: วัตถุไหนใกล้/ไกลที่สุด

# %%
def estimate_object_depth(boxes, depth_map, names, method="median"):
    """
    คำนวณ depth ของแต่ละ detected object
    
    Parameters:
    -----------
    boxes : ultralytics Boxes object
    depth_map : numpy array — depth map จาก MiDaS
    names : dict — class names mapping
    method : str — วิธีคำนวณ depth ("mean", "median", "center")
    
    Returns:
    --------
    list of dict — ข้อมูลแต่ละ object พร้อม depth
    """
    objects_with_depth = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        conf = float(box.conf[0])
        
        # ตรวจสอบขอบเขต
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1], x2)
        y2 = min(depth_map.shape[0], y2)
        
        # Crop depth region ตาม bounding box
        depth_region = depth_map[y1:y2, x1:x2]
        
        if depth_region.size == 0:
            continue
        
        # คำนวณ depth ตาม method ที่เลือก
        if method == "mean":
            obj_depth = np.mean(depth_region)
        elif method == "median":
            obj_depth = np.median(depth_region)
        elif method == "center":
            # ใช้ค่า depth ที่จุดกลาง bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            obj_depth = depth_map[cy, cx]
        else:
            obj_depth = np.mean(depth_region)
        
        objects_with_depth.append({
            "id": i,
            "class": cls_name,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
            "depth_value": obj_depth,
            "depth_std": np.std(depth_region),
            "bbox_area": (x2 - x1) * (y2 - y1),
        })
    
    # จัดเรียงตาม depth (MiDaS: ค่ามาก = ไกล)
    objects_with_depth.sort(key=lambda x: x["depth_value"])
    
    return objects_with_depth

# %%
# คำนวณ depth ของแต่ละ object
objects = estimate_object_depth(result.boxes, depth_map, result.names, method="median")

# Normalize depth เป็น 0-100 สำหรับแสดงผล
depth_values = [obj["depth_value"] for obj in objects]
if depth_values:
    d_min, d_max = min(depth_values), max(depth_values)
    for obj in objects:
        if d_max > d_min:
            obj["depth_normalized"] = ((obj["depth_value"] - d_min) / (d_max - d_min)) * 100
        else:
            obj["depth_normalized"] = 50.0

# แสดงผลลัพธ์
print("=" * 80)
print("📊 YOLO26 + Depth Estimation Results")
print("=" * 80)
print(f"{'#':>3} {'Class':>12} {'Conf':>6} {'Depth':>10} {'Norm':>8} {'Rank':>6}")
print("-" * 80)

for rank, obj in enumerate(objects, 1):
    distance_label = "🟢 ใกล้" if obj["depth_normalized"] < 33 else "🟡 กลาง" if obj["depth_normalized"] < 66 else "🔴 ไกล"
    print(f"{obj['id']:>3} {obj['class']:>12} {obj['confidence']:>6.2f} "
          f"{obj['depth_value']:>10.2f} {obj['depth_normalized']:>7.1f}% "
          f"{distance_label}")

# %% [markdown]
# ## Part 6: Visualization — Annotated Image with Depth
#
# สร้างภาพที่แสดงทั้ง bounding box และระยะห่างของแต่ละวัตถุ

# %%
def draw_depth_annotated_image(image_rgb, objects, depth_map):
    """
    วาด bounding box พร้อม depth annotation บนภาพ
    สีของ box จะเปลี่ยนตาม depth (ใกล้ = เขียว, ไกล = แดง)
    """
    img_annotated = image_rgb.copy()
    
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        norm = obj["depth_normalized"] / 100.0  # 0 = ใกล้, 1 = ไกล
        
        # สี: ใกล้ = เขียว (0,255,0), ไกล = แดง (255,0,0)
        r = int(255 * norm)
        g = int(255 * (1 - norm))
        b = 0
        color = (r, g, b)
        
        # วาด bounding box
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
        
        # Label: class + depth
        label = f"{obj['class']} | D:{obj['depth_normalized']:.0f}%"
        
        # Background สำหรับ text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        
        # Text สีขาว
        cv2.putText(img_annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img_annotated

# %%
# สร้างภาพ annotated
annotated = draw_depth_annotated_image(img_rgb, objects, depth_map)

# แสดงผล
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes[0].imshow(annotated)
axes[0].set_title("YOLO26 Detection + Depth Annotation\n(🟢 ใกล้ → 🔴 ไกล)", fontsize=14)
axes[0].axis("off")

# Depth Map overlay
depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
axes[1].imshow(img_rgb, alpha=0.5)
axes[1].imshow(depth_normalized, cmap="inferno", alpha=0.5)
axes[1].set_title("Original + Depth Map Overlay", fontsize=14)
axes[1].axis("off")

plt.suptitle("YOLO26 + MiDaS Depth Estimation", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 7: Depth-based Object Sorting & Visualization
#
# จัดกลุ่มวัตถุตามระยะห่าง: ใกล้ (Near), กลาง (Mid), ไกล (Far)

# %%
def categorize_by_depth(objects):
    """แบ่งวัตถุเป็น 3 กลุ่มตาม depth"""
    near = [o for o in objects if o["depth_normalized"] < 33]
    mid  = [o for o in objects if 33 <= o["depth_normalized"] < 66]
    far  = [o for o in objects if o["depth_normalized"] >= 66]
    return near, mid, far

near_objects, mid_objects, far_objects = categorize_by_depth(objects)

print("=" * 60)
print("📏 Object Distance Categorization")
print("=" * 60)

print(f"\n🟢 NEAR (ใกล้) — {len(near_objects)} objects:")
for o in near_objects:
    print(f"   • {o['class']} (conf: {o['confidence']:.2f}, depth: {o['depth_normalized']:.1f}%)")

print(f"\n🟡 MID (กลาง) — {len(mid_objects)} objects:")
for o in mid_objects:
    print(f"   • {o['class']} (conf: {o['confidence']:.2f}, depth: {o['depth_normalized']:.1f}%)")

print(f"\n🔴 FAR (ไกล) — {len(far_objects)} objects:")
for o in far_objects:
    print(f"   • {o['class']} (conf: {o['confidence']:.2f}, depth: {o['depth_normalized']:.1f}%)")

# %%
# Bar Chart: Depth ของแต่ละ Object
if objects:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    labels = [f"{o['class']}_{o['id']}" for o in objects]
    depths = [o["depth_normalized"] for o in objects]
    colors = [plt.cm.RdYlGn_r(d / 100.0) for d in depths]
    
    bars = axes[0].barh(labels, depths, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Relative Depth (%)", fontsize=12)
    axes[0].set_title("Object Depth Ranking\n(0% = ใกล้สุด, 100% = ไกลสุด)", fontsize=14)
    axes[0].axvline(x=33, color="green", linestyle="--", alpha=0.5, label="Near/Mid boundary")
    axes[0].axvline(x=66, color="red", linestyle="--", alpha=0.5, label="Mid/Far boundary")
    axes[0].legend()
    axes[0].set_xlim(0, 105)
    
    # Scatter plot: BBox Area vs Depth
    areas = [o["bbox_area"] for o in objects]
    axes[1].scatter(depths, areas, c=colors, s=100, edgecolors="black", linewidth=0.5)
    for o in objects:
        axes[1].annotate(f"{o['class']}_{o['id']}", 
                        (o["depth_normalized"], o["bbox_area"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[1].set_xlabel("Relative Depth (%)", fontsize=12)
    axes[1].set_ylabel("Bounding Box Area (pixels²)", fontsize=12)
    axes[1].set_title("BBox Area vs Depth\n(วัตถุใกล้มักมี bbox ใหญ่กว่า)", fontsize=14)
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Part 8: Depth Estimation บน Video (Frame-by-Frame)
#
# ประยุกต์ใช้กับ video โดยทำ detection + depth estimation ทีละ frame

# %%
def process_video_with_depth(video_path, yolo_model, midas_model, midas_transform,
                              device, max_frames=30, conf_threshold=0.4):
    """
    ประมวลผล video: YOLO26 detection + MiDaS depth estimation
    
    Parameters:
    -----------
    video_path : str — path ไปยัง video file
    yolo_model : YOLO model
    midas_model : MiDaS model
    midas_transform : MiDaS preprocessing transform
    device : torch.device
    max_frames : int — จำนวน frame สูงสุดที่จะประมวลผล
    conf_threshold : float — confidence threshold สำหรับ YOLO
    
    Returns:
    --------
    list of dict — ผลลัพธ์แต่ละ frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ไม่สามารถเปิด video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 Video: {total_frames} frames, {fps:.1f} FPS")
    print(f"📊 ประมวลผล {min(max_frames, total_frames)} frames...")
    
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
        
        # คำนวณ depth ของแต่ละ object
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
            print(f"  ✅ Frame {frame_count}/{min(max_frames, total_frames)}")
    
    cap.release()
    print(f"🎬 ประมวลผลเสร็จ: {frame_count} frames")
    return frame_results

# %%
# ตัวอย่างการใช้งานกับ video (uncomment เมื่อมี video file)
# VIDEO_PATH = ".././videos/sample.mp4"
# 
# video_results = process_video_with_depth(
#     VIDEO_PATH, model, midas, transform, device,
#     max_frames=30, conf_threshold=0.4
# )
# 
# # แสดงผล frame ตัวอย่าง
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

print("📝 Video processing function พร้อมใช้งาน!")
print("   Uncomment code ด้านบนเมื่อมี video file")

# %% [markdown]
# ## Part 9: เปรียบเทียบ MiDaS Models
#
# เปรียบเทียบ MiDaS 3 รุ่น: DPT_Large, DPT_Hybrid, MiDaS_small

# %%
import time

def compare_midas_models(image_path, model_types=None):
    """
    เปรียบเทียบ MiDaS models ต่างๆ
    """
    if model_types is None:
        model_types = ["MiDaS_small"]  # ใช้แค่ small เพื่อความเร็ว
        # model_types = ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]  # full comparison
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results_compare = {}
    
    for mt in model_types:
        print(f"\n🔄 Loading {mt}...")
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
        print(f"  ✅ {mt}: {elapsed*1000:.1f}ms/frame")
        
        # ลบ model ออกจาก memory
        del m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results_compare

# %%
# เปรียบเทียบ (ใช้แค่ small model เพื่อความเร็ว)
comparison = compare_midas_models(IMAGE_PATH, model_types=["MiDaS_small"])

# Uncomment เพื่อเปรียบเทียบทุก model (ใช้ RAM มากขึ้น):
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
# ## Part 10: Advanced — Depth-Aware Object Priority
#
# ### Use Case: ระบบ ADAS (Advanced Driver Assistance System)
# วัตถุที่ใกล้กว่า → ความเสี่ยงสูงกว่า → ต้องเตือนก่อน

# %%
def compute_danger_score(obj, weights=None):
    """
    คำนวณ danger score ของ object (สำหรับ ADAS scenario)
    
    วัตถุที่ใกล้ + ใหญ่ + เป็นคน → อันตรายมาก
    
    Parameters:
    -----------
    obj : dict — object info จาก estimate_object_depth
    weights : dict — น้ำหนักสำหรับแต่ละ factor
    
    Returns:
    --------
    float — danger score (0-100)
    """
    if weights is None:
        weights = {"proximity": 0.5, "size": 0.2, "class_risk": 0.3}
    
    # Proximity score (ใกล้ = อันตรายมาก) — invert depth
    proximity = 100 - obj["depth_normalized"]
    
    # Size score (ใหญ่ = อันตรายมาก)
    max_area = 500 * 500  # กำหนดค่าสูงสุดสำหรับ normalize
    size_score = min(100, (obj["bbox_area"] / max_area) * 100)
    
    # Class risk (บาง class อันตรายมากกว่า)
    high_risk_classes = {"person": 100, "bicycle": 80, "motorcycle": 80, "car": 70, 
                         "bus": 60, "truck": 60, "dog": 70, "cat": 50}
    class_risk = high_risk_classes.get(obj["class"], 30)
    
    # Weighted sum
    danger = (weights["proximity"] * proximity +
              weights["size"] * size_score +
              weights["class_risk"] * class_risk)
    
    return min(100, danger)

# %%
# คำนวณ danger score
if objects:
    for obj in objects:
        obj["danger_score"] = compute_danger_score(obj)
    
    # จัดเรียงตาม danger score (สูง = อันตรายมาก)
    objects_by_danger = sorted(objects, key=lambda x: x["danger_score"], reverse=True)
    
    print("=" * 70)
    print("⚠️  DANGER SCORE RANKING (ADAS Scenario)")
    print("=" * 70)
    print(f"{'Rank':>4} {'Class':>12} {'Depth%':>8} {'Area':>8} {'Danger':>8} {'Alert':>10}")
    print("-" * 70)
    
    for rank, obj in enumerate(objects_by_danger, 1):
        danger = obj["danger_score"]
        if danger >= 70:
            alert = "🚨 HIGH"
        elif danger >= 40:
            alert = "⚠️ MEDIUM"
        else:
            alert = "✅ LOW"
        
        print(f"{rank:>4} {obj['class']:>12} {obj['depth_normalized']:>7.1f}% "
              f"{obj['bbox_area']:>8} {danger:>7.1f} {alert:>10}")

# %% [markdown]
# ## Part 11: สร้าง Top-Down View (Bird's Eye View)
#
# แสดงตำแหน่งวัตถุแบบมองจากด้านบน (แผนผัง)

# %%
def create_topdown_view(objects, img_width, img_height, figsize=(8, 10)):
    """
    สร้าง top-down view แสดงตำแหน่ง x (ซ้าย-ขวา) vs depth (ใกล้-ไกล)
    """
    if not objects:
        print("ไม่มี object ให้แสดง")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        cx = (x1 + x2) / 2  # center x
        depth_pct = obj["depth_normalized"]
        danger = obj.get("danger_score", 50)
        
        # Normalize x position (0-100)
        x_norm = (cx / img_width) * 100
        
        # สีตาม danger score
        color = plt.cm.RdYlGn_r(danger / 100)
        
        # ขนาด marker ตาม bbox area
        marker_size = max(50, min(500, obj["bbox_area"] / 100))
        
        ax.scatter(x_norm, depth_pct, s=marker_size, c=[color], 
                  edgecolors="black", linewidth=1, zorder=5, alpha=0.8)
        ax.annotate(f"{obj['class']}\n({danger:.0f})", 
                   (x_norm, depth_pct),
                   textcoords="offset points", xytext=(10, 5),
                   fontsize=8, ha="left")
    
    # กล้อง (ตำแหน่งด้านล่าง)
    ax.scatter(50, -5, s=200, c="blue", marker="^", zorder=10, label="📷 Camera")
    ax.annotate("📷 Camera", (50, -5), textcoords="offset points", 
               xytext=(0, -15), fontsize=10, ha="center", fontweight="bold")
    
    # แบ่งโซน
    ax.axhspan(-10, 33, alpha=0.1, color="green", label="Near Zone")
    ax.axhspan(33, 66, alpha=0.1, color="yellow", label="Mid Zone")
    ax.axhspan(66, 110, alpha=0.1, color="red", label="Far Zone")
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(-10, 110)
    ax.set_xlabel("Horizontal Position (Left → Right)", fontsize=12)
    ax.set_ylabel("Relative Depth (Near → Far)", fontsize=12)
    ax.set_title("🗺️ Top-Down View (Bird's Eye View)\nวัตถุแต่ละจุด = 1 detected object", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # ใกล้อยู่ด้านบน
    
    plt.tight_layout()
    plt.show()

# %%
if objects:
    h, w = img_rgb.shape[:2]
    create_topdown_view(objects, w, h)

# %% [markdown]
# ## Part 12: สรุปผลและแบบฝึกหัด
#
# ### สิ่งที่เรียนรู้ในแล็บนี้
#
# 1. **YOLO26** — model ล่าสุดจาก Ultralytics สำหรับ real-time object detection
#    - NMS-Free, DFL Removed, Edge Optimized
#    - รองรับ Detection, Segmentation, Pose, OBB, Classification
#
# 2. **MiDaS** — Monocular Depth Estimation จาก Intel Labs
#    - ประมาณ relative depth จากภาพเดียว
#    - 3 รุ่น: DPT_Large (แม่นยำ), DPT_Hybrid (สมดุล), MiDaS_small (เร็ว)
#
# 3. **การรวม YOLO + Depth** — Pseudo-3D Object Detection
#    - ตรวจจับวัตถุ (2D) + ประมาณระยะห่าง (Z-axis)
#    - ประยุกต์ใช้: ADAS, Robotics, AR/VR, Surveillance
#
# ### ข้อจำกัด (Limitations)
#
# - MiDaS ให้ **relative depth** ไม่ใช่ absolute distance (เมตร)
# - ความแม่นยำขึ้นอยู่กับ scene complexity
# - ต้องใช้ calibration เพิ่มเติมสำหรับ metric depth
#
# ---
#
# ### 📝 แบบฝึกหัด (Exercises)
#
# **Exercise 1:** เปลี่ยน MiDaS model เป็น `DPT_Large` แล้วเปรียบเทียบ depth map
# กับ `MiDaS_small` — ความแตกต่างเป็นอย่างไร?
#
# **Exercise 2:** ใช้ YOLO26 model ขนาดใหญ่ขึ้น เช่น `yolo26s.pt` หรือ `yolo26m.pt`
# แล้วเปรียบเทียบจำนวนวัตถุที่ตรวจพบ
#
# **Exercise 3:** ทดลองเปลี่ยน method ใน `estimate_object_depth()` 
# จาก "median" เป็น "mean" หรือ "center" — ผลลัพธ์ต่างกันอย่างไร?
#
# **Exercise 4:** ใช้ภาพอื่นที่มีวัตถุหลายระดับความลึก (เช่น ภาพถนน, ภาพห้อง)
# แล้ววิเคราะห์ว่า depth estimation ทำงานได้ดีแค่ไหน
#
# **Exercise 5 (Advanced):** ปรับ `compute_danger_score()` ให้รวม velocity estimation
# โดยเปรียบเทียบ depth ระหว่าง frames ใน video
#
# ---
#
# ### 📚 References
#
# - [Ultralytics YOLO26 Docs](https://docs.ultralytics.com/models/yolo26/)
# - [MiDaS GitHub](https://github.com/isl-org/MiDaS)
# - [MiDaS Paper: Towards Robust Monocular Depth Estimation (TPAMI 2022)](https://arxiv.org/abs/1907.01341)
# - [Ultralytics Depth Estimation Guide](https://www.ultralytics.com/glossary/depth-estimation)