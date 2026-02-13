# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🔬 Lab: YOLO26 — Real-Time Computer Vision Inference (Reorganized)
#
# **วัตถุประสงค์ (Objectives):**
# - เข้าใจหลักการทำงานของ YOLO26 และความแตกต่างจาก YOLO11
# - ฝึกใช้งาน YOLO26 สำหรับ tasks ต่างๆ: Detection, Segmentation, Pose Estimation, Classification
# - ปรับแต่ง parameters ต่างๆ เพื่อเข้าใจ speed-accuracy tradeoff
# - ประยุกต์ใช้ Segmentation Mask สำหรับ Background Removal, Blur Object
# - ประมาณระยะห่างระหว่างวัตถุ (Object Distance Estimation)
# - ทดลอง Object Tracking บน video
#
# **เครื่องมือที่ใช้ (Tools):** Python, Ultralytics, OpenCV, Matplotlib, NumPy
#
# ---
#
# ## 📖 Background: YOLO26 คืออะไร?
#
# **YOLO26** (Released September 2025) เป็น YOLO รุ่นล่าสุดจาก Ultralytics
#
# | Feature | YOLO11 (2024) | YOLO26 (2025) |
# |---------|---------------|---------------|
# | NMS | ต้องใช้ NMS post-processing | **NMS-Free** end-to-end |
# | DFL | ใช้ DFL | **ไม่ใช้ DFL** — ลดความซับซ้อน |
# | CPU Speed | Baseline | **เร็วขึ้น ~43%** |
# | Optimizer | SGD / Adam | **MuSGD** (Hybrid SGD + Muon) |
# | Head | Single head | **Dual-head** (One-to-One / One-to-Many) |
#
# **Model Variants:** `yolo26n` (Nano), `yolo26s` (Small), `yolo26m` (Medium), `yolo26l` (Large), `yolo26x` (Extra Large)
#
# ---
#
# ## 📚 Lab Structure (จัดกลุ่มตามเทคโนโลยี)
#
# | Group | หัวข้อ | Labs |
# |-------|-------|------|
# | **Group 1** | Setup & Detection Basics | Lab 1–4 |
# | **Group 2** | Model Tuning & Optimization | Lab 5–8 |
# | **Group 3** | Detection Applications | Lab 9–11 |
# | **Group 4** | Segmentation & Mask Applications | Lab 12–15 |
# | **Group 5** | Pose Estimation | Lab 16–18 |
# | **Group 6** | Image Classification | Lab 19 |
# | **Group 7** | Distance Estimation & Tracking | Lab 20–23 |

# %% [markdown]
# # ═══════════════════════════════════════════════════════════
# # Group 1: Setup & Detection Basics (Lab 1–4)
# # ═══════════════════════════════════════════════════════════

# %% [markdown]
# ## Lab 1: Installation & Setup
# ติดตั้ง library ที่จำเป็น

# %%
import IPython
import sys

def clean_notebook():
    IPython.display.clear_output(wait=True)
    print("✅ Notebook cleaned. Ready to go!")

# ติดตั้ง packages (uncomment ถ้ายังไม่ได้ติดตั้ง)
# !uv pip install ultralytics
clean_notebook()

# %%
# ตรวจสอบ Ultralytics version
import ultralytics
ultralytics.checks()

# %% [markdown]
# ---
# ## Lab 2: Basic Object Detection + Results Object
#
# เริ่มจากการใช้ YOLO26 ทำ Object Detection ซึ่งเป็น task พื้นฐานที่สุด
# โมเดลจะ predict **bounding box** + **class label** + **confidence score**
#
# ### 2.1 Basic Detection with YOLO26

# %%
from ultralytics import YOLO

# โหลด YOLO26 nano model (pretrained บน COCO dataset — 80 classes)
model = YOLO("yolo26n.pt")

# ทำ inference บนรูปภาพ
IMAGE_PATH = ".././images/football_teamplay.jpeg"
results = model(IMAGE_PATH, imgsz=640)

# แสดงผลลัพธ์
results[0].show()

# %% [markdown]
# ### 2.2 ทำความเข้าใจ Results Object
#
# ผลลัพธ์จาก YOLO จะอยู่ใน `Results` object ที่มีข้อมูลหลายส่วน

# %%

# %%
# สำรวจ result object
result = results[0]

print("=" * 60)
print("📊 Detection Results Summary")
print("=" * 60)
print(f"จำนวนวัตถุที่ตรวจจับได้ (Objects detected): {len(result.boxes)}")
print(f"Original image shape: {result.orig_shape}")
print(f"Model speed: {result.speed}")
print()

# แสดงรายละเอียดของแต่ละ detection
for i, box in enumerate(result.boxes):
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    print(f"  Object {i+1}: {class_name} (conf: {confidence:.2f}) | bbox: ({x1}, {y1}, {x2}, {y2})")

# %% [markdown]
# ### 🔍 สังเกต:
# - `box.xyxy` — พิกัด bounding box ในรูปแบบ (x1, y1, x2, y2)
# - `box.conf` — ค่า confidence (0-1) ยิ่งสูงยิ่งมั่นใจ
# - `box.cls` — class ID ที่ตรวจจับได้
# - `model.names` — mapping จาก class ID เป็นชื่อ class

# %% [markdown]
# ---
# ## Lab 3: Custom Visualization with OpenCV
#
# ใช้ OpenCV วาด bounding box เอง เพื่อ customize การแสดงผล
# เช่น เปลี่ยนสี, เพิ่มข้อมูล, ปรับ font

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# โหลดรูปภาพ
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ทำ inference
model = YOLO("yolo26n.pt")
results = model(IMAGE_PATH)

# กำหนดสีสำหรับแต่ละ class (สุ่มสี)
np.random.seed(42)
colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(80)}

# วาด bounding box
annotated = image_rgb.copy()
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])
    color = colors[class_id]
    
    # วาดกรอบ
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    
    # วาด label background
    label = f"{class_name} {confidence:.2f}"
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
    cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

plt.figure(figsize=(15, 8))
plt.imshow(annotated)
plt.title("YOLO26 Object Detection — Custom Visualization", fontsize=14)
plt.axis('off')
plt.show()

# %% [markdown]
# ---
# ## Lab 4: Built-in plot() Method — วิธีง่ายที่สุด
#
# Ultralytics มี built-in `plot()` method ที่สะดวกมาก

# %%
from PIL import Image

# plot() คืน numpy array (BGR) ที่วาด annotation แล้ว
annotated_bgr = results[0].plot()

# แปลงเป็น RGB สำหรับ matplotlib
annotated_rgb = annotated_bgr[..., ::-1]

plt.figure(figsize=(15, 8))
plt.imshow(annotated_rgb)
plt.title("YOLO26 Detection — Using plot() method", fontsize=14)
plt.axis('off')
plt.show()

# %% [markdown]
# ---
# ## Lab 5: Model Size Comparison — Speed vs Accuracy Tradeoff
#
# YOLO26 มีหลายขนาด มาเปรียบเทียบกันว่า nano, small, medium ให้ผลต่างกันอย่างไร

# %%
import time

model_names = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"]
model_results = {}

for name in model_names:
    print(f"\n🔄 Loading {name}...")
    m = YOLO(name)
    
    # Warm-up run
    _ = m(IMAGE_PATH, verbose=False)
    
    # Timed run
    start = time.time()
    res = m(IMAGE_PATH, verbose=False)
    elapsed = time.time() - start
    
    num_detections = len(res[0].boxes)
    model_results[name] = {
        "detections": num_detections,
        "time_ms": elapsed * 1000,
        "result": res[0]
    }
    print(f"  ✅ {name}: {num_detections} objects detected in {elapsed*1000:.1f} ms")

# %% [markdown]
# ### แสดงผลเปรียบเทียบ

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, (name, data) in enumerate(model_results.items()):
    annotated = data["result"].plot()[..., ::-1]
    axes[idx].imshow(annotated)
    axes[idx].set_title(f"{name}\n{data['detections']} objects | {data['time_ms']:.1f} ms", fontsize=12)
    axes[idx].axis('off')

plt.suptitle("YOLO26 Model Size Comparison: Nano vs Small vs Medium", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 💡 สิ่งที่น่าสังเกต:
# - **Nano** เร็วที่สุดแต่อาจพลาดวัตถุเล็กๆ
# - **Small** สมดุลดี เหมาะกับงานส่วนใหญ่
# - **Medium** ตรวจจับได้มากที่สุดแต่ช้ากว่า
# - ในการใช้งานจริง ต้องเลือกให้เหมาะกับ hardware และ latency requirement

# %% [markdown]
# ---
# ## Lab 6: Confidence Threshold — ผลกระทบของค่า Confidence
#
# ลองปรับ `conf` threshold เพื่อดูว่าส่งผลต่อ detection อย่างไร

# %%
conf_thresholds = [0.5, 0.75, 0.90, 0.91]
model = YOLO("yolo26n.pt")

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

for idx, conf in enumerate(conf_thresholds):
    res = model(IMAGE_PATH, conf=conf, verbose=False)
    annotated = res[0].plot()[..., ::-1]
    
    axes[idx].imshow(annotated)
    axes[idx].set_title(f"conf={conf}\n{len(res[0].boxes)} detections", fontsize=12)
    axes[idx].axis('off')

plt.suptitle("Effect of Confidence Threshold on Detection Results", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 💡 Key Insight:
# - **conf ต่ำ (0.1):** ตรวจจับได้มาก แต่มี False Positive สูง
# - **conf สูง (0.75):** ตรวจจับน้อย แต่แม่นยำมาก
# - ในทางปฏิบัติ ค่า default = 0.25 เป็นจุดสมดุลที่ดี

# %% [markdown]
# ---
# ## Lab 7: Image Size Effects — ปรับขนาดภาพ Inference
#
# ลองปรับ `imgsz` เพื่อดูผลกระทบต่อ detection

# %%
model = YOLO("yolo26n.pt")

# ลองปรับ image size
img_sizes = [320, 480, 640]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, imgsz in enumerate(img_sizes):
    start = time.time()
    res = model(IMAGE_PATH, imgsz=imgsz, verbose=False)
    elapsed = (time.time() - start) * 1000
    
    annotated = res[0].plot()[..., ::-1]
    axes[idx].imshow(annotated)
    axes[idx].set_title(f"imgsz={imgsz}\n{len(res[0].boxes)} detections | {elapsed:.0f} ms", fontsize=12)
    axes[idx].axis('off')

plt.suptitle("Effect of Image Size on Detection", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 💡 Image Size Effect:
# - **imgsz เล็ก (320):** เร็วมาก แต่อาจพลาดวัตถุเล็ก
# - **imgsz ใหญ่ (640):** ตรวจจับได้ดีกว่า แต่ใช้เวลามากขึ้น
# - YOLO26 จะ resize รูปให้เป็น square ก่อน inference

# %% [markdown]
# ---
# ## Lab 8: Dual-Head Architecture — NMS-Free vs NMS Inference
#
# YOLO26 มีสถาปัตยกรรม **Dual-Head** ที่ unique:
# - **One-to-One Head (default):** End-to-end, ไม่ต้อง NMS → เร็วกว่า
# - **One-to-Many Head:** ใช้ NMS แบบดั้งเดิม → accuracy สูงกว่าเล็กน้อย

# %%
model = YOLO("yolo26n.pt")

# One-to-One Head (default) — NMS-Free
results_e2e = model(IMAGE_PATH, verbose=False)  # end2end=True is default

# One-to-Many Head — with NMS
results_nms = model(IMAGE_PATH, end2end=False, verbose=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

annotated_e2e = results_e2e[0].plot()[..., ::-1]
axes[0].imshow(annotated_e2e)
axes[0].set_title(f"One-to-One (NMS-Free)\n{len(results_e2e[0].boxes)} detections", fontsize=13)
axes[0].axis('off')

annotated_nms = results_nms[0].plot()[..., ::-1]
axes[1].imshow(annotated_nms)
axes[1].set_title(f"One-to-Many (with NMS)\n{len(results_nms[0].boxes)} detections", fontsize=13)
axes[1].axis('off')

plt.suptitle("YOLO26 Dual-Head Comparison", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n📊 Comparison:")
print(f"  One-to-One (NMS-Free): {len(results_e2e[0].boxes)} detections")
print(f"  One-to-Many (with NMS): {len(results_nms[0].boxes)} detections")

# %% [markdown]
# ### 💡 เมื่อไหร่ใช้ Head ไหน:
# - **One-to-One:** ใช้ตอน deploy จริง → เร็วกว่า, ไม่ต้องตั้งค่า NMS
# - **One-to-Many:** ใช้ตอนต้องการ accuracy สูงสุด → ดีกว่าเล็กน้อยในบาง scenario
#

# %% [markdown]
# ---
# ## Lab 9: Object Counting & Class Filtering
#
# ประยุกต์ใช้ YOLO26 ในการนับจำนวนวัตถุแยกตามประเภท

# %%
from collections import Counter

model = YOLO("yolo26s.pt")
results = model(IMAGE_PATH, verbose=False)

# นับจำนวนวัตถุแต่ละ class
class_counts = Counter()
for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    class_counts[class_name] += 1

print("📊 Object Count by Class:")
print("=" * 40)
for name, count in class_counts.most_common():
    print(f"  {name:>15}: {count} {'📦' * count}")

print(f"\n  {'Total':>15}: {sum(class_counts.values())}")

# %% [markdown]
# ### 9.1 Filter เฉพาะ class ที่ต้องการ
#
# ใช้ `classes` parameter เพื่อ detect เฉพาะ class ที่สนใจ

# %%
# COCO class IDs: 0=person, 32=sports ball
# ตรวจจับเฉพาะคนและลูกบอล
results_filtered = model(IMAGE_PATH, classes=[0,32], verbose=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# All classes
annotated_all = results[0].plot()[..., ::-1]
axes[0].imshow(annotated_all)
axes[0].set_title(f"All Classes ({len(results[0].boxes)} objects)", fontsize=13)
axes[0].axis('off')

# Filtered classes
annotated_filtered = results_filtered[0].plot()[..., ::-1]
axes[1].imshow(annotated_filtered)
axes[1].set_title(f"Person + Sports Ball Only ({len(results_filtered[0].boxes)} objects)", fontsize=13)
axes[1].axis('off')

plt.suptitle("Class Filtering — Detect Only What You Need", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Lab 10: Batch Inference — ทำ Inference หลายรูปพร้อมกัน
#
# YOLO26 รองรับการทำ inference หลายรูปในครั้งเดียว

# %%
model = YOLO("yolo26n.pt")

# ทำ inference บนรูปเดียวกันหลายครั้ง (ในจริงจะใช้หลายรูป)
results = model([IMAGE_PATH, IMAGE_PATH], verbose=False)

print(f"📸 Number of images processed: {len(results)}")
for i, res in enumerate(results):
    print(f"  Image {i+1}: {len(res.boxes)} objects detected")

# %% [markdown]
# ---
# ## Lab 11: Save Results — บันทึกผลลัพธ์
#
# บันทึกภาพที่ annotate แล้วลงไฟล์

# %%
model = YOLO("yolo26n.pt")
results = model(IMAGE_PATH, verbose=False)

# วิธีที่ 1: ใช้ save() method
results[0].save(filename="detection_result.jpg")
print("✅ Saved: detection_result.jpg")

# วิธีที่ 2: บันทึกจาก plot()
annotated = results[0].plot()
cv2.imwrite("detection_result_v2.jpg", annotated)
print("✅ Saved: detection_result_v2.jpg")

# วิธีที่ 3: บันทึก crop ของแต่ละวัตถุ
results[0].save_crop(save_dir="crops/")
print("✅ Saved crops to: crops/")

# %% [markdown]
# ---
# ## Lab 12: Instance Segmentation — พื้นฐาน
#
# Segmentation ไม่เพียงบอกตำแหน่ง แต่ยังบอก **pixel mask** ของแต่ละวัตถุ
#
# ### 🧠 ทำไม Segmentation สำคัญ?
# Segmentation mask คือพื้นฐานของหลาย application:
# - **Background Removal** — ลบพื้นหลังออก (Lab 14)
# - **Object Blur** — เบลอวัตถุที่เลือก (Lab 15)
# - **Video Effects** — เปลี่ยน background แบบ real-time
# - **Medical Imaging** — แยกอวัยวะ/เนื้องอก
# - **Autonomous Driving** — แยก road, car, pedestrian

# %%
# โหลด segmentation model
seg_model = YOLO("yolo26n-seg.pt")

# ทำ inference
results = seg_model(IMAGE_PATH, imgsz=640)
results[0].show()

# %% [markdown]
# ---
# ## Lab 13: Individual Object Masks — แยกวัตถุแต่ละชิ้น

# %%
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

seg_model = YOLO("yolo26n-seg.pt")
original_image = cv2.imread(IMAGE_PATH)
height, width = original_image.shape[:2]
rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

results = seg_model(rgb_image, imgsz=640, verbose=False)

if len(results) > 0 and results[0].masks is not None:
    masks = results[0].masks
    object_count = len(masks.data)
    
    # สร้าง combined mask
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(object_count):
        mask_array = masks.data[i].cpu().numpy().astype(np.uint8) * 255
        if mask_array.shape[:2] != (height, width):
            mask_array = cv2.resize(mask_array, (width, height), interpolation=cv2.INTER_NEAREST)
        combined_mask = cv2.bitwise_or(combined_mask, mask_array)
    
    # แสดง Original vs Segmented
    combined_result = cv2.bitwise_and(rgb_image, rgb_image, mask=combined_mask)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis('off')
    
    axes[1].imshow(combined_mask, cmap='gray')
    axes[1].set_title("Combined Mask", fontsize=13)
    axes[1].axis('off')
    
    axes[2].imshow(combined_result)
    axes[2].set_title("Segmented Objects", fontsize=13)
    axes[2].axis('off')
    
    plt.suptitle("YOLO26 Instance Segmentation", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # แสดงแต่ละ object แยกกัน
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()
    
    show_count = min(object_count, 8)
    cols = min(show_count, 4)
    rows = (show_count + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if show_count == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(show_count):
        mask_array = masks.data[i].cpu().numpy().astype(np.uint8) * 255
        if mask_array.shape[:2] != (height, width):
            mask_array = cv2.resize(mask_array, (width, height), interpolation=cv2.INTER_NEAREST)
        
        masked_result = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_array)
        
        cls_id = class_ids[i]
        label = seg_model.names[cls_id]
        conf = confs[i]
        
        axes[i].imshow(masked_result)
        axes[i].set_title(f"{label} ({conf:.2f})", fontsize=11)
        axes[i].axis('off')
    
    for j in range(show_count, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle("Individual Object Masks", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Total objects segmented: {object_count}")
    print(f"📐 Mask shape: {masks.shape}")
else:
    print("⚠️ No masks found in the results.")

# %% [markdown]
# ### 💡 Segmentation vs Detection:
# - Detection ให้แค่ **bounding box** (สี่เหลี่ยม)
# - Segmentation ให้ **pixel mask** (รูปร่างจริงของวัตถุ)
# - Segmentation ใช้ compute มากกว่า แต่ให้ข้อมูลละเอียดกว่ามาก

# %% [markdown]
# ---
# ## Lab 14: 🖼️ Background Removal & Replacement ด้วย Segmentation Mask
#
# ใช้ segmentation mask ที่เรียนใน Lab 12–13 มาประยุกต์ทำ:
# 1. **ลบ Background** — เหลือแค่วัตถุที่เลือก
# 2. **เปลี่ยน Background** — ใส่ background ใหม่ (สีพื้น หรือ รูปภาพอื่น)
# 3. **เลือกวัตถุ** — โดยกำหนด class ID หรือ object ID
#
# ### 🧠 หลักการ:
# ```
# Mask = 255 (white) → foreground (เก็บไว้)
# Mask = 0   (black) → background (ลบ/เปลี่ยน)
# ```

# %% [markdown]
# ### 14.1 Helper Function: สร้าง Mask จาก Segmentation Results

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def get_segmentation_masks(image_path, model=None, target_classes=None, target_object_ids=None):
    """
    สร้าง segmentation masks จาก YOLO26 โดยสามารถเลือกเฉพาะ class หรือ object ID ได้
    
    Parameters:
    -----------
    image_path : str
        path ของรูปภาพ
    model : YOLO, optional
        YOLO segmentation model (ถ้าไม่ระบุจะสร้างใหม่)
    target_classes : list of str, optional
        ชื่อ class ที่ต้องการ เช่น ['person', 'sports ball']
        ถ้า None = เลือกทุก class
    target_object_ids : list of int, optional
        index ของ object ที่ต้องการ (0-indexed) เช่น [0, 2, 5]
        ถ้า None = เลือกทุก object
    
    Returns:
    --------
    dict with keys:
        'image_rgb'      : original image in RGB
        'combined_mask'  : combined mask ของ objects ที่เลือก
        'individual_masks': list ของ mask แต่ละ object
        'labels'         : list ของ (class_name, confidence, object_id)
        'all_labels'     : list ของทุก object (สำหรับ reference)
    """
    if model is None:
        model = YOLO("yolo26n-seg.pt")
    
    original = cv2.imread(image_path)
    height, width = original.shape[:2]
    rgb_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    results = model(rgb_image, imgsz=640, verbose=False)
    
    output = {
        'image_rgb': rgb_image,
        'combined_mask': np.zeros((height, width), dtype=np.uint8),
        'individual_masks': [],
        'labels': [],
        'all_labels': [],
        'height': height,
        'width': width,
    }
    
    if len(results) == 0 or results[0].masks is None:
        print("⚠️ No segmentation masks found.")
        return output
    
    masks_data = results[0].masks.data.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()
    
    # สร้าง list ของทุก object สำหรับ reference
    for i in range(len(masks_data)):
        cls_name = model.names[class_ids[i]]
        output['all_labels'].append((cls_name, confs[i], i))
    
    # กรอง objects ตามเงื่อนไข
    for i in range(len(masks_data)):
        cls_name = model.names[class_ids[i]]
        
        if target_classes is not None and cls_name not in target_classes:
            continue
        if target_object_ids is not None and i not in target_object_ids:
            continue
        
        mask = masks_data[i].astype(np.uint8) * 255
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        output['individual_masks'].append(mask)
        output['labels'].append((cls_name, confs[i], i))
        output['combined_mask'] = cv2.bitwise_or(output['combined_mask'], mask)
    
    return output


# แสดงรายการ objects ทั้งหมดในรูป
seg_model = YOLO("yolo26n-seg.pt")
info = get_segmentation_masks(IMAGE_PATH, model=seg_model)

print("📋 All detected objects in this image:")
print("=" * 55)
print(f"{'ID':>4} | {'Class':>15} | {'Confidence':>10}")
print("-" * 55)
for cls_name, conf, obj_id in info['all_labels']:
    print(f"  {obj_id:>2} | {cls_name:>15} | {conf:.4f}")
print(f"\n💡 ใช้ target_classes=['person'] เพื่อเลือกเฉพาะคน")
print(f"💡 ใช้ target_object_ids=[0, 2] เพื่อเลือกเฉพาะ object ที่ 0 และ 2")

# %% [markdown]
# ### 14.2 Background Removal — ลบ Background ออก (เลือกตาม Class)

# %%
def remove_background(image_path, model=None, target_classes=None, target_object_ids=None,
                       bg_color=(255, 255, 255)):
    """
    ลบ background ออกจากรูปภาพ แล้วแทนที่ด้วยสีที่กำหนด
    
    Parameters:
    -----------
    bg_color : tuple (R, G, B)
        สีพื้นหลังใหม่ เช่น (255, 255, 255) = ขาว, (0, 0, 0) = ดำ
    """
    data = get_segmentation_masks(image_path, model, target_classes, target_object_ids)
    
    rgb = data['image_rgb']
    mask = data['combined_mask']
    
    new_bg = np.full_like(rgb, bg_color, dtype=np.uint8)
    
    mask_3ch = cv2.merge([mask, mask, mask])
    foreground = cv2.bitwise_and(rgb, mask_3ch)
    background = cv2.bitwise_and(new_bg, cv2.bitwise_not(mask_3ch))
    result = cv2.add(foreground, background)
    
    return result, data


# === ทดลอง: ลบ Background เลือกเฉพาะ "person" ===
result_white, data = remove_background(
    IMAGE_PATH, model=seg_model, target_classes=['person'], bg_color=(255, 255, 255)
)
result_black, _ = remove_background(
    IMAGE_PATH, model=seg_model, target_classes=['person'], bg_color=(0, 0, 0)
)
result_green, _ = remove_background(
    IMAGE_PATH, model=seg_model, target_classes=['person'], bg_color=(0, 177, 64)
)

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

axes[0].imshow(data['image_rgb'])
axes[0].set_title("Original", fontsize=13)
axes[0].axis('off')

axes[1].imshow(result_white)
axes[1].set_title("White Background\n(person only)", fontsize=13)
axes[1].axis('off')

axes[2].imshow(result_black)
axes[2].set_title("Black Background\n(person only)", fontsize=13)
axes[2].axis('off')

axes[3].imshow(result_green)
axes[3].set_title("Green Screen\n(person only)", fontsize=13)
axes[3].axis('off')

plt.suptitle("🖼️ Background Removal — Filter by Class (person)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n📊 Selected {len(data['labels'])} objects (class='person')")
for cls_name, conf, obj_id in data['labels']:
    print(f"   Object {obj_id}: {cls_name} (conf: {conf:.3f})")

# %% [markdown]
# ### 14.3 Background Removal — เลือกตาม Object ID

# %%
# เลือก object 0 และ 2 (ปรับตามรูปภาพของคุณ)
selected_ids = [0, 2]

result_selected, data_selected = remove_background(
    IMAGE_PATH, model=seg_model, target_object_ids=selected_ids, bg_color=(240, 240, 245)
)
result_all, data_all = remove_background(
    IMAGE_PATH, model=seg_model, target_object_ids=None, bg_color=(240, 240, 245)
)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].imshow(data_all['image_rgb'])
axes[0].set_title("Original", fontsize=13)
axes[0].axis('off')

axes[1].imshow(result_all)
axes[1].set_title(f"All Objects ({len(data_all['labels'])})", fontsize=13)
axes[1].axis('off')

axes[2].imshow(result_selected)
axes[2].set_title(f"Selected Objects (IDs: {selected_ids})", fontsize=13)
axes[2].axis('off')

plt.suptitle("🎯 Background Removal — Filter by Object ID", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n📊 Selected objects: {selected_ids}")
for cls_name, conf, obj_id in data_selected['labels']:
    print(f"   Object {obj_id}: {cls_name} (conf: {conf:.3f})")

# %% [markdown]
# ### 14.4 Background Replacement — เปลี่ยน Background เป็นรูปภาพอื่น

# %%
def replace_background_with_image(image_path, bg_image_path=None, model=None,
                                   target_classes=None, target_object_ids=None):
    """
    เปลี่ยน background เป็นรูปภาพอื่น
    ถ้าไม่ระบุ bg_image_path จะสร้าง gradient background แทน
    """
    data = get_segmentation_masks(image_path, model, target_classes, target_object_ids)
    
    rgb = data['image_rgb']
    mask = data['combined_mask']
    h, w = data['height'], data['width']
    
    if bg_image_path is not None:
        bg = cv2.imread(bg_image_path)
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        bg = cv2.resize(bg, (w, h))
    else:
        # สร้าง gradient background (sunset effect)
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        for y_pos in range(h):
            ratio = y_pos / h
            r = int(255 * (1 - ratio) + 20 * ratio)
            g = int(100 * (1 - ratio) + 10 * ratio)
            b = int(50 * (1 - ratio) + 80 * ratio)
            bg[y_pos, :] = [r, g, b]
    
    mask_3ch = cv2.merge([mask, mask, mask])
    foreground = cv2.bitwise_and(rgb, mask_3ch)
    background = cv2.bitwise_and(bg, cv2.bitwise_not(mask_3ch))
    result = cv2.add(foreground, background)
    
    return result, data


# === ทดลอง: เปลี่ยน background เป็น gradient ===
result_gradient, data = replace_background_with_image(
    IMAGE_PATH, model=seg_model, target_classes=['person']
)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].imshow(data['image_rgb'])
axes[0].set_title("Original", fontsize=13)
axes[0].axis('off')

axes[1].imshow(result_gradient)
axes[1].set_title("Gradient Background\n(person only)", fontsize=13)
axes[1].axis('off')

plt.suptitle("🌅 Background Replacement with Custom Image/Gradient", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 💡 Background Removal — Key Takeaways:
# - **Segmentation mask** คือหัวใจของ background removal
# - **target_classes** เลือกเฉพาะ class (เช่น 'person', 'car')
# - **target_object_ids** เลือกเฉพาะวัตถุบางชิ้น
# - เป็นพื้นฐานของ Zoom virtual background
# - สามารถต่อยอดไป **real-time** ได้โดยใช้ร่วมกับ webcam feed

# %% [markdown]
# ---
# ## Lab 15: 🔮 Blur Selected Object & Bokeh Effect
#
# ใช้ segmentation mask เพื่อ **เบลอเฉพาะวัตถุที่เลือก** หรือ **เบลอ background**
#
# ### Use Cases:
# - **Privacy Protection** — เบลอหน้าคน/ป้ายทะเบียน
# - **Focus Effect** — เบลอ background เน้นวัตถุหลัก (Bokeh)
# - **Content Moderation** — ปิดบัง/เบลอเนื้อหาที่ไม่เหมาะสม
#
# ### 🧠 หลักการ:
# ```
# Blur Object:     result = original × (1 - mask) + blurred × mask
# Blur Background:  result = original × mask + blurred × (1 - mask)
# ```

# %%
def blur_objects(image_path, model=None, target_classes=None, target_object_ids=None,
                 blur_strength=51, blur_background=False):
    """
    เบลอวัตถุที่เลือก หรือเบลอ background (Bokeh effect)
    
    Parameters:
    -----------
    blur_strength : int (odd number)
        ความเข้มของ blur (ยิ่งมากยิ่งเบลอ)
    blur_background : bool
        True  = เบลอ background → Bokeh effect
        False = เบลอ foreground
    """
    data = get_segmentation_masks(image_path, model, target_classes, target_object_ids)
    
    rgb = data['image_rgb']
    mask = data['combined_mask']
    
    if blur_strength % 2 == 0:
        blur_strength += 1
    blurred = cv2.GaussianBlur(rgb, (blur_strength, blur_strength), 0)
    
    mask_float = (mask / 255.0).astype(np.float32)
    mask_3ch = cv2.merge([mask_float, mask_float, mask_float])
    
    if blur_background:
        result = (rgb * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
    else:
        result = (blurred * mask_3ch + rgb * (1 - mask_3ch)).astype(np.uint8)
    
    return result, data


# %% [markdown]
# ### 15.1 Blur Specific Objects

# %%
# === เบลอเฉพาะ "person" ===
result_blur_person, data = blur_objects(
    IMAGE_PATH, model=seg_model, target_classes=['person'],
    blur_strength=51, blur_background=False
)

# === เบลอทุก object ===
result_blur_all, _ = blur_objects(
    IMAGE_PATH, model=seg_model, target_classes=None,
    blur_strength=51, blur_background=False
)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

axes[0].imshow(data['image_rgb'])
axes[0].set_title("Original", fontsize=13)
axes[0].axis('off')

axes[1].imshow(result_blur_person)
axes[1].set_title("Blur Person Only", fontsize=13)
axes[1].axis('off')

axes[2].imshow(result_blur_all)
axes[2].set_title("Blur All Objects", fontsize=13)
axes[2].axis('off')

plt.suptitle("🔮 Blur Selected Objects", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 15.2 Bokeh Effect — เบลอ Background (เน้น Foreground)

# %%
blur_strengths = [21, 51, 101]

fig, axes = plt.subplots(1, len(blur_strengths) + 1, figsize=(24, 6))

axes[0].imshow(data['image_rgb'])
axes[0].set_title("Original", fontsize=13)
axes[0].axis('off')

for idx, strength in enumerate(blur_strengths):
    result_bokeh, _ = blur_objects(
        IMAGE_PATH, model=seg_model, target_classes=['person'],
        blur_strength=strength, blur_background=True
    )
    axes[idx + 1].imshow(result_bokeh)
    axes[idx + 1].set_title(f"Bokeh (blur={strength})", fontsize=13)
    axes[idx + 1].axis('off')

plt.suptitle("📸 Bokeh Effect — Background Blur (เน้น Person)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 15.3 Blur by Object ID

# %%
blur_ids = [0, 1]  # ปรับตามรูปภาพ

result_blur_selected, data_sel = blur_objects(
    IMAGE_PATH, model=seg_model, target_object_ids=blur_ids,
    blur_strength=71, blur_background=False
)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].imshow(data_sel['image_rgb'])
axes[0].set_title("Original", fontsize=13)
axes[0].axis('off')

axes[1].imshow(result_blur_selected)
axes[1].set_title(f"Blurred Objects (IDs: {blur_ids})", fontsize=13)
axes[1].axis('off')

plt.suptitle("🎯 Blur Specific Objects by ID", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n📊 Blurred objects:")
for cls_name, conf, obj_id in data_sel['labels']:
    print(f"   Object {obj_id}: {cls_name} (conf: {conf:.3f})")

# %% [markdown]
# ### 💡 Blur — Key Takeaways:
# - **Blur Object** → privacy (เบลอหน้า/ป้ายทะเบียน)
# - **Blur Background (Bokeh)** → portrait mode เหมือนในมือถือ
# - `blur_strength` ยิ่งมากยิ่งเบลอ (ต้องเป็นเลขคี่)
# - เลือกเฉพาะ class หรือ object ID ได้

# %% [markdown]
# ---
# ## Lab 16: Basic Pose Estimation
#
# Pose Estimation ตรวจจับ **keypoints** บนร่างกายคน (17 จุดตาม COCO format)
#
# ### COCO Keypoints (17 จุด):
# ```
# 0: Nose          1: Left Eye       2: Right Eye
# 3: Left Ear      4: Right Ear      5: Left Shoulder
# 6: Right Shoulder 7: Left Elbow    8: Right Elbow
# 9: Left Wrist    10: Right Wrist   11: Left Hip
# 12: Right Hip     13: Left Knee    14: Right Knee
# 15: Left Ankle    16: Right Ankle
# ```

# %%
pose_model = YOLO("yolo26n-pose.pt")

results = pose_model(IMAGE_PATH, imgsz=640)
results[0].show()

# %% [markdown]
# ---
# ## Lab 17: Custom Skeleton Drawing with OpenCV

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np

pose_model = YOLO("yolo26n-pose.pt")
results = pose_model(IMAGE_PATH, verbose=False)

image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# COCO Skeleton connections
skeleton = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]

kpt_names = [
    "Nose", "L-Eye", "R-Eye", "L-Ear", "R-Ear",
    "L-Shoulder", "R-Shoulder", "L-Elbow", "R-Elbow",
    "L-Wrist", "R-Wrist", "L-Hip", "R-Hip",
    "L-Knee", "R-Knee", "L-Ankle", "R-Ankle"
]

annotated = image_rgb.copy()

for person_idx, person in enumerate(results[0].keypoints.data.cpu().numpy()):
    # วาด keypoints
    for idx, (x, y, conf) in enumerate(person):
        if conf > 0.5:
            cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.circle(annotated, (int(x), int(y)), 7, (255, 255, 255), 1)
    
    # วาด skeleton
    for start, end in skeleton:
        if person[start][2] > 0.5 and person[end][2] > 0.5:
            pt1 = (int(person[start][0]), int(person[start][1]))
            pt2 = (int(person[end][0]), int(person[end][1]))
            cv2.line(annotated, pt1, pt2, (0, 200, 255), 2)

plt.figure(figsize=(15, 10))
plt.imshow(annotated)
plt.title("YOLO26 Pose Estimation — Custom Skeleton Drawing", fontsize=14)
plt.axis('off')
plt.show()

# %% [markdown]
# ---
# ## Lab 18: Keypoints Data Analysis

# %%
keypoints_data = results[0].keypoints.data.cpu().numpy()

print(f"จำนวนคนที่ตรวจจับได้: {len(keypoints_data)}")
print(f"Shape ของ keypoints data: {keypoints_data.shape}")
print(f"  → (จำนวนคน, จำนวน keypoints, 3)  # 3 = x, y, confidence\n")

if len(keypoints_data) > 0:
    person = keypoints_data[0]
    print("📍 Keypoints ของคนที่ 1:")
    print("-" * 50)
    for idx, (x, y, conf) in enumerate(person):
        status = "✅" if conf > 0.5 else "❌"
        print(f"  {status} {kpt_names[idx]:>12}: ({x:.1f}, {y:.1f}) conf={conf:.2f}")

# %% [markdown]
# ---
# ## Lab 20: Object Distance Estimation & Visualization
#
# **ประมาณระยะห่างระหว่างวัตถุ** จากข้อมูล bounding box
#
# ### 🧠 หลักการ:
# ```
# centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
# distance = sqrt((cx1 - cx2)² + (cy1 - cy2)²)
# ```
#
# ### ⚠️ ข้อจำกัด:
# - ระยะที่คำนวณได้เป็น **pixel distance** บนภาพ 2D (ไม่ใช่ระยะจริง 3D)
# - แต่ pixel distance บอก **relative relationship** ได้ดี
# - ใช้ใน: Social distancing, กีฬา analysis, การจัดกลุ่มวัตถุ

# %% [markdown]
# ### 20.1 คำนวณ Centroid และ Distance

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from itertools import combinations
import math


def compute_object_distances(image_path, model=None, target_classes=None):
    """
    คำนวณระยะห่าง (pixel distance) ระหว่างทุกคู่ของวัตถุที่ตรวจจับได้
    """
    if model is None:
        model = YOLO("yolo26s.pt")
    
    original = cv2.imread(image_path)
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    results = model(rgb, imgsz=640, verbose=False)
    
    objects = []
    for i, box in enumerate(results[0].boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        
        if target_classes is not None and cls_name not in target_classes:
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        conf = float(box.conf[0])
        
        objects.append({
            'id': len(objects), 'original_id': i, 'class': cls_name, 'conf': conf,
            'bbox': (x1, y1, x2, y2), 'centroid': (cx, cy),
            'width': x2 - x1, 'height': y2 - y1,
        })
    
    distances = []
    for a, b in combinations(range(len(objects)), 2):
        obj_a, obj_b = objects[a], objects[b]
        dist = math.sqrt(
            (obj_a['centroid'][0] - obj_b['centroid'][0]) ** 2 +
            (obj_a['centroid'][1] - obj_b['centroid'][1]) ** 2
        )
        distances.append({
            'id_a': obj_a['id'], 'id_b': obj_b['id'],
            'class_a': obj_a['class'], 'class_b': obj_b['class'],
            'distance': dist,
            'centroid_a': obj_a['centroid'], 'centroid_b': obj_b['centroid'],
        })
    
    distances.sort(key=lambda x: x['distance'])
    
    return {'objects': objects, 'distances': distances, 'image_rgb': rgb}


# === คำนวณระยะห่าง ===
det_model = YOLO("yolo26s.pt")
dist_data = compute_object_distances(IMAGE_PATH, model=det_model)

print("📋 Detected Objects:")
print("=" * 65)
print(f"{'ID':>4} | {'Class':>15} | {'Conf':>6} | {'Centroid (x, y)':>20} | {'Size (w×h)':>12}")
print("-" * 65)
for obj in dist_data['objects']:
    cx, cy = obj['centroid']
    print(f"  {obj['id']:>2} | {obj['class']:>15} | {obj['conf']:.3f} | ({cx:>7.1f}, {cy:>7.1f}) | {obj['width']:>4}×{obj['height']:<4}")

print(f"\n📏 Distance Pairs (sorted by distance, top 10):")
print("=" * 70)
print(f"{'Pair':>10} | {'Classes':>30} | {'Distance (px)':>14}")
print("-" * 70)
for d in dist_data['distances'][:10]:
    pair_str = f"{d['id_a']}↔{d['id_b']}"
    class_str = f"{d['class_a']} ↔ {d['class_b']}"
    print(f"  {pair_str:>8} | {class_str:>30} | {d['distance']:>12.1f} px")

# %% [markdown]
# ### 20.2 Visualize Distance — แสดงเส้นระยะห่างบนรูปภาพ

# %%
def visualize_distances(dist_data, show_top_n=None, show_pairs=None,
                         line_color=(255, 255, 0), show_all_objects=True):
    """แสดงเส้นระยะห่างระหว่างวัตถุบนรูปภาพ"""
    annotated = dist_data['image_rgb'].copy()
    objects = dist_data['objects']
    distances = dist_data['distances']
    
    np.random.seed(42)
    obj_colors = {obj['id']: tuple(np.random.randint(80, 255, 3).tolist()) for obj in objects}
    
    if show_all_objects:
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            cx, cy = int(obj['centroid'][0]), int(obj['centroid'][1])
            color = obj_colors[obj['id']]
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (cx, cy), 6, color, -1)
            cv2.circle(annotated, (cx, cy), 8, (255, 255, 255), 2)
            
            label = f"ID:{obj['id']} {obj['class']}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if show_pairs is not None:
        display_dists = [d for d in distances
                         if (d['id_a'], d['id_b']) in show_pairs
                         or (d['id_b'], d['id_a']) in show_pairs]
    elif show_top_n is not None:
        display_dists = distances[:show_top_n]
    else:
        display_dists = distances
    
    for d in display_dists:
        pt1 = (int(d['centroid_a'][0]), int(d['centroid_a'][1]))
        pt2 = (int(d['centroid_b'][0]), int(d['centroid_b'][1]))
        
        cv2.line(annotated, pt1, pt2, line_color, 2)
        
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        dist_label = f"{d['distance']:.0f}px"
        
        (tw, th), _ = cv2.getTextSize(dist_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (mid_x - 2, mid_y - th - 4),
                       (mid_x + tw + 4, mid_y + 4), (0, 0, 0), -1)
        cv2.putText(annotated, dist_label, (mid_x, mid_y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)
    
    return annotated


# === แสดง Top 5 คู่ที่ใกล้ที่สุด ===
annotated_top5 = visualize_distances(dist_data, show_top_n=8, line_color=(255, 255, 0))

plt.figure(figsize=(16, 10))
plt.imshow(annotated_top5)
plt.title("📏 Object Distance Estimation — Top 5 Closest Pairs", fontsize=15)
plt.axis('off')
plt.show()

# %% [markdown]
# ### 20.3 Distance ระหว่าง Objects ที่เจาะจง

# %%
specific_pairs = [(0, 1), (0, 2)]  # ปรับตามรูปภาพ

annotated_specific = visualize_distances(
    dist_data, show_pairs=specific_pairs, line_color=(0, 255, 255)
)

plt.figure(figsize=(16, 10))
plt.imshow(annotated_specific)
plt.title(f"📏 Distance Between Specific Object Pairs: {specific_pairs}", fontsize=15)
plt.axis('off')
plt.show()

for d in dist_data['distances']:
    if (d['id_a'], d['id_b']) in specific_pairs or (d['id_b'], d['id_a']) in specific_pairs:
        print(f"  Object {d['id_a']} ({d['class_a']}) ↔ Object {d['id_b']} ({d['class_b']}): {d['distance']:.1f} px")

# %% [markdown]
# ---
# ## Lab 21: Distance Matrix & Heatmap

# %%
n = len(dist_data['objects'])

if n >= 2:
    dist_matrix = np.zeros((n, n))
    
    for d in dist_data['distances']:
        dist_matrix[d['id_a'], d['id_b']] = d['distance']
        dist_matrix[d['id_b'], d['id_a']] = d['distance']
    
    labels = [f"ID:{obj['id']}\n{obj['class']}" for obj in dist_data['objects']]
    
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.6)))
    
    im = ax.imshow(dist_matrix, cmap='YlOrRd_r', aspect='auto')
    
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{dist_matrix[i,j]:.0f}",
                         ha='center', va='center', fontsize=8,
                         color='white' if dist_matrix[i,j] < dist_matrix.max() * 0.5 else 'black')
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=8)
    
    plt.colorbar(im, label='Distance (pixels)')
    plt.title("📊 Object Distance Matrix (Heatmap)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    non_zero = dist_matrix[dist_matrix > 0]
    if len(non_zero) > 0:
        print(f"\n📊 Distance Statistics:")
        print(f"   Min distance:  {non_zero.min():.1f} px")
        print(f"   Max distance:  {non_zero.max():.1f} px")
        print(f"   Mean distance: {non_zero.mean():.1f} px")
        print(f"   Median:        {np.median(non_zero):.1f} px")
else:
    print("⚠️ Need at least 2 objects to compute distances.")

# %% [markdown]
# ---
# ## Lab 23: Object Tracking on Video (Bonus)
#
# YOLO26 รองรับ Object Tracking ด้วย BoT-SORT / ByteTrack
# แต่ละวัตถุจะได้ **Track ID** ที่ unique ตลอด video
#
# ```python
# # ตัวอย่าง Tracking บน video (ต้องมีไฟล์ video)
# from ultralytics import YOLO
# import cv2
#
# model = YOLO("yolo26n.pt")
#
# cap = cv2.VideoCapture("video.mp4")
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     
#     # track() จะให้ track_id สำหรับแต่ละวัตถุ
#     results = model.track(frame, persist=True, verbose=False)
#     
#     annotated = results[0].plot()
#     cv2.imshow("YOLO26 Tracking", annotated)
#     
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# ```
#
# **Key Parameters สำหรับ Tracking:**
# - `persist=True` — บอกว่าเป็น frame ถัดไปของ video เดียวกัน (ต้องใช้เสมอ)
# - `tracker="bytetrack.yaml"` — เลือก tracker algorithm
# - `tracker="botsort.yaml"` — tracker อีกตัวที่มี ReID support
#
# ### 💡 Distance + Tracking — Key Takeaways:
# - **Distance** = spatial relationship (ภาพเดียว)
# - **Tracking** = temporal relationship (ข้ามเวลา/frame)
# - รวมกันได้: track วัตถุ + คำนวณระยะห่างต่อ frame → เช่น ดูว่า 2 คนเข้าใกล้กันเมื่อไหร่

# %% [markdown]
# ## 📝 Summary: YOLO26 Tasks Overview
#
# | Task | Model | Output | Use Case |
# |------|-------|--------|----------|
# | **Detection** | `yolo26n.pt` | Bounding boxes + classes | วัตถุอยู่ตรงไหน |
# | **Segmentation** | `yolo26n-seg.pt` | Pixel masks | รูปร่างจริงของวัตถุ |
# | **Pose Estimation** | `yolo26n-pose.pt` | 17 keypoints per person | ท่าทางคน |
# | **Classification** | `yolo26n-cls.pt` | Class probabilities | รูปนี้เป็นอะไร |
# | **OBB** | `yolo26n-obb.pt` | Rotated bounding boxes | วัตถุที่หมุนเอียง |
# | **Tracking** | `.track()` method | Track IDs across frames | ติดตามวัตถุใน video |
#
# ### 🔑 YOLO26 Key Innovations:
# 1. **NMS-Free Inference** — ลด latency, deploy ง่ายขึ้น
# 2. **MuSGD Optimizer** — training ที่เสถียรกว่า
# 3. **43% Faster CPU Inference** — เหมาะกับ edge devices
# 4. **Dual-Head Architecture** — เลือกได้ระหว่าง speed vs accuracy
# 5. **YOLOE-26** — Open-vocabulary detection ด้วย text/visual prompts
#

# %%
