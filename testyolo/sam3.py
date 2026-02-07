# %% [markdown]
# # Lab: SAM 3 Advanced Concept Segmentation and Video Tracking
# **Course:** Advanced Computer Vision with MLOps  
# **Topic:** Segment Anything Model 3 (SAM 3) - Promptable Concept Segmentation
#
# ---
#
# ## วัตถุประสงค์การเรียนรู้ (Learning Objectives)
#
# เมื่อจบ Lab นี้ นักศึกษาจะสามารถ:
# 1. เข้าใจความแตกต่างระหว่าง SAM 2 และ SAM 3
# 2. ใช้ SAM 3 ในการทำ Concept Segmentation ด้วย Text Prompts
# 3. ใช้ Image Exemplar Prompts เพื่อค้นหา Object ที่คล้ายกัน
# 4. ประยุกต์ใช้ SAM 3 ในการ Track Objects ใน Video
# 5. ประเมินประสิทธิภาพของ Model ด้วย Metrics ที่เหมาะสม
# 6. แก้ปัญหาขั้นสูงเกี่ยวกับ Multi-Object Tracking และ Interactive Refinement
#
# ---
#
# ## ข้อมูลเบื้องต้น (Background)
#
# ### SAM 3 คืออะไร?
#
# SAM 3 (Segment Anything Model 3) เป็น Foundation Model จาก Meta ที่ออกแบบมาสำหรับ **Promptable Concept Segmentation (PCS)** โดยมีความสามารถหลักดังนี้:
#
# - **Text-based Segmentation**: ใช้ Noun Phrases เช่น "yellow school bus" หรือ "person wearing red hat"
# - **Image Exemplar Prompts**: ใช้ Bounding Box ของ Object ตัวอย่างเพื่อค้นหา Object ที่คล้ายกันทั้งหมด
# - **Video Tracking**: Track หลาย Objects พร้อมกันด้วย Concept-based Prompts
# - **Interactive Refinement**: ปรับปรุงผลลัพธ์แบบ Iterative ด้วย Positive/Negative Exemplars
#
# ### ความแตกต่างระหว่าง SAM 2 vs SAM 3
#
# | คุณสมบัติ | SAM 2 | SAM 3 |
# |----------|-------|-------|
# | **Task** | Single object per prompt | All instances of concept |
# | **Prompts** | Points, boxes, masks | + Text, exemplars |
# | **Detection** | ต้องใช้ Detector ภายนอก | Built-in open-vocabulary detector |
# | **Zero-Shot** | ไม่รองรับ | 47.0 AP บน LVIS |
# | **Inference Speed** | ~23 ms/object | 30 ms (100+ objects) |
#
# ### Key Metrics
#
# - **CGF1 (Classification-Gated F1)**: รวม Localization และ Classification
# - **pmF1 (Positive Macro F1)**: วัด Localization Quality
# - **IL_MCC**: วัดความแม่นยำของ Binary Classification ("Concept มีอยู่หรือไม่?")
#
# ---
#
# ## ส่วนที่ 1: Setup และ Installation
#
# ### 1.1 ติดตั้ง Dependencies

# %%
# ติดตั้ง Ultralytics เวอร์ชันล่าสุด (ต้อง >= 8.3.237)
!uv pip install -U ultralytics

# %%
# ติดตั้ง CLIP Package ที่ถูกต้อง (สำหรับ Text Encoding)
!uv pip uninstall clip 
!uv pip install git+https://github.com/ultralytics/CLIP.git

# %%
# Import Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import SAM
from ultralytics.models.sam import SAM3SemanticPredictor, SAM3VideoSemanticPredictor
from ultralytics.utils.plotting import Annotator, colors

# ตั้งค่า Matplotlib
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

# %% [markdown]
# ### 1.2 ดาวน์โหลด Model Weights
#
# **⚠️ สำคัญ:** SAM 3 Weights ไม่ได้ Download อัตโนมัติ คุณต้อง:
# 1. Request access ที่: https://huggingface.co/facebook/sam3
# 2. Download `sam3.pt` จาก: https://huggingface.co/facebook/sam3/resolve/main/sam3.pt
# 3. วางไฟล์ใน Working Directory หรือระบุ Full Path

# %%
from pathlib import Path
from huggingface_hub import hf_hub_download

# ใส่ Token ของคุณ (ได้จาก https://huggingface.co/settings/tokens)
HF_TOKEN = "hf_hfobm"

model_path = Path("sam3.pt")

if not model_path.exists():
    print(f"⏳ กำลังดาวน์โหลด {model_path.name}...")
    hf_hub_download(
        repo_id="facebook/sam3",
        filename="sam3.pt",
        local_dir=".",
        token=HF_TOKEN,  # ใส่ token ตรงนี้
        local_dir_use_symlinks=False
    )
    print(f"✅ ดาวน์โหลดเสร็จสิ้น")
else:
    print(f"✅ พบไฟล์ {model_path}")

# %% [markdown]
# ---
#
# ## ส่วนที่ 2: Text-based Concept Segmentation
#
# ในส่วนนี้เราจะเรียนรู้การใช้ **Text Prompts** เพื่อ Segment Objects

# %% [markdown]
# ### 2.1 Single Concept Segmentation

# %%
# สร้าง Predictor สำหรับ Concept Segmentation
overrides = dict(
    conf=0.25,              # Confidence Threshold
    task="segment",
    mode="predict",
    model=model_path,
    half=True,              # ใช้ FP16 สำหรับความเร็ว
    save=True,
    verbose=True
)

predictor = SAM3SemanticPredictor(overrides=overrides)

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_results(image_path, results, title="SAM3 Segmentation Results", figsize=(15, 10)):
    """
    แสดงผลภาพพร้อม Segmentation Masks จาก SAM3
    
    Parameters:
    -----------
    image_path : str
        Path ของภาพต้นฉบับ
    results : list
        ผลลัพธ์จาก SAM3 predictor
    title : str
        ชื่อที่แสดงบนภาพ
    figsize : tuple
        ขนาดของ Figure (width, height)
    
    Returns:
    --------
    None (แสดงภาพด้วย matplotlib)
    """
    # โหลดภาพต้นฉบับ
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # ตรวจสอบว่ามี results หรือไม่
    if not results or results[0].masks is None:
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.title(f"{title}\nNo objects found")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return
    
    # ดึงข้อมูล masks และ boxes
    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    # ดึง confidence scores (ถ้ามี)
    scores = None
    if hasattr(results[0].boxes, 'conf') and results[0].boxes.conf is not None:
        scores = results[0].boxes.conf.cpu().numpy()
    
    # สร้าง Figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ภาพซ้าย: ภาพต้นฉบับ
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # ภาพขวา: ภาพพร้อม Masks
    axes[1].imshow(img)
    
    # สร้างสีสำหรับแต่ละ mask
    np.random.seed(42)  # ให้สีคงที่
    colors = np.random.randint(0, 255, size=(len(masks), 3))
    
    # วาด Masks
    overlay = np.zeros_like(img, dtype=np.float32)
    for i, mask in enumerate(masks):
        color = colors[i]
        overlay[mask > 0] = color
    
    # Blend overlay กับภาพต้นฉบับ
    axes[1].imshow(overlay.astype(np.uint8), alpha=0.5)
    
    # วาด Bounding Boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = colors[i] / 255  # Normalize สำหรับ matplotlib
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                              fill=False, edgecolor=color, linewidth=2)
        axes[1].add_patch(rect)
        
        # แสดง label และ score
        label = f"ID:{i}"
        if scores is not None:
            label += f" ({scores[i]:.2f})"
        axes[1].text(x1, y1-5, label, color='white', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    axes[1].set_title(f"{title}\nFound {len(masks)} objects")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_results_grid(image_path, results, concepts, figsize=(18, 6)):
    """
    แสดงผลแบบ Grid โดยแยกแต่ละ Concept (ถ้ามีหลาย concepts)
    
    Parameters:
    -----------
    image_path : str
        Path ของภาพต้นฉบับ
    results : list
        ผลลัพธ์จาก SAM3 predictor
    concepts : list
        รายการ concepts ที่ค้นหา เช่น ["person", "car"]
    figsize : tuple
        ขนาดของ Figure
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    n_cols = len(concepts) + 1  # +1 สำหรับภาพต้นฉบับ
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    # ภาพต้นฉบับ
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    if not results or results[0].masks is None:
        for i, concept in enumerate(concepts):
            axes[i+1].imshow(img)
            axes[i+1].set_title(f"{concept}\nNot found")
            axes[i+1].axis('off')
        plt.tight_layout()
        plt.show()
        return
    
    masks = results[0].masks.data.cpu().numpy()
    
    # สำหรับตอนนี้แสดงทุก masks ใน subplot เดียว
    # (SAM3 อาจไม่แยก masks ตาม concept โดยตรง)
    for i, concept in enumerate(concepts):
        axes[i+1].imshow(img)
        
        # วาดทุก masks
        for j, mask in enumerate(masks):
            colored_mask = np.zeros_like(img)
            color = plt.cm.tab10(j % 10)[:3]
            color = (np.array(color) * 255).astype(np.uint8)
            colored_mask[mask > 0] = color
            axes[i+1].imshow(colored_mask, alpha=0.4)
        
        axes[i+1].set_title(f"Query: '{concept}'\n{len(masks)} objects")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()


# %%
# ลอง generic terms
results = predictor(text=["object", "thing"])
display_results(test_image, results, title="Generic Objects")

# %%
# ทดลองกับ Sample Image
test_image = "./envi2.png"

# Set Image (Extract Features ครั้งเดียว)
predictor.set_image(test_image)

# Query ด้วย Text Prompt
results = predictor(text=["car"])

# แสดงผลด้วยฟังก์ชันที่สร้าง
display_results(test_image, results, title="SAM3: person detection")



# %%
# หรือถ้าค้นหาหลาย concepts
results_multi = predictor(text=["person", "car", "bicycle"])
display_results(test_image, results_multi, title="Multiple Concepts")



# %%
# แสดงแบบ Grid
display_results_grid(test_image, results_multi, ["person", "car", "bicycle"])

# %% [markdown]
# **คำอธิบาย:**
# - `set_image()`: Extract Image Features ครั้งเดียว แล้วเก็บไว้ใน Memory
# - `text=["person"]`: ค้นหาทุก Instance ของ "person" ในภาพ
# - Model จะคืน Masks ของทุกคนที่พบในภาพ

# %% [markdown]
# ### 2.2 Multiple Concepts Segmentation

# %%
# Query หลาย Concepts พร้อมกัน
results = predictor(text=["person", "car", "bicycle"])

# Visualize Results
if results and results[0].masks is not None:
    # ดึงภาพต้นฉบับ
    img = cv2.imread(test_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    
    # วาด Masks
    for i, mask in enumerate(masks):
        colored_mask = np.zeros_like(img)
        color = np.random.randint(0, 255, size=3)
        colored_mask[mask > 0] = color
        plt.imshow(colored_mask, alpha=0.5)
    
    plt.title(f"Found {len(masks)} objects")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("ไม่พบ Objects ที่ตรงกับ Text Prompts")

# %% [markdown]
# ### 💡 ปัญหาที่ 1: Descriptive Text Prompts
#
# **โจทย์:** ใช้ Descriptive Phrases เพื่อหา Objects ที่เฉพาะเจาะจงมากขึ้น
#
# **ตัวอย่าง:**
# - "person wearing red shirt"
# - "yellow school bus"
# - "dog with black spots"

# %%
# TODO: ลองใช้ Descriptive Prompts
# Hint: SAM 3 สามารถเข้าใจ Simple Adjectives และ Attributes

results = predictor(text=["person wearing red shirt", "person wearing blue shirt"])

# Visualize และเปรียบเทียบผลลัพธ์

# %% [markdown]
# ---
#
# ## ส่วนที่ 3: Image Exemplar-based Segmentation
#
# ใช้ **Bounding Boxes** เป็น Visual Prompts เพื่อหา Objects ที่คล้ายกัน

# %% [markdown]
# ### 3.1 Single Exemplar

# %%
# ใช้ Bounding Box ของ Object ตัวหนึ่งเป็นตัวอย่าง
# Format: [x1, y1, x2, y2] (Top-left และ Bottom-right coordinates)

# TODO: ปรับ Coordinates ให้เหมาะกับภาพของคุณ
example_bbox = [[100, 150, 300, 400]]  # ตัวอย่าง: Bounding box ของรถคันหนึ่ง

results = predictor(bboxes=example_bbox)

# %% [markdown]
# **คำอธิบาย:**
# - SAM 3 จะใช้ Object ใน Bounding Box เป็น "Exemplar"
# - Model จะค้นหาทุก Object ที่คล้ายกับ Exemplar ในภาพ
# - เหมาะสำหรับกรณีที่ไม่รู้ว่าจะเรียก Object นี้ว่าอะไร

# %% [markdown]
# ### 3.2 Multiple Exemplars (Positive and Negative)

# %%
# ใช้หลาย Bounding Boxes พร้อมกัน
# Positive Examples: Objects ที่ต้องการหา
# Negative Examples: Objects ที่ไม่ต้องการ

positive_bboxes = [
    [100, 150, 300, 400],  # Object 1
    [500, 200, 700, 500],  # Object 2
]

results = predictor(bboxes=positive_bboxes)

# %% [markdown]
# ### 💡 ปัญหาที่ 2: Interactive Refinement
#
# **โจทย์:** ปรับปรุงผลลัพธ์แบบ Iterative
# 1. เริ่มจาก Text Prompt
# 2. ถ้าผลลัพธ์ไม่ดี ให้เพิ่ม Positive/Negative Exemplars
# 3. วัดผลว่า Accuracy ดีขึ้นเท่าไหร่

# %%
# TODO: Implement Interactive Refinement Pipeline
# Hint: ใช้ set_image() ครั้งเดียว แล้วเรียก predictor() หลายครั้ง

# Iteration 1: Text only
# results_v1 = predictor(text=["cat"])

# Iteration 2: Text + 1 Exemplar
# results_v2 = predictor(text=["cat"], bboxes=[...])

# Iteration 3: Text + 2 Exemplars
# results_v3 = predictor(text=["cat"], bboxes=[..., ...])

# เปรียบเทียบ CGF1 Score ในแต่ละ Iteration

# %% [markdown]
# ---
#
# ## ส่วนที่ 4: Feature Reuse for Efficiency
#
# เพื่อประสิทธิภาพ เราสามารถ **Extract Features ครั้งเดียว** แล้วใช้ซ้ำสำหรับหลาย Queries

# %% [markdown]
# ### 4.1 Feature Extraction and Reuse

# %%
# สร้าง Predictors 2 ตัว
predictor1 = SAM3SemanticPredictor(overrides=overrides)
predictor2 = SAM3SemanticPredictor(overrides=overrides)

# Predictor 1: Extract Features
source = test_image
predictor1.set_image(source)
src_shape = cv2.imread(source).shape[:2]

# Predictor 2: Setup Model
predictor2.setup_model()

# %%
# ใช้ Features จาก Predictor 1 กับ Predictor 2
# Query 1: Text Prompt
masks1, boxes1 = predictor2.inference_features(
    predictor1.features, 
    src_shape=src_shape, 
    text=["person"]
)

# Query 2: Bounding Box Prompt
masks2, boxes2 = predictor2.inference_features(
    predictor1.features, 
    src_shape=src_shape, 
    bboxes=[[100, 150, 300, 400]]
)

# %% [markdown]
# **ประโยชน์:**
# - ลดเวลา Inference เมื่อต้อง Query หลายครั้งบนภาพเดียวกัน
# - เหมาะสำหรับ Interactive Applications

# %%
# Visualize Feature Reuse Results
def visualize_masks(image_path, masks, boxes, title):
    """Helper Function สำหรับแสดง Segmentation Masks"""
    if masks is None or len(masks) == 0:
        print(f"No masks found for: {title}")
        return
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    annotator = Annotator(img, pil=False)
    
    # วาด Masks
    mask_colors = [colors(i, True) for i in range(len(masks))]
    annotator.masks(masks, mask_colors)
    
    # วาด Bounding Boxes
    for i, box in enumerate(boxes):
        annotator.box_label(box, label=f"ID:{i}", color=mask_colors[i])
    
    plt.figure(figsize=(15, 10))
    plt.imshow(annotator.result())
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize Both Results
if masks1 is not None:
    visualize_masks(source, masks1.cpu().numpy(), boxes1.cpu().numpy(), 
                    "Query 1: Text Prompt 'person'")

if masks2 is not None:
    visualize_masks(source, masks2.cpu().numpy(), boxes2.cpu().numpy(), 
                    "Query 2: Bounding Box Prompt")

# %% [markdown]
# ### 💡 ปัญหาที่ 3: Batch Processing Efficiency
#
# **โจทย์:** ออกแบบ Pipeline สำหรับ Process ภาพหลายๆ ภาพอย่างมีประสิทธิภาพ
#
# **เงื่อนไข:**
# - มีภาพ 100 ภาพ
# - แต่ละภาพต้อง Query 5 Concepts
# - วัดเวลาที่ใช้ ระหว่าง:
#   1. Extract Features ทุกครั้ง
#   2. Reuse Features

# %%
# TODO: Implement Batch Processing Pipeline
import time

def process_without_reuse(image_paths, text_queries):
    """Process โดยไม่ Reuse Features"""
    start = time.time()
    # ... implementation ...
    end = time.time()
    return end - start

def process_with_reuse(image_paths, text_queries):
    """Process โดย Reuse Features"""
    start = time.time()
    # ... implementation ...
    end = time.time()
    return end - start

# เปรียบเทียบเวลา
# time_without = process_without_reuse(...)
# time_with = process_with_reuse(...)
# speedup = time_without / time_with

# %% [markdown]
# ---
#
# ## ส่วนที่ 5: Video Concept Tracking
#
# SAM 3 สามารถ Track Objects ใน Video ได้ทั้ง Text Prompts และ Visual Prompts

# %% [markdown]
# ### 5.1 Text-based Video Tracking

# %%
# สร้าง Video Predictor
video_overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    imgsz=640,              # ลด Resolution เพื่อความเร็ว
    model=model_path,
    half=True,
    save=True,
    verbose=True
)

video_predictor = SAM3VideoSemanticPredictor(overrides=video_overrides)

# %%
# Track Concepts ใน Video ด้วย Text Prompts
video_path = "path/to/your/video.mp4"  # TODO: ใส่ Path ของ Video

# Track "person" และ "car" ใน Video
results = video_predictor(
    source=video_path,
    text=["person", "car"],
    stream=True  # Process Frame-by-Frame
)

# Process และแสดงผล
for r in results:
    # r.show()  # แสดงผลแบบ Real-time
    
    # หรือ Save Frame
    # r.save(f"output_frame_{r.frame_id}.jpg")
    
    # หรือ Extract Information
    if r.masks is not None:
        num_objects = len(r.masks)
        print(f"Frame {r.frame_id}: Found {num_objects} objects")

# %% [markdown]
# ### 5.2 Bounding Box-based Video Tracking

# %%
# Track Objects โดยระบุ Initial Bounding Boxes ใน Frame แรก
results = video_predictor(
    source=video_path,
    bboxes=[[100, 150, 300, 400], [500, 200, 700, 500]],  # TODO: ปรับตาม Video
    labels=[1, 1],  # Positive Labels
    stream=True
)

for r in results:
    r.show()

# %% [markdown]
# ### 💡 ปัญหาที่ 4: Multi-Object Tracking with Occlusion
#
# **โจทย์:** Track หลาย Objects ที่มี Occlusion (บังกัน)
#
# **สถานการณ์:**
# - Video มีคน 5 คน กำลังเดิน
# - บางครั้งคนเดินบังกัน
# - ต้อง Maintain Object Identity ตลอด Video
#
# **ความท้าทาย:**
# 1. SAM 3 จะ Re-detect Objects อย่างไรเมื่อ Un-occluded?
# 2. Temporal Consistency จะดีแค่ไหน?
# 3. เกิด ID Switch บ่อยไหม?

# %%
# TODO: Implement Occlusion Handling
# Hint: SAM 3 ใช้ Memory Bank และ Temporal Disambiguation

# 1. เตรียม Video ที่มี Occlusion
# 2. Track ด้วย Text Prompt "person"
# 3. วิเคราะห์ว่ามี ID Switch กี่ครั้ง
# 4. ลองใช้ Periodic Re-prompting (ทุกๆ N Frames)

# %% [markdown]
# ---
#
# ## ส่วนที่ 6: Performance Evaluation
#
# วัดประสิทธิภาพของ SAM 3 ด้วย Metrics ที่เหมาะสม

# %% [markdown]
# ### 6.1 Classification-Gated F1 (CGF1)
#
# CGF1 = 100 × pmF1 × IL_MCC
#
# Where:
# - **pmF1**: Positive Macro F1 (Localization Quality)
# - **IL_MCC**: Image-Level Matthews Correlation Coefficient (Classification Accuracy)

# %%
def calculate_cgf1(predictions, ground_truth):
    """
    คำนวณ CGF1 Score
    
    Parameters:
    -----------
    predictions : list of dict
        [{'masks': np.array, 'labels': list, 'scores': list}]
    ground_truth : list of dict
        [{'masks': np.array, 'labels': list}]
    
    Returns:
    --------
    cgf1 : float
        CGF1 Score (0-100)
    """
    # TODO: Implement CGF1 Calculation
    # 1. คำนวณ pmF1 จาก IoU ของ Masks
    # 2. คำนวณ IL_MCC จาก Presence Prediction
    # 3. CGF1 = pmF1 * IL_MCC * 100
    
    pass

# %% [markdown]
# ### 6.2 Benchmark Testing

# %%
# ทดสอบ SAM 3 บน Standard Benchmark
# เช่น LVIS, COCO, หรือ Dataset ที่คุณสร้างเอง

def evaluate_on_dataset(predictor, dataset_path, text_queries):
    """
    Evaluate SAM 3 บน Dataset
    
    Returns:
    --------
    metrics : dict
        {'CGF1': float, 'pmF1': float, 'IL_MCC': float, 'inference_time': float}
    """
    results = []
    
    for image_path in Path(dataset_path).glob("*.jpg"):
        # Load Image
        predictor.set_image(str(image_path))
        
        # Predict
        start = time.time()
        pred = predictor(text=text_queries)
        end = time.time()
        
        # Store Results
        results.append({
            'image': image_path.name,
            'prediction': pred,
            'time': end - start
        })
    
    # Calculate Metrics
    # ... implementation ...
    
    return results

# %%
# TODO: Run Evaluation
# dataset_path = "path/to/test/dataset"
# text_queries = ["person", "car", "bicycle"]
# results = evaluate_on_dataset(predictor, dataset_path, text_queries)

# %% [markdown]
# ### 💡 ปัญหาที่ 5: SAM 3 vs YOLO11 Comparison
#
# **โจทย์:** เปรียบเทียบประสิทธิภาพระหว่าง SAM 3 และ YOLO11-seg
#
# **Metrics ที่ต้องวัด:**
# 1. Inference Speed (FPS)
# 2. Memory Usage (MB)
# 3. Accuracy (mAP, F1)
# 4. Zero-shot Capability
#
# **Dataset:** COCO Validation Set (หรือ Subset ของมัน)

# %%
# TODO: Implement Comparison Pipeline
from ultralytics import YOLO

# 1. Load Models
# sam3 = SAM(model_path)
# yolo11 = YOLO("yolo11n-seg.pt")

# 2. Run Inference on Same Dataset
# 3. Compare Metrics
# 4. Plot Comparison Chart

# %% [markdown]
# ---
#
# ## ส่วนที่ 7: Advanced Applications
#
# ประยุกต์ใช้ SAM 3 ในปัญหาจริง

# %% [markdown]
# ### 7.1 Object Counting

# %%
def count_objects(predictor, image_path, concept):
    """
    นับจำนวน Objects ของ Concept ที่กำหนด
    
    Parameters:
    -----------
    predictor : SAM3SemanticPredictor
        SAM 3 Predictor
    image_path : str
        Path ของภาพ
    concept : str
        Concept ที่ต้องการนับ (e.g., "person", "car")
    
    Returns:
    --------
    count : int
        จำนวน Objects
    """
    predictor.set_image(image_path)
    results = predictor(text=[concept])
    
    if results and results[0].masks is not None:
        return len(results[0].masks)
    else:
        return 0

# %%
# ทดสอบ Object Counting
test_concepts = ["person", "car", "bicycle", "traffic light"]

for concept in test_concepts:
    count = count_objects(predictor, test_image, concept)
    print(f"{concept}: {count} instances")

# %% [markdown]
# ### 💡 ปัญหาที่ 6: Crowd Counting Application
#
# **โจทย์:** สร้าง Application สำหรับนับคนในสถานที่แออัด
#
# **ความท้าทาย:**
# 1. คนอยู่ชิดกันมาก (Occlusion)
# 2. ขนาดของคนต่างกันมาก (Scale Variation)
# 3. มุมกล้องต่างกัน (Viewpoint)
#
# **เป้าหมาย:**
# - Accuracy > 90% (เทียบกับ Ground Truth)
# - Speed > 1 FPS บน CPU

# %%
# TODO: Implement Crowd Counting Pipeline
# 1. Load Crowd Images
# 2. Use SAM 3 with text="person"
# 3. Apply Post-processing (NMS, Size Filtering, etc.)
# 4. Evaluate Accuracy

# %% [markdown]
# ### 7.2 Video Analytics: People Flow Analysis

# %%
def analyze_people_flow(video_path, entry_line, exit_line):
    """
    วิเคราะห์การเข้า-ออกของคน
    
    Parameters:
    -----------
    video_path : str
        Path ของ Video
    entry_line : tuple
        (x1, y1, x2, y2) ของเส้นทางเข้า
    exit_line : tuple
        (x1, y1, x2, y2) ของเส้นทางออก
    
    Returns:
    --------
    analytics : dict
        {'total_entered': int, 'total_exited': int, 'current_count': int}
    """
    predictor = SAM3VideoSemanticPredictor(overrides=video_overrides)
    
    entered = 0
    exited = 0
    tracked_ids = set()
    
    results = predictor(source=video_path, text=["person"], stream=True)
    
    for r in results:
        if r.masks is None:
            continue
        
        # Check แต่ละ Object ว่าข้ามเส้นหรือไม่
        # ... implementation ...
        
        pass
    
    return {
        'total_entered': entered,
        'total_exited': exited,
        'current_count': entered - exited
    }

# %% [markdown]
# ### 💡 ปัญหาที่ 7: Retail Analytics System
#
# **โจทย์:** สร้างระบบวิเคราะห์พฤติกรรมลูกค้าในร้านค้า
#
# **Requirements:**
# 1. นับจำนวนคนที่เข้า-ออกร้าน
# 2. Track เส้นทางการเดินของลูกค้า (Heatmap)
# 3. วิเคราะห์ Dwell Time (เวลาที่อยู่ในแต่ละโซน)
# 4. ตรวจจับ Queue (คิว) หน้า Checkout
#
# **Output:**
# - Dashboard แสดงข้อมูล Real-time
# - Report รายวัน/รายสัปดาห์

# %%
# TODO: Implement Retail Analytics System
# Hint: ใช้ SAM 3 Video Tracking + Kalman Filter + Zone Detection

# %% [markdown]
# ---
#
# ## ส่วนที่ 8: SAM 3 Agent (MLLM Integration)
#
# ใช้ SAM 3 ร่วมกับ Multimodal LLM เพื่อจัดการกับ Complex Queries

# %% [markdown]
# ### 8.1 Complex Query Examples
#
# SAM 3 เหมาะกับ Simple Noun Phrases สำหรับ Complex Queries ต้องใช้ร่วมกับ MLLM:
#
# **Simple (SAM 3 Native):**
# - "yellow school bus"
# - "person wearing red hat"
#
# **Complex (SAM 3 Agent):**
# - "People sitting down but not holding a gift box"
# - "The dog closest to the camera without a collar"
# - "Red objects larger than the person's hand"

# %% [markdown]
# ### 💡 ปัญหาที่ 8: Visual Reasoning Pipeline
#
# **โจทย์:** สร้าง Pipeline ที่รับ Complex Natural Language Query
#
# **ตัวอย่าง Query:**
# - "Find all chairs that are empty"
# - "Segment people who are looking at their phones"
# - "Highlight fruits that appear ripe"
#
# **Architecture:**
# 1. MLLM แปลง Complex Query → Simple Queries
# 2. SAM 3 ทำ Segmentation
# 3. MLLM วิเคราะห์ Masks และ Filter ผลลัพธ์

# %%
# TODO: Implement SAM 3 Agent Pipeline
# Pseudo-code:
# 
# def sam3_agent(image, complex_query):
#     # 1. MLLM: Break down query
#     simple_queries = mllm.decompose(complex_query)
#     
#     # 2. SAM 3: Segment
#     all_masks = []
#     for query in simple_queries:
#         masks = sam3.predict(image, text=query)
#         all_masks.append(masks)
#     
#     # 3. MLLM: Analyze and filter
#     final_masks = mllm.filter(all_masks, complex_query)
#     
#     return final_masks

# %% [markdown]
# ---
#
# ## ส่วนที่ 9: Limitations and Troubleshooting
#
# ### 9.1 Known Limitations
#
# 1. **Phrase Complexity**: ดีที่สุดกับ Simple Noun Phrases
# 2. **Ambiguity**: Concepts ที่คลุมเครือ (e.g., "small window", "cozy room")
# 3. **Computational Cost**: ใหญ่และช้ากว่า YOLO
# 4. **Rare Concepts**: อาจไม่ดีกับ Concepts ที่หายาก

# %% [markdown]
# ### 9.2 Common Errors and Solutions

# %%
# Error 1: TypeError: 'SimpleTokenizer' object is not callable
# Solution:
# !pip uninstall clip -y
# !pip install git+https://github.com/ultralytics/CLIP.git

# Error 2: Model weights not found
# Solution: ดาวน์โหลดจาก HuggingFace และวางใน Correct Path

# Error 3: Out of Memory
# Solution: 
# - ลด imgsz (e.g., 640 → 480)
# - ใช้ half=True (FP16)
# - ลด Batch Size

# Error 4: Slow Inference
# Solutions:
# - ใช้ Feature Reuse (inference_features)
# - ลด imgsz
# - ใช้ GPU ที่เร็วกว่า (H100, A100)

# %% [markdown]
# ### 💡 ปัญหาที่ 9: Optimization Challenge
#
# **โจทย์:** Optimize SAM 3 Pipeline สำหรับ Production
#
# **Constraints:**
# - Real-time (> 10 FPS) บน GPU T4
# - Memory < 8 GB
# - Accuracy > 90% ของ Original
#
# **Techniques:**
# 1. Model Quantization (INT8)
# 2. Feature Caching
# 3. Dynamic Resolution
# 4. Batch Processing

# %%
# TODO: Implement Optimization Pipeline
# 1. Profile Current Performance
# 2. Apply Optimizations
# 3. Benchmark และเปรียบเทียบ

# %% [markdown]
# ---
#
# ## ส่วนที่ 10: Integration with MLOps Pipeline
#
# นำ SAM 3 มาใช้ใน Production Environment

# %% [markdown]
# ### 10.1 Model Serving with MLflow

# %%
import mlflow
from mlflow.models import infer_signature

# Log SAM 3 Model to MLflow
with mlflow.start_run(run_name="SAM3-Experiment"):
    # Log Parameters
    mlflow.log_params({
        "model": "sam3.pt",
        "conf_threshold": 0.25,
        "half_precision": True
    })
    
    # Log Metrics
    mlflow.log_metrics({
        "cgf1": 65.0,  # Example
        "inference_time_ms": 30
    })
    
    # Log Model
    # mlflow.pytorch.log_model(model, "sam3-model")

# %% [markdown]
# ### 10.2 Docker Deployment

# %%
# Dockerfile Example
dockerfile_content = """
FROM ultralytics/ultralytics:latest

# Install Dependencies
RUN pip install mlflow

# Copy Model Weights
COPY sam3.pt /app/sam3.pt

# Copy Inference Script
COPY inference.py /app/inference.py

# Expose API Port
EXPOSE 8000

# Run API Server
CMD ["python", "/app/inference.py"]
"""

# %%
# inference.py Example
inference_script = """
from fastapi import FastAPI, File, UploadFile
from ultralytics.models.sam import SAM3SemanticPredictor
import cv2
import numpy as np

app = FastAPI()
predictor = SAM3SemanticPredictor(overrides={"model": "sam3.pt"})

@app.post("/predict")
async def predict(file: UploadFile, text: str):
    # Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Predict
    predictor.set_image(img)
    results = predictor(text=[text])
    
    # Return Results
    return {
        "num_objects": len(results[0].masks) if results[0].masks else 0,
        "masks": results[0].masks.tolist() if results[0].masks else []
    }
"""

# %% [markdown]
# ### 💡 ปัญหาที่ 10: End-to-End MLOps Pipeline
#
# **โจทย์:** สร้าง Complete MLOps Pipeline สำหรับ SAM 3
#
# **Components:**
# 1. **Data Pipeline**: Collect และ Annotate Dataset
# 2. **Training/Fine-tuning**: (ถ้าจำเป็น)
# 3. **Experiment Tracking**: MLflow
# 4. **Model Registry**: Store Model Versions
# 5. **Deployment**: Docker + Kubernetes
# 6. **Monitoring**: Track Performance Metrics
# 7. **CI/CD**: Automated Testing และ Deployment
#
# **Deliverables:**
# - Architecture Diagram
# - Code Implementation
# - Deployment Scripts
# - Monitoring Dashboard

# %%
# TODO: Design และ Implement MLOps Pipeline

# %% [markdown]
# ---
#
# ## สรุป (Summary)
#
# ใน Lab นี้ เราได้เรียนรู้:
#
# 1. ✅ ความแตกต่างระหว่าง SAM 2 และ SAM 3
# 2. ✅ Text-based Concept Segmentation
# 3. ✅ Image Exemplar Prompts และ Interactive Refinement
# 4. ✅ Feature Reuse สำหรับประสิทธิภาพ
# 5. ✅ Video Tracking ด้วย Text และ Visual Prompts
# 6. ✅ Performance Metrics (CGF1, pmF1, IL_MCC)
# 7. ✅ Advanced Applications (Counting, Analytics, Reasoning)
# 8. ✅ Optimization และ Production Deployment
# 9. ✅ MLOps Integration
#
# ### Key Takeaways:
#
# - **SAM 3** เหมาะสำหรับ Open-vocabulary Tasks ที่ต้องการ Flexibility
# - **YOLO11** เหมาะสำหรับ Production ที่ต้องการ Speed และ Efficiency
# - **Feature Reuse** ช่วยเพิ่มประสิทธิภาพได้มาก
# - **Interactive Refinement** ทำให้ Accuracy ดีขึ้นอย่างมีนัยสำคัญ
# - **SAM 3 Agent** (MLLM) ขยายความสามารถสู่ Complex Reasoning
#
# ---
#
# ## แบบฝึกหัดเพิ่มเติม (Additional Exercises)
#
# 1. **Exercise 1**: Fine-tune SAM 3 บน Domain-specific Dataset
# 2. **Exercise 2**: สร้าง Real-time Dashboard สำหรับ Video Analytics
# 3. **Exercise 3**: เปรียบเทียบ SAM 3 กับ YOLO-World
# 4. **Exercise 4**: Implement Active Learning Pipeline
# 5. **Exercise 5**: Deploy บน Edge Device (Jetson, Raspberry Pi)
#
# ---
#
# ## อ้างอิง (References)
#
# 1. SAM 3 Paper: https://openreview.net/forum?id=r35clVtGzw
# 2. Ultralytics Documentation: https://docs.ultralytics.com/models/sam-3/
# 3. SAM 2 Documentation: https://docs.ultralytics.com/models/sam-2/
# 4. YOLO11 Documentation: https://docs.ultralytics.com/models/yolo11/
#
# ---
#
# **จัดทำโดย:** [ชื่ออาจารย์]  
# **Course:** Advanced Computer Vision with MLOps  
# **ปีการศึกษา:** 2567
