# %% [markdown]
# # 🥊 Pose Time Series Analysis Lab: การวิเคราะห์ท่ามวยไทยด้วย Time Series และ Angle Analysis
#
# ---
#
# ## 📋 Lab Overview
#
# **วัตถุประสงค์การเรียนรู้ (Learning Objectives):**
# 1. เข้าใจโครงสร้างข้อมูล Pose Estimation ในรูปแบบ Time Series
# 2. วิเคราะห์และเปรียบเทียบการเคลื่อนไหวของบุคคลต่างๆ ข้ามเวลา
# 3. คำนวณ Joint Angles จาก Keypoint Coordinates
# 4. สร้าง Visualization เพื่อเปรียบเทียบท่าทางต่างๆ (Actions)
# 5. วิเคราะห์ Pattern ของท่ามวยไทยผ่าน Time Series และ Angle Analysis
#
# **Prerequisites:**
# - Python 3.8+
# - ความเข้าใจพื้นฐานเรื่อง Pandas DataFrame
# - ความเข้าใจพื้นฐานเรื่อง Trigonometry (สำหรับการคำนวณมุม)
#
# **Estimated Time:** 2-3 hours
#
# ---

# %% [markdown]
# ## 📚 Part 1: Environment Setup และ Data Loading
#
# ### 1.1 Import Libraries
#
# นำเข้า libraries ที่จำเป็นสำหรับการวิเคราะห์ Time Series และการสร้าง Visualization

# %%
# =====================================================
# STEP 1.1: Import Required Libraries
# =====================================================
# Description: นำเข้า libraries ทั้งหมดที่จำเป็น
# - pandas: สำหรับจัดการข้อมูล DataFrame
# - numpy: สำหรับการคำนวณเชิงตัวเลข
# - matplotlib & seaborn: สำหรับสร้างกราฟ
# - scipy: สำหรับ signal processing (smoothing)
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.ndimage import uniform_filter1d
import warnings

# ตั้งค่า Visualization
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
sns.set_style("whitegrid")
warnings.filterwarnings('ignore')

# สำหรับแสดงผลภาษาไทย (ถ้ามี font)
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
except:
    pass

print("✅ Libraries imported successfully!")
print(f"   📊 Pandas version: {pd.__version__}")
print(f"   🔢 NumPy version: {np.__version__}")

# %% [markdown]
# ### 1.2 Load Pose Data
#
# โหลดข้อมูล Pose Estimation จากไฟล์ CSV
# 
# **Data Structure:**
# - `frame_idx`: หมายเลขเฟรม (ดัชนีเวลา)
# - `timestamp`: เวลาในหน่วยวินาที
# - `person_id`: รหัสบุคคลที่ถูกติดตาม
# - `keypoint_x`, `keypoint_y`: พิกัดของแต่ละ keypoint
# - `keypoint_conf`: ความเชื่อมั่นของแต่ละ keypoint
# - `action`: ป้ายกำกับท่าทาง (Label)

# %%
# =====================================================
# STEP 1.2: Load Pose Data from CSV
# =====================================================
# Description: โหลดข้อมูลจากไฟล์ CSV ที่เตรียมไว้
# ไฟล์นี้ประกอบด้วยข้อมูล pose estimation ของหลายคน
# ที่ถูก track ข้ามเฟรม พร้อม action labels
# =====================================================

# โหลดข้อมูล
df_pose = pd.read_csv('pose_data.csv')

# แสดงข้อมูลเบื้องต้น
print("=" * 70)
print("📊 POSE DATA OVERVIEW")
print("=" * 70)
print(f"\n📁 Dataset Shape: {df_pose.shape[0]:,} rows × {df_pose.shape[1]} columns")
print(f"📅 Frame Range: {df_pose['frame_idx'].min()} to {df_pose['frame_idx'].max()}")
print(f"⏱️  Time Range: {df_pose['timestamp'].min():.2f}s to {df_pose['timestamp'].max():.2f}s")
print(f"👥 Unique Persons: {df_pose['person_id'].nunique()}")
print(f"🎯 Actions: {df_pose['action'].nunique()}")

print("\n" + "=" * 70)
print("📋 DATA SAMPLE (First 5 rows)")
print("=" * 70)
df_pose.head()

# %%
# =====================================================
# STEP 1.3: Data Info Summary
# =====================================================
# Description: แสดงรายละเอียดของ columns ทั้งหมด
# เพื่อให้เข้าใจโครงสร้างข้อมูล
# =====================================================

print("=" * 70)
print("📋 DATA STRUCTURE INFO")
print("=" * 70)
df_pose.info()

# %% [markdown]
# ### 1.3 Define Constants
#
# กำหนดค่าคงที่สำหรับ COCO Keypoint format และ Skeleton Connections

# %%
# =====================================================
# STEP 1.4: Define COCO Keypoint Constants
# =====================================================
# Description: กำหนดชื่อ Keypoints ตามมาตรฐาน COCO
# COCO format มี 17 keypoints ครอบคลุมทั้งร่างกาย
# =====================================================

# รายชื่อ Keypoints (17 จุด)
KEYPOINT_NAMES = [
    "nose",           # 0 - จมูก
    "left_eye",       # 1 - ตาซ้าย
    "right_eye",      # 2 - ตาขวา
    "left_ear",       # 3 - หูซ้าย
    "right_ear",      # 4 - หูขวา
    "left_shoulder",  # 5 - ไหล่ซ้าย
    "right_shoulder", # 6 - ไหล่ขวา
    "left_elbow",     # 7 - ข้อศอกซ้าย
    "right_elbow",    # 8 - ข้อศอกขวา
    "left_wrist",     # 9 - ข้อมือซ้าย
    "right_wrist",    # 10 - ข้อมือขวา
    "left_hip",       # 11 - สะโพกซ้าย
    "right_hip",      # 12 - สะโพกขวา
    "left_knee",      # 13 - เข่าซ้าย
    "right_knee",     # 14 - เข่าขวา
    "left_ankle",     # 15 - ข้อเท้าซ้าย
    "right_ankle"     # 16 - ข้อเท้าขวา
]

# การจัดกลุ่มส่วนต่างๆ ของร่างกาย
BODY_PARTS = {
    'head': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
    'upper_body': ['left_shoulder', 'right_shoulder', 'left_elbow', 
                   'right_elbow', 'left_wrist', 'right_wrist'],
    'lower_body': ['left_hip', 'right_hip', 'left_knee', 
                   'right_knee', 'left_ankle', 'right_ankle'],
    'left_arm': ['left_shoulder', 'left_elbow', 'left_wrist'],
    'right_arm': ['right_shoulder', 'right_elbow', 'right_wrist'],
    'left_leg': ['left_hip', 'left_knee', 'left_ankle'],
    'right_leg': ['right_hip', 'right_knee', 'right_ankle'],
    'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
}

# เส้นเชื่อมสำหรับ Skeleton Visualization
SKELETON_CONNECTIONS = [
    # Head connections
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    # Upper body
    ('left_shoulder', 'right_shoulder'),  # shoulders
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    # Torso
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    # Lower body
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
]

print("✅ Constants defined!")
print(f"   📍 Total Keypoints: {len(KEYPOINT_NAMES)}")
print(f"   🦴 Body Parts Groups: {list(BODY_PARTS.keys())}")
print(f"   🔗 Skeleton Connections: {len(SKELETON_CONNECTIONS)}")

# %% [markdown]
# ---
#
# ## 📚 Part 2: Exploratory Data Analysis (EDA)
#
# ### 2.1 Action Distribution Analysis
#
# วิเคราะห์การกระจายตัวของ Actions ต่างๆ ในข้อมูล

# %%
# =====================================================
# STEP 2.1: Action Distribution Analysis
# =====================================================
# Description: วิเคราะห์จำนวนและสัดส่วนของแต่ละ Action
# เพื่อเข้าใจความสมดุลของข้อมูล
# =====================================================

# นับจำนวนแต่ละ Action
action_counts = df_pose['action'].value_counts()
action_percentages = df_pose['action'].value_counts(normalize=True) * 100

print("=" * 70)
print("🎯 ACTION DISTRIBUTION")
print("=" * 70)

# แสดงตาราง
action_summary = pd.DataFrame({
    'Action': action_counts.index,
    'Count': action_counts.values,
    'Percentage (%)': action_percentages.values.round(2)
})
print(action_summary.to_string(index=False))

# สร้าง Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar Chart
colors = plt.cm.Set3(np.linspace(0, 1, len(action_counts)))
bars = axes[0].barh(action_counts.index, action_counts.values, color=colors)
axes[0].set_xlabel('Number of Frames')
axes[0].set_title('📊 Action Distribution (Bar Chart)')
axes[0].invert_yaxis()

# เพิ่มตัวเลขบน bar
for bar, count in zip(bars, action_counts.values):
    axes[0].text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2, 
                 f'{count:,}', va='center', fontsize=10)

# Pie Chart
axes[1].pie(action_counts.values, labels=action_counts.index, 
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('📊 Action Distribution (Pie Chart)')

plt.tight_layout()
plt.savefig('01_action_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Chart saved: 01_action_distribution.png")

# %% [markdown]
# ### 2.2 Person ID Analysis
#
# วิเคราะห์จำนวน Frames ต่อ Person และ Person ต่อ Action

# %%
# =====================================================
# STEP 2.2: Person ID Analysis
# =====================================================
# Description: วิเคราะห์ว่าแต่ละ Person มีกี่ frames
# และแต่ละ Action มี Person กี่คน
# =====================================================

# นับ frames ต่อ person
person_frame_counts = df_pose.groupby('person_id').size().reset_index(name='frame_count')
person_frame_counts = person_frame_counts.sort_values('frame_count', ascending=False)

print("=" * 70)
print("👥 PERSON ID ANALYSIS")
print("=" * 70)

print("\n📋 Frames per Person (Top 10):")
print(person_frame_counts.head(10).to_string(index=False))

print(f"\n📊 Statistics:")
print(f"   Total Unique Persons: {len(person_frame_counts)}")
print(f"   Max Frames per Person: {person_frame_counts['frame_count'].max():,}")
print(f"   Min Frames per Person: {person_frame_counts['frame_count'].min():,}")
print(f"   Mean Frames per Person: {person_frame_counts['frame_count'].mean():.2f}")

# Person distribution per action
print("\n" + "=" * 70)
print("🎯 PERSONS PER ACTION")
print("=" * 70)

person_per_action = df_pose.groupby('action')['person_id'].nunique().reset_index()
person_per_action.columns = ['Action', 'Unique Persons']
print(person_per_action.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Frames per person (top 15)
top_persons = person_frame_counts.head(15)
axes[0].bar(top_persons['person_id'].astype(str), top_persons['frame_count'], 
            color='steelblue', edgecolor='navy')
axes[0].set_xlabel('Person ID')
axes[0].set_ylabel('Number of Frames')
axes[0].set_title('👥 Frames per Person (Top 15)')
axes[0].tick_params(axis='x', rotation=45)

# Persons per action
colors = plt.cm.Set2(np.linspace(0, 1, len(person_per_action)))
axes[1].bar(person_per_action['Action'], person_per_action['Unique Persons'], 
            color=colors, edgecolor='black')
axes[1].set_xlabel('Action')
axes[1].set_ylabel('Number of Unique Persons')
axes[1].set_title('🎯 Unique Persons per Action')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('02_person_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Chart saved: 02_person_analysis.png")

# %% [markdown]
# ### 2.3 Select 3 Representative Person IDs for Analysis
#
# เลือก 3 Person IDs ที่มีข้อมูลครบถ้วนสำหรับการวิเคราะห์เปรียบเทียบ

# %%
# =====================================================
# STEP 2.3: Select 3 Representative Person IDs
# =====================================================
# Description: เลือก 3 Person IDs ที่มี frames มากที่สุด
# และปรากฏใน Action labels (ไม่ใช่ Unknown)
# เพื่อใช้ในการวิเคราะห์เปรียบเทียบ
# =====================================================

# กรองเฉพาะ frames ที่มี label (ไม่ใช่ Unknown)
df_labeled = df_pose[df_pose['action'] != 'Unknown'].copy()

# นับ frames ต่อ person สำหรับข้อมูลที่มี label
person_labeled_counts = df_labeled.groupby('person_id').agg({
    'frame_idx': 'count',
    'action': lambda x: x.nunique()
}).reset_index()
person_labeled_counts.columns = ['person_id', 'labeled_frames', 'actions_count']
person_labeled_counts = person_labeled_counts.sort_values('labeled_frames', ascending=False)

print("=" * 70)
print("🎯 SELECTING REPRESENTATIVE PERSON IDs")
print("=" * 70)

print("\n📋 Persons with Labeled Frames (Top 10):")
print(person_labeled_counts.head(10).to_string(index=False))

# เลือก 3 person IDs ที่มี frames มากที่สุด
SELECTED_PERSON_IDS = person_labeled_counts.head(3)['person_id'].tolist()

print(f"\n✅ Selected Person IDs for Analysis: {SELECTED_PERSON_IDS}")

# แสดงรายละเอียดของ selected persons
print("\n" + "=" * 70)
print("📊 SELECTED PERSONS DETAILS")
print("=" * 70)

for pid in SELECTED_PERSON_IDS:
    person_data = df_labeled[df_labeled['person_id'] == pid]
    actions = person_data['action'].unique()
    print(f"\n👤 Person ID: {pid}")
    print(f"   Total Labeled Frames: {len(person_data):,}")
    print(f"   Actions: {list(actions)}")
    print(f"   Frame Range: {person_data['frame_idx'].min()} - {person_data['frame_idx'].max()}")
    print(f"   Time Range: {person_data['timestamp'].min():.2f}s - {person_data['timestamp'].max():.2f}s")

# %% [markdown]
# ---
#
# ## 📚 Part 3: Time Series Analysis
#
# ### 3.1 Keypoint Position Time Series
#
# วิเคราะห์การเปลี่ยนแปลงตำแหน่ง Keypoints ตามเวลาสำหรับแต่ละ Action

# %%
# =====================================================
# STEP 3.1: Keypoint Position Time Series Analysis
# =====================================================
# Description: วิเคราะห์การเคลื่อนที่ของ keypoints หลักๆ
# ตามเวลา เพื่อเข้าใจ pattern การเคลื่อนไหวในแต่ละท่า
# 
# Key Keypoints สำหรับการวิเคราะห์:
# - wrists: การเคลื่อนไหวของมือ (สำคัญสำหรับหมัด)
# - ankles: การเคลื่อนไหวของเท้า (สำคัญสำหรับการเตะ)
# - nose: การเคลื่อนไหวของศีรษะ
# =====================================================

# Keypoints ที่น่าสนใจสำหรับการวิเคราะห์ท่ามวย
ANALYSIS_KEYPOINTS = ['right_wrist', 'left_wrist', 'right_ankle', 'left_ankle', 'nose']

def plot_keypoint_timeseries(df, person_id, keypoints, figsize=(16, 12)):
    """
    สร้างกราฟ Time Series ของตำแหน่ง Keypoints
    
    Parameters:
    -----------
    df : DataFrame - ข้อมูล pose
    person_id : int - Person ID ที่ต้องการวิเคราะห์
    keypoints : list - รายชื่อ keypoints
    figsize : tuple - ขนาดรูป
    """
    # กรองข้อมูลสำหรับ person นี้
    person_data = df[df['person_id'] == person_id].copy()
    person_data = person_data.sort_values('frame_idx')
    
    # สร้าง figure
    fig, axes = plt.subplots(len(keypoints), 2, figsize=figsize)
    fig.suptitle(f'👤 Keypoint Time Series - Person ID: {person_id}', 
                 fontsize=16, fontweight='bold')
    
    # สร้าง color map สำหรับ actions
    actions = person_data['action'].unique()
    action_colors = dict(zip(actions, plt.cm.Set2(np.linspace(0, 1, len(actions)))))
    
    for idx, kpt in enumerate(keypoints):
        # Plot X coordinate
        ax_x = axes[idx, 0]
        for action in actions:
            action_data = person_data[person_data['action'] == action]
            ax_x.scatter(action_data['timestamp'], action_data[f'{kpt}_x'], 
                        label=action, alpha=0.6, s=2, color=action_colors[action])
        
        ax_x.set_ylabel(f'{kpt}\nX Position (pixels)')
        ax_x.set_title(f'{kpt.replace("_", " ").title()} - X Coordinate')
        if idx == 0:
            ax_x.legend(loc='upper right', fontsize=8, markerscale=5)
        
        # Plot Y coordinate
        ax_y = axes[idx, 1]
        for action in actions:
            action_data = person_data[person_data['action'] == action]
            ax_y.scatter(action_data['timestamp'], action_data[f'{kpt}_y'], 
                        label=action, alpha=0.6, s=2, color=action_colors[action])
        
        ax_y.set_ylabel(f'{kpt}\nY Position (pixels)')
        ax_y.set_title(f'{kpt.replace("_", " ").title()} - Y Coordinate')
        
        # กลับแกน Y เพราะ pixel coordinate มี Y เพิ่มขึ้นเมื่อไปทางล่าง
        ax_y.invert_yaxis()
    
    # Set x labels for bottom row
    axes[-1, 0].set_xlabel('Time (seconds)')
    axes[-1, 1].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    return fig

# สร้างกราฟสำหรับ Person ID แรกที่เลือก
print("=" * 70)
print("📈 KEYPOINT TIME SERIES VISUALIZATION")
print("=" * 70)

for pid in SELECTED_PERSON_IDS[:1]:  # แสดงเฉพาะ person แรก
    fig = plot_keypoint_timeseries(df_labeled, pid, ANALYSIS_KEYPOINTS)
    plt.savefig(f'03_keypoint_timeseries_person_{pid}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Chart saved: 03_keypoint_timeseries_person_{pid}.png")

# %% [markdown]
# ### 3.2 Compare Keypoint Trajectories Across Actions
#
# เปรียบเทียบ Trajectory ของ Keypoints ในแต่ละ Action

# %%
# =====================================================
# STEP 3.2: Compare Keypoint Trajectories Across Actions
# =====================================================
# Description: เปรียบเทียบการเคลื่อนที่ของ keypoints
# ในแต่ละ action เพื่อหา pattern ที่แตกต่างกัน
# =====================================================

def plot_action_comparison(df, person_ids, keypoint, figsize=(16, 10)):
    """
    เปรียบเทียบ trajectory ของ keypoint ในแต่ละ action
    สำหรับ selected persons
    
    Parameters:
    -----------
    df : DataFrame - ข้อมูล pose
    person_ids : list - รายการ person IDs
    keypoint : str - ชื่อ keypoint
    figsize : tuple - ขนาดรูป
    """
    # กรองข้อมูล
    df_selected = df[df['person_id'].isin(person_ids)].copy()
    actions = [a for a in df_selected['action'].unique() if a != 'Unknown']
    
    # สร้าง figure
    n_actions = len(actions)
    n_cols = 3
    n_rows = (n_actions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    fig.suptitle(f'🎯 {keypoint.replace("_", " ").title()} Trajectory Comparison\n'
                 f'Selected Persons: {person_ids}', 
                 fontsize=14, fontweight='bold')
    
    # Color map สำหรับ persons
    person_colors = dict(zip(person_ids, ['#e74c3c', '#3498db', '#2ecc71']))
    
    for idx, action in enumerate(actions):
        ax = axes[idx]
        action_data = df_selected[df_selected['action'] == action]
        
        for pid in person_ids:
            person_data = action_data[action_data['person_id'] == pid]
            if len(person_data) > 0:
                # Normalize time to start from 0
                person_data = person_data.sort_values('timestamp')
                time_normalized = person_data['timestamp'] - person_data['timestamp'].min()
                
                ax.plot(time_normalized, person_data[f'{keypoint}_x'], 
                       label=f'Person {pid} (X)', linestyle='-', alpha=0.8,
                       color=person_colors[pid])
                ax.plot(time_normalized, person_data[f'{keypoint}_y'], 
                       label=f'Person {pid} (Y)', linestyle='--', alpha=0.8,
                       color=person_colors[pid])
        
        ax.set_title(action.replace('_', ' '), fontsize=11)
        ax.set_xlabel('Normalized Time (s)')
        ax.set_ylabel('Position (pixels)')
        
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    # ซ่อน axes ที่ไม่ใช้
    for idx in range(len(actions), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

# สร้างกราฟเปรียบเทียบสำหรับ right_wrist (สำคัญสำหรับหมัด)
print("=" * 70)
print("📊 ACTION TRAJECTORY COMPARISON")
print("=" * 70)

for kpt in ['right_wrist', 'right_ankle']:
    fig = plot_action_comparison(df_labeled, SELECTED_PERSON_IDS, kpt)
    filename = f'04_action_comparison_{kpt}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Chart saved: {filename}")

# %% [markdown]
# ### 3.3 Velocity Analysis
#
# วิเคราะห์ความเร็วของ Keypoints เพื่อหา pattern การเคลื่อนไหวที่รวดเร็ว (เช่น หมัด, เตะ)

# %%
# =====================================================
# STEP 3.3: Velocity Analysis
# =====================================================
# Description: คำนวณความเร็วของ keypoints
# ความเร็ว = การเปลี่ยนแปลงตำแหน่งต่อเวลา
# 
# Formula: velocity = sqrt((dx/dt)^2 + (dy/dt)^2)
# 
# ความเร็วสูง = การเคลื่อนไหวที่รวดเร็ว (หมัด, เตะ)
# ความเร็วต่ำ = การเคลื่อนไหวช้าๆ (ท่าตั้งรับ)
# =====================================================

def calculate_velocity(df, person_id, keypoint, window_size=3):
    """
    คำนวณความเร็วของ keypoint
    
    Parameters:
    -----------
    df : DataFrame - ข้อมูล pose
    person_id : int - Person ID
    keypoint : str - ชื่อ keypoint
    window_size : int - ขนาด window สำหรับ smoothing
    
    Returns:
    --------
    DataFrame พร้อมคอลัมน์ velocity
    """
    # กรองและ sort ข้อมูล
    person_data = df[df['person_id'] == person_id].copy()
    person_data = person_data.sort_values('timestamp').reset_index(drop=True)
    
    # คำนวณ dx, dy, dt
    person_data['dx'] = person_data[f'{keypoint}_x'].diff()
    person_data['dy'] = person_data[f'{keypoint}_y'].diff()
    person_data['dt'] = person_data['timestamp'].diff()
    
    # คำนวณ velocity
    person_data['velocity'] = np.sqrt(
        (person_data['dx'] / person_data['dt'])**2 + 
        (person_data['dy'] / person_data['dt'])**2
    )
    
    # Smooth velocity
    if window_size > 1 and len(person_data) > window_size:
        person_data['velocity_smooth'] = uniform_filter1d(
            person_data['velocity'].fillna(0), size=window_size
        )
    else:
        person_data['velocity_smooth'] = person_data['velocity']
    
    return person_data

def plot_velocity_analysis(df, person_ids, keypoint, figsize=(16, 10)):
    """
    สร้างกราฟวิเคราะห์ความเร็วเปรียบเทียบระหว่าง actions
    """
    actions = [a for a in df['action'].unique() if a != 'Unknown']
    
    fig, axes = plt.subplots(len(person_ids), 1, figsize=figsize, sharex=True)
    if len(person_ids) == 1:
        axes = [axes]
    
    fig.suptitle(f'⚡ Velocity Analysis - {keypoint.replace("_", " ").title()}\n'
                 f'Higher velocity = Faster movement (e.g., punch, kick)', 
                 fontsize=14, fontweight='bold')
    
    # Color map สำหรับ actions
    action_colors = dict(zip(actions, plt.cm.tab10(np.linspace(0, 1, len(actions)))))
    
    for idx, pid in enumerate(person_ids):
        ax = axes[idx]
        
        # คำนวณ velocity
        vel_data = calculate_velocity(df, pid, keypoint, window_size=5)
        vel_data = vel_data[vel_data['action'] != 'Unknown']
        
        # Plot velocity for each action
        for action in actions:
            action_data = vel_data[vel_data['action'] == action]
            if len(action_data) > 0:
                ax.plot(action_data['timestamp'], action_data['velocity_smooth'],
                       label=action, alpha=0.7, linewidth=0.8,
                       color=action_colors[action])
        
        ax.set_ylabel(f'Person {pid}\nVelocity (px/s)')
        ax.set_ylim(0, vel_data['velocity_smooth'].quantile(0.99) * 1.1)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    axes[-1].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    return fig

# วิเคราะห์ความเร็วสำหรับ wrists และ ankles
print("=" * 70)
print("⚡ VELOCITY ANALYSIS")
print("=" * 70)

for kpt in ['right_wrist', 'left_wrist']:
    fig = plot_velocity_analysis(df_labeled, SELECTED_PERSON_IDS, kpt)
    filename = f'05_velocity_{kpt}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Chart saved: {filename}")

# %% [markdown]
# ### 3.4 Velocity Statistics per Action
#
# สรุปสถิติความเร็วเฉลี่ยของแต่ละ Action เพื่อจัดอันดับท่าที่เคลื่อนไหวเร็วที่สุด

# %%
# =====================================================
# STEP 3.4: Velocity Statistics per Action
# =====================================================
# Description: คำนวณและเปรียบเทียบสถิติความเร็ว
# ของแต่ละ action เพื่อระบุท่าที่เคลื่อนไหวเร็ว/ช้า
# =====================================================

def calculate_action_velocity_stats(df, person_ids, keypoints):
    """
    คำนวณสถิติความเร็วต่อ action
    """
    results = []
    
    for pid in person_ids:
        for kpt in keypoints:
            vel_data = calculate_velocity(df, pid, kpt, window_size=5)
            vel_data = vel_data[vel_data['action'] != 'Unknown']
            
            # คำนวณสถิติต่อ action
            for action in vel_data['action'].unique():
                action_data = vel_data[vel_data['action'] == action]['velocity_smooth']
                action_data = action_data.dropna()
                
                if len(action_data) > 0:
                    results.append({
                        'person_id': pid,
                        'keypoint': kpt,
                        'action': action,
                        'mean_velocity': action_data.mean(),
                        'max_velocity': action_data.max(),
                        'std_velocity': action_data.std(),
                        'median_velocity': action_data.median()
                    })
    
    return pd.DataFrame(results)

# คำนวณสถิติ
velocity_stats = calculate_action_velocity_stats(
    df_labeled, 
    SELECTED_PERSON_IDS, 
    ['right_wrist', 'left_wrist', 'right_ankle', 'left_ankle']
)

print("=" * 70)
print("📊 VELOCITY STATISTICS BY ACTION")
print("=" * 70)

# เฉลี่ยข้าม persons และ keypoints
action_velocity_summary = velocity_stats.groupby('action').agg({
    'mean_velocity': 'mean',
    'max_velocity': 'mean',
    'std_velocity': 'mean'
}).round(2)

action_velocity_summary = action_velocity_summary.sort_values('mean_velocity', ascending=False)
print("\n📋 Average Velocity Statistics by Action:")
print(action_velocity_summary.to_string())

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mean velocity by action
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(action_velocity_summary)))[::-1]
bars = axes[0].barh(action_velocity_summary.index, 
                    action_velocity_summary['mean_velocity'],
                    color=colors, edgecolor='black')
axes[0].set_xlabel('Mean Velocity (pixels/second)')
axes[0].set_title('📊 Mean Velocity by Action\n(Higher = Faster movement)')
axes[0].invert_yaxis()

# Max velocity by action
bars = axes[1].barh(action_velocity_summary.index, 
                    action_velocity_summary['max_velocity'],
                    color=colors, edgecolor='black')
axes[1].set_xlabel('Max Velocity (pixels/second)')
axes[1].set_title('📊 Max Velocity by Action\n(Peak movement speed)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('06_velocity_stats.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Chart saved: 06_velocity_stats.png")

# %% [markdown]
# ---
#
# ## 📚 Part 4: Joint Angle Analysis
#
# ### 4.1 Understanding Joint Angles
#
# **Joint Angle คืออะไร?**
# - มุมที่เกิดจากข้อต่อ 3 จุด (เช่น ไหล่-ข้อศอก-ข้อมือ)
# - ใช้ในการวิเคราะห์ท่าทางและการเคลื่อนไหว
# - สำคัญมากในกีฬาและการวิเคราะห์การเคลื่อนไหว
#
# **Angles ที่จะคำนวณ:**
# 1. Elbow Angle (มุมข้อศอก): shoulder-elbow-wrist
# 2. Knee Angle (มุมเข่า): hip-knee-ankle
# 3. Shoulder Angle (มุมไหล่): elbow-shoulder-hip
# 4. Hip Angle (มุมสะโพก): shoulder-hip-knee

# %%
# =====================================================
# STEP 4.1: Joint Angle Calculation Functions
# =====================================================
# Description: สร้างฟังก์ชันคำนวณมุมข้อต่อ
# 
# หลักการ: ใช้ dot product และ cross product
# angle = atan2(cross_product, dot_product)
# 
# ผลลัพธ์: มุมเป็นองศา (degrees) 0-180°
# =====================================================

def calculate_angle(p1, p2, p3):
    """
    คำนวณมุมที่จุด p2 จากเส้น p1-p2 และ p2-p3
    
    Parameters:
    -----------
    p1, p2, p3 : np.array - พิกัด (x, y) ของแต่ละจุด
    
    Returns:
    --------
    angle : float - มุมเป็นองศา (0-180)
    
    Diagram:
           p1
            \
             \  angle
              p2-------p3
    """
    # Vector จาก p2 ไป p1 และ p3
    v1 = p1 - p2
    v2 = p3 - p2
    
    # ตรวจสอบว่ามีค่า valid หรือไม่
    if np.any(p1 == 0) or np.any(p2 == 0) or np.any(p3 == 0):
        return np.nan
    
    # คำนวณ dot product และ cross product
    dot = np.dot(v1, v2)
    cross = np.cross(v1, v2)
    
    # คำนวณมุม (radians)
    angle_rad = np.arctan2(np.abs(cross), dot)
    
    # แปลงเป็น degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_all_angles(row):
    """
    คำนวณมุมข้อต่อทั้งหมดสำหรับ 1 row
    
    Returns:
    --------
    dict ของมุมต่างๆ
    """
    angles = {}
    
    # ดึงพิกัด keypoints
    def get_point(name):
        return np.array([row[f'{name}_x'], row[f'{name}_y']])
    
    # ===== Elbow Angles (มุมข้อศอก) =====
    # Left Elbow: shoulder-elbow-wrist
    angles['left_elbow_angle'] = calculate_angle(
        get_point('left_shoulder'),
        get_point('left_elbow'),
        get_point('left_wrist')
    )
    
    # Right Elbow
    angles['right_elbow_angle'] = calculate_angle(
        get_point('right_shoulder'),
        get_point('right_elbow'),
        get_point('right_wrist')
    )
    
    # ===== Knee Angles (มุมเข่า) =====
    # Left Knee: hip-knee-ankle
    angles['left_knee_angle'] = calculate_angle(
        get_point('left_hip'),
        get_point('left_knee'),
        get_point('left_ankle')
    )
    
    # Right Knee
    angles['right_knee_angle'] = calculate_angle(
        get_point('right_hip'),
        get_point('right_knee'),
        get_point('right_ankle')
    )
    
    # ===== Shoulder Angles (มุมไหล่) =====
    # Left Shoulder: elbow-shoulder-hip
    angles['left_shoulder_angle'] = calculate_angle(
        get_point('left_elbow'),
        get_point('left_shoulder'),
        get_point('left_hip')
    )
    
    # Right Shoulder
    angles['right_shoulder_angle'] = calculate_angle(
        get_point('right_elbow'),
        get_point('right_shoulder'),
        get_point('right_hip')
    )
    
    # ===== Hip Angles (มุมสะโพก) =====
    # Left Hip: shoulder-hip-knee
    angles['left_hip_angle'] = calculate_angle(
        get_point('left_shoulder'),
        get_point('left_hip'),
        get_point('left_knee')
    )
    
    # Right Hip
    angles['right_hip_angle'] = calculate_angle(
        get_point('right_shoulder'),
        get_point('right_hip'),
        get_point('right_knee')
    )
    
    # ===== Torso Angle (มุมลำตัว) =====
    # คำนวณจากเส้นที่เชื่อม shoulder กับ hip
    shoulder_mid = (get_point('left_shoulder') + get_point('right_shoulder')) / 2
    hip_mid = (get_point('left_hip') + get_point('right_hip')) / 2
    
    # มุมของลำตัวเทียบกับแนวตั้ง
    vertical = np.array([0, -1])  # แกน Y ชี้ขึ้น (แต่ใน pixel coordinate Y ลงล่าง)
    torso_vector = shoulder_mid - hip_mid
    
    if np.linalg.norm(torso_vector) > 0:
        torso_vector_norm = torso_vector / np.linalg.norm(torso_vector)
        dot = np.dot(torso_vector_norm, vertical)
        angles['torso_lean_angle'] = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    else:
        angles['torso_lean_angle'] = np.nan
    
    return angles

print("✅ Angle calculation functions defined!")
print("\n📐 Angles to be calculated:")
print("   - Elbow Angle: shoulder-elbow-wrist (มุมข้อศอก)")
print("   - Knee Angle: hip-knee-ankle (มุมเข่า)")
print("   - Shoulder Angle: elbow-shoulder-hip (มุมไหล่)")
print("   - Hip Angle: shoulder-hip-knee (มุมสะโพก)")
print("   - Torso Lean: angle from vertical (มุมเอียงลำตัว)")

# %% [markdown]
# ### 4.2 Calculate Angles for All Data
#
# คำนวณมุมข้อต่อสำหรับทุก row ใน DataFrame

# %%
# =====================================================
# STEP 4.2: Calculate Angles for All Data
# =====================================================
# Description: คำนวณมุมข้อต่อทั้งหมดและเพิ่มเป็น columns ใหม่
# =====================================================

print("=" * 70)
print("📐 CALCULATING JOINT ANGLES")
print("=" * 70)

# คำนวณมุมสำหรับแต่ละ row
print("\n🔄 Processing angles for all rows...")

# ใช้ apply เพื่อคำนวณ
angle_results = df_labeled.apply(calculate_all_angles, axis=1)
angle_df = pd.DataFrame(angle_results.tolist())

# รวมกับ DataFrame หลัก
df_with_angles = pd.concat([df_labeled.reset_index(drop=True), angle_df], axis=1)

print(f"✅ Angles calculated for {len(df_with_angles):,} rows")

# แสดงสถิติเบื้องต้น
angle_columns = ['left_elbow_angle', 'right_elbow_angle', 'left_knee_angle', 
                 'right_knee_angle', 'left_shoulder_angle', 'right_shoulder_angle',
                 'left_hip_angle', 'right_hip_angle', 'torso_lean_angle']

print("\n📊 Angle Statistics Summary:")
print(df_with_angles[angle_columns].describe().round(2).to_string())

# %% [markdown]
# ### 4.3 Angle Time Series Visualization
#
# แสดงการเปลี่ยนแปลงของมุมข้อต่อตามเวลา

# %%
# =====================================================
# STEP 4.3: Angle Time Series Visualization
# =====================================================
# Description: สร้างกราฟแสดงการเปลี่ยนแปลงของมุมข้อต่อ
# ตามเวลาสำหรับแต่ละ action
# =====================================================

def plot_angle_timeseries(df, person_id, angles, figsize=(16, 12)):
    """
    สร้างกราฟ Time Series ของมุมข้อต่อ
    """
    person_data = df[df['person_id'] == person_id].copy()
    person_data = person_data.sort_values('timestamp')
    
    # สร้าง figure
    fig, axes = plt.subplots(len(angles), 1, figsize=figsize, sharex=True)
    if len(angles) == 1:
        axes = [axes]
    
    fig.suptitle(f'📐 Joint Angle Time Series - Person ID: {person_id}', 
                 fontsize=14, fontweight='bold')
    
    # Color map สำหรับ actions
    actions = [a for a in person_data['action'].unique() if a != 'Unknown']
    action_colors = dict(zip(actions, plt.cm.Set2(np.linspace(0, 1, len(actions)))))
    
    for idx, angle_name in enumerate(angles):
        ax = axes[idx]
        
        for action in actions:
            action_data = person_data[person_data['action'] == action]
            if len(action_data) > 0:
                ax.scatter(action_data['timestamp'], action_data[angle_name],
                          label=action, alpha=0.5, s=3, color=action_colors[action])
        
        ax.set_ylabel(f'{angle_name.replace("_", " ").title()}\n(degrees)')
        ax.set_ylim(0, 180)
        ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90°')
        ax.axhline(y=180, color='gray', linestyle=':', alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=3, markerscale=3)
    
    axes[-1].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    return fig

# Plot angle time series
ANALYSIS_ANGLES = ['right_elbow_angle', 'left_elbow_angle', 
                   'right_knee_angle', 'left_knee_angle',
                   'torso_lean_angle']

print("=" * 70)
print("📐 ANGLE TIME SERIES VISUALIZATION")
print("=" * 70)

for pid in SELECTED_PERSON_IDS[:1]:  # แสดงเฉพาะ person แรก
    fig = plot_angle_timeseries(df_with_angles, pid, ANALYSIS_ANGLES)
    filename = f'07_angle_timeseries_person_{pid}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Chart saved: {filename}")

# %% [markdown]
# ### 4.4 Angle Statistics per Action
#
# เปรียบเทียบค่าเฉลี่ยและการกระจายของมุมในแต่ละ Action

# %%
# =====================================================
# STEP 4.4: Angle Statistics per Action
# =====================================================
# Description: คำนวณและเปรียบเทียบสถิติมุมของแต่ละ action
# เพื่อระบุลักษณะเฉพาะของแต่ละท่า
# =====================================================

# คำนวณสถิติมุมต่อ action
angle_stats = df_with_angles.groupby('action')[angle_columns].agg(['mean', 'std', 'min', 'max'])
angle_stats = angle_stats.round(2)

print("=" * 70)
print("📊 ANGLE STATISTICS BY ACTION")
print("=" * 70)

# แสดงค่าเฉลี่ยมุม
print("\n📋 Mean Angle by Action:")
mean_angles = df_with_angles.groupby('action')[angle_columns].mean().round(2)
print(mean_angles.to_string())

# Visualization - Box plots for key angles
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

key_angles = ['right_elbow_angle', 'left_elbow_angle', 
              'right_knee_angle', 'left_knee_angle',
              'right_shoulder_angle', 'torso_lean_angle']

actions_list = [a for a in df_with_angles['action'].unique() if a != 'Unknown']

for idx, angle in enumerate(key_angles):
    ax = axes[idx]
    
    # สร้าง box plot
    data_for_plot = [df_with_angles[df_with_angles['action'] == action][angle].dropna() 
                     for action in actions_list]
    
    bp = ax.boxplot(data_for_plot, labels=[a.replace('_', '\n')[:15] for a in actions_list],
                    patch_artist=True)
    
    # สีสำหรับ box
    colors = plt.cm.Set3(np.linspace(0, 1, len(actions_list)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Angle (degrees)')
    ax.set_title(f'📐 {angle.replace("_", " ").title()}')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylim(0, 180)

plt.suptitle('📊 Joint Angle Distribution by Action', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('08_angle_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Chart saved: 08_angle_boxplots.png")

# %% [markdown]
# ### 4.5 Angle Heatmap Comparison
#
# สร้าง Heatmap เปรียบเทียบค่าเฉลี่ยมุมของแต่ละ Action

# %%
# =====================================================
# STEP 4.5: Angle Heatmap Comparison
# =====================================================
# Description: สร้าง Heatmap แสดงค่าเฉลี่ยมุมของแต่ละ action
# ทำให้เห็นภาพรวมได้ง่าย
# =====================================================

# คำนวณค่าเฉลี่ยมุมต่อ action
mean_angles_matrix = df_with_angles[df_with_angles['action'] != 'Unknown'].groupby('action')[angle_columns].mean()

# สร้าง Heatmap
fig, ax = plt.subplots(figsize=(14, 8))

# แปลงชื่อ columns ให้อ่านง่าย
display_columns = [col.replace('_', ' ').replace(' angle', '').title() 
                   for col in angle_columns]

sns.heatmap(mean_angles_matrix.values, 
            annot=True, 
            fmt='.1f',
            cmap='RdYlBu_r',
            xticklabels=display_columns,
            yticklabels=mean_angles_matrix.index,
            vmin=0, 
            vmax=180,
            cbar_kws={'label': 'Angle (degrees)'},
            ax=ax)

ax.set_title('📊 Mean Joint Angles Heatmap by Action\n'
             '(Darker red = larger angle, Darker blue = smaller angle)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Joint Angle')
ax.set_ylabel('Action')

plt.tight_layout()
plt.savefig('09_angle_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Chart saved: 09_angle_heatmap.png")

# %% [markdown]
# ---
#
# ## 📚 Part 5: Compare Actions - Combined Analysis
#
# ### 5.1 Multi-Person Action Comparison
#
# เปรียบเทียบการเคลื่อนไหวของหลายคนในแต่ละ Action

# %%
# =====================================================
# STEP 5.1: Multi-Person Action Comparison
# =====================================================
# Description: เปรียบเทียบ pattern การเคลื่อนไหว
# ของ 3 persons ที่เลือกในแต่ละ action
# =====================================================

def plot_multi_person_comparison(df, person_ids, action, figsize=(16, 10)):
    """
    เปรียบเทียบ keypoints และ angles ระหว่างหลาย persons
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Color map สำหรับ persons
    person_colors = dict(zip(person_ids, ['#e74c3c', '#3498db', '#2ecc71']))
    
    action_data = df[df['action'] == action]
    
    # 1. Right Wrist X-Y trajectory
    ax = axes[0, 0]
    for pid in person_ids:
        person_data = action_data[action_data['person_id'] == pid]
        if len(person_data) > 0:
            ax.plot(person_data['right_wrist_x'], person_data['right_wrist_y'],
                   label=f'Person {pid}', alpha=0.7, linewidth=1,
                   color=person_colors[pid])
            # Mark start and end
            ax.scatter(person_data['right_wrist_x'].iloc[0], 
                      person_data['right_wrist_y'].iloc[0],
                      marker='o', s=100, color=person_colors[pid], zorder=5)
            ax.scatter(person_data['right_wrist_x'].iloc[-1], 
                      person_data['right_wrist_y'].iloc[-1],
                      marker='x', s=100, color=person_colors[pid], zorder=5)
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('🤜 Right Wrist Trajectory (○=start, ×=end)')
    ax.invert_yaxis()
    ax.legend()
    
    # 2. Right Ankle X-Y trajectory
    ax = axes[0, 1]
    for pid in person_ids:
        person_data = action_data[action_data['person_id'] == pid]
        if len(person_data) > 0:
            ax.plot(person_data['right_ankle_x'], person_data['right_ankle_y'],
                   label=f'Person {pid}', alpha=0.7, linewidth=1,
                   color=person_colors[pid])
            ax.scatter(person_data['right_ankle_x'].iloc[0], 
                      person_data['right_ankle_y'].iloc[0],
                      marker='o', s=100, color=person_colors[pid], zorder=5)
            ax.scatter(person_data['right_ankle_x'].iloc[-1], 
                      person_data['right_ankle_y'].iloc[-1],
                      marker='x', s=100, color=person_colors[pid], zorder=5)
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title('🦶 Right Ankle Trajectory (○=start, ×=end)')
    ax.invert_yaxis()
    ax.legend()
    
    # 3. Elbow Angle over time
    ax = axes[1, 0]
    for pid in person_ids:
        person_data = action_data[action_data['person_id'] == pid].sort_values('timestamp')
        if len(person_data) > 0:
            time_norm = person_data['timestamp'] - person_data['timestamp'].min()
            ax.plot(time_norm, person_data['right_elbow_angle'],
                   label=f'Person {pid}', alpha=0.7, linewidth=1.5,
                   color=person_colors[pid])
    
    ax.set_xlabel('Normalized Time (seconds)')
    ax.set_ylabel('Right Elbow Angle (degrees)')
    ax.set_title('💪 Right Elbow Angle Over Time')
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 180)
    ax.legend()
    
    # 4. Knee Angle over time
    ax = axes[1, 1]
    for pid in person_ids:
        person_data = action_data[action_data['person_id'] == pid].sort_values('timestamp')
        if len(person_data) > 0:
            time_norm = person_data['timestamp'] - person_data['timestamp'].min()
            ax.plot(time_norm, person_data['right_knee_angle'],
                   label=f'Person {pid}', alpha=0.7, linewidth=1.5,
                   color=person_colors[pid])
    
    ax.set_xlabel('Normalized Time (seconds)')
    ax.set_ylabel('Right Knee Angle (degrees)')
    ax.set_title('🦵 Right Knee Angle Over Time')
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 180)
    ax.legend()
    
    plt.suptitle(f'🥊 Action: {action.replace("_", " ")}\n'
                 f'Comparing {len(person_ids)} Persons', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

# สร้างกราฟเปรียบเทียบสำหรับแต่ละ action
print("=" * 70)
print("🥊 MULTI-PERSON ACTION COMPARISON")
print("=" * 70)

actions_to_compare = [a for a in df_with_angles['action'].unique() if a != 'Unknown']

for action in actions_to_compare[:3]:  # แสดง 3 actions แรก
    fig = plot_multi_person_comparison(df_with_angles, SELECTED_PERSON_IDS, action)
    filename = f'10_comparison_{action[:20]}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Chart saved: {filename}")

# %% [markdown]
# ### 5.2 Action Feature Summary
#
# สรุป Feature หลักของแต่ละ Action เพื่อทำความเข้าใจลักษณะเฉพาะ

# %%
# =====================================================
# STEP 5.2: Action Feature Summary
# =====================================================
# Description: สรุปลักษณะเฉพาะของแต่ละ action
# โดยใช้ทั้ง position, velocity, และ angle features
# =====================================================

def extract_action_features(df, person_ids):
    """
    สกัด features สำหรับแต่ละ action
    """
    features_list = []
    
    for action in df['action'].unique():
        if action == 'Unknown':
            continue
            
        action_data = df[df['action'] == action]
        
        # คำนวณ features
        features = {'action': action}
        
        # 1. Duration (ระยะเวลา)
        features['duration_mean'] = action_data.groupby('person_id').apply(
            lambda x: x['timestamp'].max() - x['timestamp'].min()
        ).mean()
        
        # 2. Frame count
        features['frame_count'] = len(action_data)
        
        # 3. Persons count
        features['persons_count'] = action_data['person_id'].nunique()
        
        # 4. Mean angles
        for angle in angle_columns:
            features[f'{angle}_mean'] = action_data[angle].mean()
            features[f'{angle}_std'] = action_data[angle].std()
        
        # 5. Position ranges (movement extent)
        for kpt in ['right_wrist', 'right_ankle']:
            features[f'{kpt}_x_range'] = action_data[f'{kpt}_x'].max() - action_data[f'{kpt}_x'].min()
            features[f'{kpt}_y_range'] = action_data[f'{kpt}_y'].max() - action_data[f'{kpt}_y'].min()
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

# สกัด features
action_features = extract_action_features(df_with_angles, SELECTED_PERSON_IDS)

print("=" * 70)
print("📋 ACTION FEATURE SUMMARY")
print("=" * 70)

# แสดง summary table
display_cols = ['action', 'duration_mean', 'frame_count', 'persons_count',
                'right_elbow_angle_mean', 'right_knee_angle_mean',
                'right_wrist_x_range', 'right_wrist_y_range']

print("\n📊 Key Features by Action:")
print(action_features[display_cols].round(2).to_string(index=False))

# Visualization - Radar Chart สำหรับ comparing actions
def create_radar_chart(df_features, actions, metrics, figsize=(10, 8)):
    """
    สร้าง Radar Chart เปรียบเทียบ actions
    """
    # Normalize metrics
    df_norm = df_features.copy()
    for metric in metrics:
        max_val = df_norm[metric].max()
        if max_val > 0:
            df_norm[metric] = df_norm[metric] / max_val
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(actions)))
    
    for idx, action in enumerate(actions):
        action_row = df_norm[df_norm['action'] == action]
        if len(action_row) == 0:
            continue
            
        values = action_row[metrics].values.flatten().tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=action.replace('_', ' '),
                color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    # Fix labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '\n').replace(' mean', '').replace(' range', '\nrange')[:20] 
                        for m in metrics], fontsize=9)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('📊 Action Feature Comparison (Normalized)', fontsize=14, fontweight='bold')
    
    return fig

# สร้าง Radar Chart
radar_metrics = ['right_elbow_angle_mean', 'right_knee_angle_mean', 
                 'torso_lean_angle_mean', 'right_wrist_x_range', 
                 'right_wrist_y_range', 'right_ankle_x_range']

fig = create_radar_chart(action_features, actions_to_compare[:5], radar_metrics)
plt.tight_layout()
plt.savefig('11_action_radar_chart.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Chart saved: 11_action_radar_chart.png")

# %% [markdown]
# ---
#
# ## 📚 Part 6: Advanced Visualization
#
# ### 6.1 Phase Portrait (วิเคราะห์ Phase Space)
#
# Phase Portrait แสดงความสัมพันธ์ระหว่างตำแหน่งและความเร็ว
# ช่วยให้เห็น pattern การเคลื่อนไหวแบบ cyclic

# %%
# =====================================================
# STEP 6.1: Phase Portrait Analysis
# =====================================================
# Description: Phase Portrait แสดง position vs velocity
# ใช้วิเคราะห์ periodic movements และ movement dynamics
# 
# แกน X: Position
# แกน Y: Velocity (derivative of position)
# =====================================================

def create_phase_portrait(df, person_id, keypoint, actions, figsize=(14, 10)):
    """
    สร้าง Phase Portrait สำหรับ keypoint
    """
    person_data = df[df['person_id'] == person_id].copy()
    person_data = person_data.sort_values('timestamp')
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    action_colors = dict(zip(actions, plt.cm.Set2(np.linspace(0, 1, len(actions)))))
    
    for ax_idx, (pos_col, title) in enumerate([
        (f'{keypoint}_x', f'{keypoint} X'),
        (f'{keypoint}_y', f'{keypoint} Y')
    ]):
        # คำนวณ velocity
        person_data[f'{pos_col}_vel'] = person_data[pos_col].diff() / person_data['timestamp'].diff()
        
        # Plot position vs time
        ax_pos = axes[ax_idx, 0]
        for action in actions:
            action_data = person_data[person_data['action'] == action]
            if len(action_data) > 0:
                ax_pos.scatter(action_data['timestamp'], action_data[pos_col],
                              alpha=0.5, s=3, color=action_colors[action], label=action)
        
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Position (pixels)')
        ax_pos.set_title(f'{title} - Position over Time')
        if ax_idx == 0:
            ax_pos.legend(loc='upper right', fontsize=8, markerscale=5)
        
        # Plot phase portrait
        ax_phase = axes[ax_idx, 1]
        for action in actions:
            action_data = person_data[person_data['action'] == action]
            if len(action_data) > 1:
                # Smooth velocity
                vel_smooth = uniform_filter1d(action_data[f'{pos_col}_vel'].fillna(0), size=5)
                ax_phase.scatter(action_data[pos_col].values[:-1], vel_smooth[:-1],
                               alpha=0.5, s=3, color=action_colors[action], label=action)
        
        ax_phase.set_xlabel('Position (pixels)')
        ax_phase.set_ylabel('Velocity (pixels/s)')
        ax_phase.set_title(f'{title} - Phase Portrait')
        ax_phase.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'📊 Phase Portrait Analysis - {keypoint.replace("_", " ").title()}\n'
                 f'Person ID: {person_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

# สร้าง Phase Portrait
print("=" * 70)
print("📊 PHASE PORTRAIT ANALYSIS")
print("=" * 70)

for kpt in ['right_wrist', 'right_ankle']:
    fig = create_phase_portrait(df_with_angles, SELECTED_PERSON_IDS[0], 
                                kpt, actions_to_compare[:5])
    filename = f'12_phase_portrait_{kpt}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Chart saved: {filename}")

# %% [markdown]
# ### 6.2 Correlation Analysis
#
# วิเคราะห์ความสัมพันธ์ระหว่าง angles ต่างๆ

# %%
# =====================================================
# STEP 6.2: Correlation Analysis
# =====================================================
# Description: วิเคราะห์ correlation ระหว่าง joint angles
# เพื่อเข้าใจการประสานงานของข้อต่อต่างๆ
# =====================================================

# คำนวณ Correlation Matrix
correlation_matrix = df_with_angles[angle_columns].corr()

# สร้าง Heatmap
fig, ax = plt.subplots(figsize=(12, 10))

# แปลงชื่อ columns ให้อ่านง่าย
display_labels = [col.replace('_angle', '').replace('_', ' ').title() 
                  for col in angle_columns]

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1, 
            vmax=1,
            xticklabels=display_labels,
            yticklabels=display_labels,
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax)

ax.set_title('📊 Joint Angle Correlation Matrix\n'
             '(Red=positive, Blue=negative correlation)', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('13_angle_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Chart saved: 13_angle_correlation.png")

# แสดง top correlations
print("\n📊 Top Positive Correlations:")
corr_pairs = []
for i in range(len(angle_columns)):
    for j in range(i+1, len(angle_columns)):
        corr_pairs.append({
            'Angle 1': angle_columns[i],
            'Angle 2': angle_columns[j],
            'Correlation': correlation_matrix.iloc[i, j]
        })

corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
print(corr_df.head(5).to_string(index=False))

print("\n📊 Top Negative Correlations:")
print(corr_df.tail(5).to_string(index=False))

# %% [markdown]
# ---
#
# ## 📚 Part 7: Summary and Conclusions
#
# ### 7.1 Generate Final Report
#
# สรุปผลการวิเคราะห์ทั้งหมด

# %%
# =====================================================
# STEP 7.1: Generate Final Summary Report
# =====================================================
# Description: สรุปผลการวิเคราะห์ทั้งหมดเป็น report
# =====================================================

print("=" * 70)
print("📋 FINAL ANALYSIS REPORT")
print("=" * 70)

print(f"""
🎯 DATASET OVERVIEW
─────────────────────────────────────────
• Total Frames: {len(df_pose):,}
• Total Labeled Frames: {len(df_labeled):,}
• Unique Persons: {df_pose['person_id'].nunique()}
• Actions Analyzed: {len(actions_to_compare)}
• Selected Persons for Analysis: {SELECTED_PERSON_IDS}

📊 ACTIONS SUMMARY
─────────────────────────────────────────""")

for action in actions_to_compare:
    action_data = df_with_angles[df_with_angles['action'] == action]
    print(f"\n🥊 {action.replace('_', ' ')}")
    print(f"   Frames: {len(action_data):,}")
    print(f"   Mean Right Elbow Angle: {action_data['right_elbow_angle'].mean():.1f}°")
    print(f"   Mean Right Knee Angle: {action_data['right_knee_angle'].mean():.1f}°")
    print(f"   Mean Torso Lean: {action_data['torso_lean_angle'].mean():.1f}°")

print(f"""
📈 KEY FINDINGS
─────────────────────────────────────────
1. Actions ที่ต้องการการเหยียดแขน (elbow angle สูง) เด่นชัด
2. ท่าที่ต้องงอเข่า (knee angle ต่ำ) สามารถแยกแยะได้ดี
3. มุมลำตัว (torso lean) ช่วยบ่งบอกท่าทางโจมตี/ป้องกัน
4. Velocity analysis ช่วยระบุช่วงเวลาของการเคลื่อนไหวเร็ว

📁 FILES GENERATED
─────────────────────────────────────────
""")

import os
for f in sorted(os.listdir('.')):
    if f.endswith('.png'):
        print(f"   ✅ {f}")

print("""
─────────────────────────────────────────
📌 RECOMMENDATIONS FOR FURTHER ANALYSIS
─────────────────────────────────────────
1. ใช้ Machine Learning เพื่อ classify actions อัตโนมัติ
2. วิเคราะห์ temporal patterns ด้วย DTW หรือ LSTM
3. สร้าง feature vectors จาก angles สำหรับ clustering
4. เพิ่ม 3D reconstruction ถ้ามีข้อมูลจากหลายกล้อง
""")

# %% [markdown]
# ### 7.2 Export Processed Data
#
# Export ข้อมูลที่ประมวลผลแล้วสำหรับใช้งานต่อ

# %%
# =====================================================
# STEP 7.2: Export Processed Data
# =====================================================
# Description: บันทึกข้อมูลที่ประมวลผลแล้ว
# รวมถึง angles ที่คำนวณ
# =====================================================

# Export DataFrame พร้อม angles
output_filename = 'pose_data_with_angles.csv'
df_with_angles.to_csv(output_filename, index=False)
print(f"✅ Exported: {output_filename}")

# Export action features
features_filename = 'action_features_summary.csv'
action_features.to_csv(features_filename, index=False)
print(f"✅ Exported: {features_filename}")

# Export velocity statistics
velocity_filename = 'velocity_statistics.csv'
velocity_stats.to_csv(velocity_filename, index=True)
print(f"✅ Exported: {velocity_filename}")

print("\n" + "=" * 70)
print("🎉 LAB COMPLETED SUCCESSFULLY!")
print("=" * 70)

# %% [markdown]
# ---
#
# ## 📖 Additional Resources
#
# **สำหรับศึกษาเพิ่มเติม:**
#
# 1. **Pose Estimation:**
#    - COCO Keypoint Detection: https://cocodataset.org/#keypoints-2020
#    - YOLOv11 Documentation: https://docs.ultralytics.com/
#
# 2. **Time Series Analysis:**
#    - Pandas Time Series: https://pandas.pydata.org/docs/user_guide/timeseries.html
#    - Signal Processing: https://docs.scipy.org/doc/scipy/reference/signal.html
#
# 3. **Angle Calculation:**
#    - Biomechanics Tutorials: https://biomechanics.stanford.edu/
#    - Joint Angle Analysis: OpenSim Documentation
#
# 4. **Visualization:**
#    - Matplotlib Gallery: https://matplotlib.org/stable/gallery/
#    - Seaborn Tutorial: https://seaborn.pydata.org/tutorial.html
#
# ---
#
# **End of Lab** 🎓