# %% [markdown]
# # Lab: Image Augmentation using PyTorch
#
# ## วัตถุประสงค์การเรียนรู้ (Learning Objectives)
# 1. เข้าใจหลักการและความสำคัญของ Image Augmentation
# 2. สามารถใช้ `torchvision.transforms` สำหรับการ augment รูปภาพ
# 3. เข้าใจ transforms แต่ละประเภทและการประยุกต์ใช้
# 4. สามารถสร้าง Custom Dataset พร้อม Augmentation Pipeline
# 5. ประยุกต์ใช้กับ Dataset จริงจาก Kaggle (Animal Faces)

# %% [markdown]
# ## Part 1: Introduction to Image Augmentation
#
# **Image Augmentation** คือเทคนิคการสร้างข้อมูลรูปภาพเพิ่มเติมจากรูปภาพต้นฉบับ
# โดยการแปลงรูปภาพด้วยวิธีการต่างๆ เช่น:
# - Geometric transformations (flip, rotate, crop, scale)
# - Color transformations (brightness, contrast, saturation)
# - Noise injection
# - และอื่นๆ
#
# **ประโยชน์:**
# - เพิ่มขนาด Dataset โดยไม่ต้องเก็บข้อมูลเพิ่ม
# - ลด Overfitting
# - ทำให้ Model มีความ robust มากขึ้น
# - จำลองสถานการณ์ที่หลากหลายในโลกจริง

# %% [markdown]
# ใช้งานกับ dataset จริงจาก Kaggle
#
# **Dataset:** https://www.kaggle.com/datasets/andrewmvd/animal-faces

# %%

import IPython
import sys

def clean_notebook():
    IPython.display.clear_output(wait=True)
    print("Notebook cleaned.")

# !uv pip install kagglehub

# Clean up the notebook
clean_notebook()

# %%
# Import Libraries
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from pathlib import Path

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

# %%
# Configuration
IMG_PATH = './Taj_Mahal.jpg'
DATASET_PATH = None  # Path สำหรับ Animal Faces dataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# %% [markdown]
# ## Part 2: Load and Display Original Image
#
# เริ่มต้นด้วยการโหลดรูปภาพ Taj Mahal และแสดงผล

# %%
def show_image(image, title="Image"):
    """Display a single image (PIL or Tensor)"""
    if isinstance(image, torch.Tensor):
        # Convert tensor to numpy
        if image.dim() == 4:
            image = image[0]
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_images_grid(images, titles=None, nrow=4, figsize=(16, 12)):
    """Display multiple images in a grid"""
    n = len(images)
    ncol = min(nrow, n)
    nrow_actual = (n + ncol - 1) // ncol
    
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=figsize)
    axes = np.array(axes).flatten() if n > 1 else [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        if isinstance(img, torch.Tensor):
            if img.dim() == 4:
                img = img[0]
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
    
    # Hide empty subplots
    for j in range(len(images), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# %%
# Load the original image
original_image = Image.open(IMG_PATH)
print(f"Image size: {original_image.size}")
print(f"Image mode: {original_image.mode}")

show_image(original_image, "Original Taj Mahal Image")

# %% [markdown]
# ## Part 3: Basic Transforms
#
# ### 3.1 Geometric Transforms
#
# การแปลงทางเรขาคณิตช่วยให้ model เรียนรู้ที่จะรู้จักวัตถุในตำแหน่งและมุมต่างๆ

# %%
# Convert to tensor first (normalized to [0, 1])
to_tensor = T.ToTensor()
img_tensor = to_tensor(original_image)
print(f"Tensor shape: {img_tensor.shape}")

# %%
img_tensor

# %%
# 3.1.1 Horizontal Flip
horizontal_flip = T.RandomHorizontalFlip(p=1.0)  # p=1.0 means always flip
flipped_h = horizontal_flip(img_tensor)

# 3.1.2 Vertical Flip
vertical_flip = T.RandomVerticalFlip(p=1.0)
flipped_v = vertical_flip(img_tensor)

# 3.1.3 Random Rotation
rotation = T.RandomRotation(degrees=45)
rotated_images = [rotation(img_tensor) for _ in range(4)]

# Display results
show_images_grid(
    [img_tensor, flipped_h, flipped_v],
    titles=['Original', 'Horizontal Flip', 'Vertical Flip'],
    nrow=3,
    figsize=(12, 4)
)

# %%
# Show rotation variations
show_images_grid(
    rotated_images,
    titles=[f'Rotation {i+1}' for i in range(4)],
    nrow=4,
    figsize=(14, 4)
)

# %% [markdown]
# ### 3.2 Cropping Transforms
#
# การ crop ช่วยให้ model โฟกัสที่ส่วนต่างๆ ของรูปภาพ

# %%
# 3.2.1 Center Crop
center_crop = T.CenterCrop(size=400)
cropped_center = center_crop(img_tensor)

# 3.2.2 Random Crop
random_crop = T.RandomCrop(size=300)
cropped_random = [random_crop(img_tensor) for _ in range(4)]

# 3.2.3 Random Resized Crop (commonly used in training)
resized_crop = T.RandomResizedCrop(size=224, scale=(0.5, 1.0), ratio=(0.8, 1.2))
cropped_resized = [resized_crop(img_tensor) for _ in range(4)]

# %%
# Display cropping results
show_images_grid(
    [img_tensor, cropped_center] + cropped_random,
    titles=['Original', 'Center Crop'] + [f'Random Crop {i+1}' for i in range(4)],
    nrow=3,
    figsize=(12, 8)
)

# %%
# Display resized crops
show_images_grid(
    cropped_resized,
    titles=[f'RandomResizedCrop {i+1}' for i in range(4)],
    nrow=4,
    figsize=(14, 4)
)

# %% [markdown]
# ### 3.3 Color Transforms
#
# การปรับสี/แสงช่วยให้ model ทนทานต่อสภาพแสงที่แตกต่างกัน

# %%
# 3.3.1 ColorJitter - Adjust brightness, contrast, saturation, hue
color_jitter = T.ColorJitter(
    brightness=0.5,   # how much to jitter brightness
    contrast=0.5,     # how much to jitter contrast
    saturation=0.5,   # how much to jitter saturation
    hue=0.2           # how much to jitter hue (between -0.5 and 0.5)
)
jittered_images = [color_jitter(img_tensor) for _ in range(6)]

# %%
show_images_grid(
    [img_tensor] + jittered_images,
    titles=['Original'] + [f'ColorJitter {i+1}' for i in range(6)],
    nrow=4,
    figsize=(14, 8)
)

# %%
# 3.3.2 Grayscale
grayscale = T.Grayscale(num_output_channels=3)
gray_image = grayscale(img_tensor)

# 3.3.3 Random Grayscale (with probability)
random_gray = T.RandomGrayscale(p=0.5)

# 3.3.4 Gaussian Blur
gaussian_blur = T.GaussianBlur(kernel_size=15, sigma=(2.0, 5.0))
blurred = gaussian_blur(img_tensor)

# 3.3.5 Adjust Sharpness
sharpness = T.RandomAdjustSharpness(sharpness_factor=3, p=1.0)
sharpened = sharpness(img_tensor)

# %%
show_images_grid(
    [img_tensor, gray_image, blurred, sharpened],
    titles=['Original', 'Grayscale', 'Gaussian Blur', 'Sharpened'],
    nrow=4,
    figsize=(14, 4)
)

# %% [markdown]
# ### 3.4 Advanced Transforms
#
# เทคนิค augmentation ขั้นสูงที่ใช้กันใน modern deep learning

# %%
# 3.4.1 Random Erasing (Cutout-like)
random_erasing = T.RandomErasing(
    p=1.0,
    scale=(0.02, 0.33),
    ratio=(0.3, 3.3),
    value=0  # Fill with black (can also use 'random')
)
erased_images = [random_erasing(img_tensor.clone()) for _ in range(4)]

# %%
show_images_grid(
    erased_images,
    titles=[f'Random Erasing {i+1}' for i in range(4)],
    nrow=4,
    figsize=(14, 4)
)

# %%
# 3.4.2 Random Perspective Transform
perspective = T.RandomPerspective(distortion_scale=0.5, p=1.0)
perspective_images = [perspective(img_tensor) for _ in range(4)]

show_images_grid(
    perspective_images,
    titles=[f'Perspective {i+1}' for i in range(4)],
    nrow=4,
    figsize=(14, 4)
)

# %%
# 3.4.3 Random Affine Transform
affine = T.RandomAffine(
    degrees=30,
    translate=(0.2, 0.2),
    scale=(0.8, 1.2),
    shear=15
)
affine_images = [affine(img_tensor) for _ in range(4)]

show_images_grid(
    affine_images,
    titles=[f'Affine {i+1}' for i in range(4)],
    nrow=4,
    figsize=(14, 4)
)

# %% [markdown]
# ## Part 4: Composing Transforms
#
# ในการใช้งานจริง เรามักจะรวม transforms หลายๆ ตัวเข้าด้วยกัน

# %%
# Training Transform Pipeline (commonly used)
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test Transform Pipeline (minimal augmentation)
val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
# Apply training transforms multiple times to see variations
augmented_train = []
for i in range(8):
    aug_img = train_transform(original_image)
    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    aug_img_denorm = aug_img * std + mean
    augmented_train.append(aug_img_denorm)

show_images_grid(
    augmented_train,
    titles=[f'Train Aug {i+1}' for i in range(8)],
    nrow=4,
    figsize=(14, 8)
)

# %% [markdown]
# ## Part 5: RandAugment และ AutoAugment
#
# เทคนิค Augmentation อัตโนมัติที่ได้จากการทำ research

# %%
# RandAugment - Simple yet effective augmentation strategy
randaugment = T.RandAugment(num_ops=2, magnitude=9)

# AutoAugment - Policy learned from data (ImageNet policy)
autoaugment = T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET)

# TrivialAugment - Even simpler than RandAugment
trivialaugment = T.TrivialAugmentWide()

# %%
# Apply different auto augment strategies
pil_image = original_image.copy()

rand_aug_images = [T.ToTensor()(randaugment(pil_image)) for _ in range(4)]
auto_aug_images = [T.ToTensor()(autoaugment(pil_image.copy())) for _ in range(4)]
trivial_aug_images = [T.ToTensor()(trivialaugment(pil_image.copy())) for _ in range(4)]

# %%
print("RandAugment Results:")
show_images_grid(rand_aug_images, titles=[f'RandAug {i+1}' for i in range(4)], nrow=4, figsize=(14, 4))

# %%
print("AutoAugment Results:")
show_images_grid(auto_aug_images, titles=[f'AutoAug {i+1}' for i in range(4)], nrow=4, figsize=(14, 4))

# %%
print("TrivialAugment Results:")
show_images_grid(trivial_aug_images, titles=[f'TrivialAug {i+1}' for i in range(4)], nrow=4, figsize=(14, 4))

# %% [markdown]
# ## Part 9: Comparison of Augmentation Strategies
#
# เปรียบเทียบ strategies ต่างๆ

# %%
# Define different augmentation strategies
strategies = {
    'No Augmentation': T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ]),
    'Basic (Flip + Crop)': T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ]),
    'Moderate': T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.ToTensor()
    ]),
    'Aggressive': T.Compose([
        T.RandomResizedCrop(224, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(0.4, 0.4, 0.4, 0.2),
        T.GaussianBlur(5),
        T.ToTensor()
    ]),
    'RandAugment': T.Compose([
        T.Resize((224, 224)),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor()
    ]),
    'AutoAugment': T.Compose([
        T.Resize((224, 224)),
        T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
        T.ToTensor()
    ])
}

# %%
# Apply all strategies to the image
comparison_images = []
comparison_titles = []

for name, transform in strategies.items():
    img = transform(original_image)
    comparison_images.append(img)
    comparison_titles.append(name)

show_images_grid(comparison_images, titles=comparison_titles, nrow=3, figsize=(14, 10))

# %% [markdown]
# ## Part 10: Custom Transform Function
#
# สร้าง custom transform ของตัวเอง

# %%
class AddGaussianNoise:
    """Add Gaussian noise to image"""
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

class MixChannels:
    """Randomly mix color channels"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, tensor):
        if random.random() < self.p:
            # Random permutation of channels
            perm = torch.randperm(3)
            return tensor[perm, :, :]
        return tensor
    
    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'

class RandomCutout:
    """Apply multiple random cutouts"""
    def __init__(self, n_holes=4, size=32):
        self.n_holes = n_holes
        self.size = size
    
    def __call__(self, tensor):
        h, w = tensor.shape[1], tensor.shape[2]
        mask = torch.ones_like(tensor)
        
        for _ in range(self.n_holes):
            y = random.randint(0, h - self.size)
            x = random.randint(0, w - self.size)
            mask[:, y:y+self.size, x:x+self.size] = 0
        
        return tensor * mask
    
    def __repr__(self):
        return f'{self.__class__.__name__}(n_holes={self.n_holes}, size={self.size})'

# %%
# Test custom transforms
custom_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

img_for_custom = custom_transform(original_image)

# Apply custom transforms
noisy = AddGaussianNoise(std=0.1)(img_for_custom.clone())
mixed = MixChannels(p=1.0)(img_for_custom.clone())
cutout = RandomCutout(n_holes=6, size=40)(img_for_custom.clone())

show_images_grid(
    [img_for_custom, noisy, mixed, cutout],
    titles=['Original', 'Gaussian Noise', 'Mixed Channels', 'Multiple Cutouts'],
    nrow=4,
    figsize=(14, 4)
)

# %% [markdown]
# ## Part 12: Summary และ Best Practices
#
# ### Summary of Transforms
#
# | Transform | Use Case | Parameters |
# |-----------|----------|------------|
# | RandomHorizontalFlip | Always (most images) | p=0.5 |
# | RandomResizedCrop | Training | size, scale, ratio |
# | ColorJitter | Handle lighting variations | brightness, contrast, saturation, hue |
# | RandomRotation | Objects at different angles | degrees |
# | GaussianBlur | Reduce noise sensitivity | kernel_size, sigma |
# | RandAugment | General purpose | num_ops, magnitude |
# | RandomErasing | Occlusion robustness | p, scale, ratio |
#
# ### Best Practices
#
# 1. **Start Simple**: Begin with basic augmentations (flip, crop) then add more
# 2. **Match Domain**: Choose augmentations relevant to your use case
# 3. **Don't Overdo**: Too much augmentation can hurt performance
# 4. **Validate Separately**: Never augment validation/test data (except resize/crop)
# 5. **Use AutoAugment/RandAugment**: Good defaults for many tasks
# 6. **Monitor Training**: Watch for signs of under/over-augmentation

# %%
# Final comprehensive pipeline example
final_train_transform = T.Compose([
    # Spatial transforms
    T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.RandomPerspective(distortion_scale=0.2, p=0.3),
    
    # Color transforms
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomGrayscale(p=0.05),
    
    # Blur
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    
    # Convert to tensor and normalize
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
    # Regularization
    T.RandomErasing(p=0.1, scale=(0.02, 0.2))
])

print("Final comprehensive training pipeline created!")
print("\nTransform chain:")
for i, t in enumerate(final_train_transform.transforms):
    print(f"  {i+1}. {t}")

# %%
# Apply final transform
final_samples = []
for _ in range(8):
    aug = final_train_transform(original_image)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    final_samples.append(aug * std + mean)

print("\nFinal Pipeline Results:")
show_images_grid(final_samples, titles=[f'Sample {i+1}' for i in range(8)], nrow=4, figsize=(14, 8))

# %% [markdown]
# ## แบบฝึกหัด (Exercises)
#
# 1. **Exercise 1**: สร้าง transform pipeline สำหรับ medical image classification 
#    (hint: ระวังการ flip และ rotation ที่อาจไม่เหมาะสม)
#
# 2. **Exercise 2**: เปรียบเทียบ performance ของ model ที่ train ด้วย:
#    - No augmentation
#    - Basic augmentation
#    - RandAugment
#
# 3. **Exercise 3**: สร้าง custom transform ที่:
#    - เพิ่ม salt-and-pepper noise
#    - ทำ channel shuffle
#    - สร้าง mosaic augmentation
#
# 4. **Exercise 4**: Download Animal Faces dataset และ:
#    - วิเคราะห์ class distribution
#    - สร้าง augmentation pipeline ที่เหมาะสม
#    - Train simple classifier และวัดผล

# %%
print("=" * 60)
print("Lab: Image Augmentation using PyTorch - Complete!")
print("=" * 60)
print("\nKey Takeaways:")
print("• Image augmentation helps reduce overfitting")
print("• torchvision.transforms provides comprehensive tools")
print("• Different tasks need different augmentation strategies")
print("• RandAugment/AutoAugment are good starting points")
print("• Always keep validation data unaugmented")
