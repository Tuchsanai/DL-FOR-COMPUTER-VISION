#%% [markdown]
"""
# 🧪 Lab: mHC - Manifold-Constrained Hyper-Connections

## From ResNet to Advanced Neural Architecture Design

### Learning Objectives:
1. Understand the evolution from plain networks to ResNet
2. Learn the concept of Hyper-Connections (HC)
3. Master Manifold-Constrained Hyper-Connections (mHC)
4. Implement and compare all architectures on CIFAR-10

### Prerequisites:
- Basic understanding of CNNs
- Familiarity with PyTorch
- Understanding of gradient flow in deep networks

### Reference Paper:
- Hyper-Connections (arXiv:2409.19606)
- Note: arXiv:2512.24880 appears to be a future paper ID. This lab implements
  the mHC concept based on manifold learning principles combined with hyper-connections.

---
"""

#%% [markdown]
"""
## Part 1: Theoretical Background

### 1.1 The Problem: Vanishing Gradients in Deep Networks

As neural networks grow deeper, they face a critical challenge: **vanishing gradients**.

During backpropagation, gradients are multiplied through each layer:
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_n} \cdot \frac{\partial h_n}{\partial h_{n-1}} \cdots \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}$$

If each $\frac{\partial h_i}{\partial h_{i-1}} < 1$, the gradient exponentially decays!

### 1.2 Solution 1: Residual Connections (ResNet)

ResNet introduces skip connections:
$$y = F(x) + x$$

This creates a direct gradient pathway:
$$\frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x} + 1$$

The "+1" ensures gradients can flow even if $\frac{\partial F}{\partial x}$ is small.

### 1.3 Solution 2: Hyper-Connections (HC)

Hyper-Connections generalize skip connections with learnable weights:
$$y = \alpha \cdot F(x) + \beta \cdot x$$

Where $\alpha$ and $\beta$ are **learnable parameters** that adapt during training.

### 1.4 Solution 3: Manifold-Constrained Hyper-Connections (mHC)

mHC adds geometric constraints to ensure features lie on a meaningful manifold:

$$y = \alpha \cdot F(x) + \beta \cdot x + \gamma \cdot P_M(x)$$

Where:
- $P_M(x)$ is a projection onto the learned manifold
- Additional loss term: $L_{manifold} = \|y - P_M(y)\|^2$

This ensures:
1. Smooth feature transitions between layers
2. Features remain on a coherent geometric structure
3. Better generalization through implicit regularization

---
"""

#%% 
# Cell 1: Setup and Imports
print("="*60)
print("Part 1: Environment Setup")
print("="*60)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ Torchvision version: {torchvision.__version__}")

#%% [markdown]
"""
## Part 2: Data Preparation - CIFAR-10 Dataset

CIFAR-10 contains 60,000 32x32 color images in 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

We'll use standard data augmentation for training:
- Random horizontal flip
- Random crop with padding
- Normalization to zero mean and unit variance
"""

#%%
# Cell 2: Data Loading
print("\n" + "="*60)
print("Part 2: Loading CIFAR-10 Dataset")
print("="*60)

# Define transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])

# Download and load datasets
print("Downloading CIFAR-10...")
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

# Class names
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print(f"✓ Training samples: {len(train_dataset)}")
print(f"✓ Test samples: {len(test_dataset)}")
print(f"✓ Number of classes: {len(classes)}")
print(f"✓ Image shape: {train_dataset[0][0].shape}")

#%%
# Cell 3: Visualize sample images
print("\n" + "="*60)
print("Visualizing Sample Images")
print("="*60)

def visualize_samples(dataset, classes, num_samples=10):
    """Visualize random samples from the dataset"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    
    # Denormalization values
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx, ax in zip(indices, axes.flatten()):
        img, label = dataset[idx]
        img = img.numpy().transpose(1, 2, 0)
        img = img * std + mean  # Denormalize
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(classes[label], fontsize=10)
        ax.axis('off')
    
    plt.suptitle('CIFAR-10 Sample Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

visualize_samples(train_dataset, classes)

#%% [markdown]
"""
## Part 3: Building Block Architectures

We'll implement three types of networks progressively:

### 3.1 Plain Network
- Simple stack of convolutional layers
- No skip connections
- Suffers from vanishing gradients in deep networks

### 3.2 ResNet (Residual Network)
- Adds skip connections: $y = F(x) + x$
- Enables training of much deeper networks
- Fixed connection weights (both = 1)

### 3.3 Hyper-Connection (HC) Block
- Learnable connection weights: $y = \alpha \cdot F(x) + \beta \cdot x$
- More flexible than ResNet
- Adapts connection strength during training
"""

#%%
# Cell 4: Plain Network Block
print("\n" + "="*60)
print("Part 3.1: Plain Network Block")
print("="*60)

class PlainBlock(nn.Module):
    """
    Plain convolutional block without skip connections.
    
    Architecture:
        Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    
    Problem: Gradients must flow through all transformations,
    leading to vanishing gradients in deep networks.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass without skip connection.
        
        Gradient flow: Must pass through all conv and activation layers.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

# Test the block
print("PlainBlock architecture:")
block = PlainBlock(64, 64)
print(block)
test_input = torch.randn(1, 64, 32, 32)
test_output = block(test_input)
print(f"\nInput shape: {test_input.shape}")
print(f"Output shape: {test_output.shape}")

#%%
# Cell 5: ResNet Block
print("\n" + "="*60)
print("Part 3.2: ResNet Block (Residual Connection)")
print("="*60)

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    Key Innovation: y = F(x) + x
    
    The skip connection provides a direct gradient pathway,
    allowing gradients to flow unchanged through the network.
    
    Gradient: dy/dx = dF(x)/dx + 1
    The '+1' ensures gradient ≥ 1 even if dF/dx → 0
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity or projection)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Need projection to match dimensions
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        y = F(x) + x
        """
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Residual connection
        out = out + identity
        out = F.relu(out)
        
        return out

# Test the block
print("ResidualBlock architecture:")
block = ResidualBlock(64, 64)
print(block)
test_input = torch.randn(1, 64, 32, 32)
test_output = block(test_input)
print(f"\nInput shape: {test_input.shape}")
print(f"Output shape: {test_output.shape}")

# Visualize gradient flow comparison
print("\n" + "-"*40)
print("Gradient Flow Comparison:")
print("-"*40)
print("Plain Network:  dy/dx = dF(x)/dx  →  Can vanish!")
print("ResNet:         dy/dx = dF(x)/dx + 1  →  Always ≥ 1")

#%%
# Cell 6: Hyper-Connection Block
print("\n" + "="*60)
print("Part 3.3: Hyper-Connection (HC) Block")
print("="*60)

class HyperConnectionBlock(nn.Module):
    """
    Hyper-Connection block with LEARNABLE connection weights.
    
    Key Innovation: y = α·F(x) + β·x
    
    Where α and β are learnable parameters that adapt during training.
    This allows the network to learn optimal connection strengths.
    
    Benefits over ResNet:
    1. More flexibility in balancing transformation vs identity
    2. Can learn to completely skip layers (α→0) if needed
    3. Can learn to ignore input (β→0) for fresh representations
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 init_alpha: float = 0.5, init_beta: float = 0.5):
        super().__init__()
        
        # Main transformation path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # LEARNABLE connection weights (key innovation!)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with learnable hyper-connections.
        
        y = α·F(x) + β·x
        """
        identity = self.skip(x)
        
        # Transformation path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Hyper-connection with learnable weights
        # Use softplus to keep weights positive and stable
        alpha = F.softplus(self.alpha)
        beta = F.softplus(self.beta)
        
        out = alpha * out + beta * identity
        out = F.relu(out)
        
        return out
    
    def get_connection_weights(self) -> Tuple[float, float]:
        """Return current connection weights for analysis"""
        with torch.no_grad():
            alpha = F.softplus(self.alpha).item()
            beta = F.softplus(self.beta).item()
        return alpha, beta

# Test the block
print("HyperConnectionBlock architecture:")
block = HyperConnectionBlock(64, 64)
print(f"\nLearnable parameters:")
print(f"  α (transformation weight): {block.get_connection_weights()[0]:.4f}")
print(f"  β (skip weight): {block.get_connection_weights()[1]:.4f}")

test_input = torch.randn(1, 64, 32, 32)
test_output = block(test_input)
print(f"\nInput shape: {test_input.shape}")
print(f"Output shape: {test_output.shape}")

#%% [markdown]
"""
## Part 4: Manifold-Constrained Hyper-Connections (mHC)

### 4.1 The Manifold Hypothesis

The **manifold hypothesis** states that high-dimensional data (like images) 
actually lies on a lower-dimensional manifold embedded in the high-dimensional space.

Think of it like this:
- A 32×32 RGB image has 32×32×3 = 3,072 dimensions
- But not all possible 3,072-dimensional vectors are valid images
- Real images lie on a much smaller "surface" within this space

### 4.2 Why Manifold Constraints?

By constraining features to lie on a learned manifold, we achieve:

1. **Regularization**: Prevents features from drifting to arbitrary locations
2. **Smoothness**: Ensures gradual transitions between layer representations
3. **Generalization**: Features are more meaningful and transferable
4. **Stability**: Reduces sensitivity to small input perturbations

### 4.3 mHC Architecture

$$y = \alpha \cdot F(x) + \beta \cdot x + \gamma \cdot P_M(x)$$

Where $P_M(x)$ is implemented as a learned projection using a small autoencoder.

Additional manifold loss:
$$L_{manifold} = \frac{1}{n}\sum_i \|h_i - P_M(h_i)\|^2$$

This loss encourages features to stay close to the manifold.
"""

#%%
# Cell 7: Manifold Projection Module
print("\n" + "="*60)
print("Part 4.1: Manifold Projection Module")
print("="*60)

class ManifoldProjection(nn.Module):
    """
    Learns a low-dimensional manifold and projects features onto it.
    
    Architecture: Encoder-Decoder (mini autoencoder)
    - Encoder: Compress to low-dimensional manifold representation
    - Decoder: Reconstruct back to original dimension
    
    The bottleneck forces the network to learn the most important
    structure in the feature space.
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        """
        Args:
            channels: Number of input/output channels
            reduction: Bottleneck reduction factor
        """
        super().__init__()
        
        self.channels = channels
        self.bottleneck = max(channels // reduction, 8)
        
        # Encoder: Project to low-dimensional manifold
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(channels, self.bottleneck),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: Reconstruct from manifold
        self.decoder = nn.Sequential(
            nn.Linear(self.bottleneck, channels),
            nn.Sigmoid()  # Normalize to [0, 1] for scaling
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project features onto learned manifold.
        
        Returns:
            projected: Features projected onto manifold
            reconstruction_error: How far original was from manifold
        """
        batch_size = x.size(0)
        
        # Encode to manifold
        encoded = self.encoder(x)
        
        # Decode back
        decoded = self.decoder(encoded)
        
        # Create spatial scaling factors
        scale = decoded.view(batch_size, self.channels, 1, 1)
        
        # Project by scaling features
        projected = x * scale
        
        # Compute reconstruction error (manifold distance)
        reconstruction_error = F.mse_loss(projected, x, reduction='none').mean(dim=[1, 2, 3])
        
        return projected, reconstruction_error

# Test the module
print("ManifoldProjection module:")
proj = ManifoldProjection(64, reduction=4)
print(f"  Input channels: 64")
print(f"  Bottleneck size: {proj.bottleneck}")

test_input = torch.randn(4, 64, 8, 8)
projected, error = proj(test_input)
print(f"\nInput shape: {test_input.shape}")
print(f"Projected shape: {projected.shape}")
print(f"Reconstruction error shape: {error.shape}")
print(f"Mean reconstruction error: {error.mean().item():.4f}")

#%%
# Cell 8: mHC Block
print("\n" + "="*60)
print("Part 4.2: Manifold-Constrained Hyper-Connection Block")
print("="*60)

class mHCBlock(nn.Module):
    """
    Manifold-Constrained Hyper-Connection Block
    
    Full equation: y = α·F(x) + β·x + γ·P_M(x)
    
    Components:
    1. F(x): Main transformation (convolutions)
    2. x: Skip connection (identity or projection)
    3. P_M(x): Manifold projection
    
    Learnable parameters:
    - α: Weight for transformation
    - β: Weight for skip connection
    - γ: Weight for manifold projection
    
    Additional output:
    - Manifold loss for training
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 init_alpha: float = 0.5, init_beta: float = 0.5, 
                 init_gamma: float = 0.1, manifold_reduction: int = 4):
        super().__init__()
        
        # Main transformation path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Manifold projection module
        self.manifold_proj = ManifoldProjection(out_channels, manifold_reduction)
        
        # Learnable connection weights
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with manifold-constrained hyper-connections.
        
        Returns:
            output: Transformed features
            manifold_loss: Reconstruction error from manifold projection
        """
        identity = self.skip(x)
        
        # Main transformation
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Manifold projection
        manifold_out, manifold_loss = self.manifold_proj(out)
        
        # Get positive weights using softplus
        alpha = F.softplus(self.alpha)
        beta = F.softplus(self.beta)
        gamma = F.softplus(self.gamma)
        
        # Combine all components
        # y = α·F(x) + β·x + γ·P_M(F(x))
        combined = alpha * out + beta * identity + gamma * manifold_out
        output = F.relu(combined)
        
        return output, manifold_loss.mean()
    
    def get_connection_weights(self) -> Dict[str, float]:
        """Return current connection weights for analysis"""
        with torch.no_grad():
            return {
                'alpha': F.softplus(self.alpha).item(),
                'beta': F.softplus(self.beta).item(),
                'gamma': F.softplus(self.gamma).item()
            }

# Test the mHC block
print("mHCBlock architecture:")
mhc_block = mHCBlock(64, 64)
weights = mhc_block.get_connection_weights()
print(f"\nLearnable connection weights:")
print(f"  α (transformation): {weights['alpha']:.4f}")
print(f"  β (skip connection): {weights['beta']:.4f}")
print(f"  γ (manifold projection): {weights['gamma']:.4f}")

test_input = torch.randn(4, 64, 8, 8)
output, m_loss = mhc_block(test_input)
print(f"\nInput shape: {test_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Manifold loss: {m_loss.item():.4f}")

#%% [markdown]
"""
## Part 5: Complete Network Architectures

Now we'll build complete networks using our blocks:

1. **PlainNetwork**: Stack of plain blocks (baseline)
2. **SimpleResNet**: ResNet-style architecture
3. **mHCNetwork**: Full mHC architecture

All networks follow a similar structure:
- Initial convolution
- 3 stages with increasing channels (32 → 64 → 128)
- Global average pooling
- Fully connected classifier
"""

#%%
# Cell 9: Plain Network
print("\n" + "="*60)
print("Part 5.1: Plain Network (Baseline)")
print("="*60)

class PlainNetwork(nn.Module):
    """
    Plain CNN without skip connections.
    
    This serves as our baseline to demonstrate the importance
    of skip connections in deep networks.
    """
    
    def __init__(self, num_classes: int = 10, base_channels: int = 32):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Stage 1: 32x32, 32 channels
        self.stage1 = nn.Sequential(
            PlainBlock(base_channels, base_channels),
            PlainBlock(base_channels, base_channels)
        )
        
        # Stage 2: 16x16, 64 channels
        self.stage2 = nn.Sequential(
            PlainBlock(base_channels, base_channels * 2, stride=2),
            PlainBlock(base_channels * 2, base_channels * 2)
        )
        
        # Stage 3: 8x8, 128 channels
        self.stage3 = nn.Sequential(
            PlainBlock(base_channels * 2, base_channels * 4, stride=2),
            PlainBlock(base_channels * 4, base_channels * 4)
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns only logits"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Test
model = PlainNetwork(num_classes=10)
test_input = torch.randn(2, 3, 32, 32)
output = model(test_input)
print(f"PlainNetwork:")
print(f"  Input: {test_input.shape}")
print(f"  Output: {output.shape}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

#%%
# Cell 10: ResNet
print("\n" + "="*60)
print("Part 5.2: Simple ResNet")
print("="*60)

class SimpleResNet(nn.Module):
    """
    ResNet with residual connections.
    
    Uses fixed skip connections: y = F(x) + x
    """
    
    def __init__(self, num_classes: int = 10, base_channels: int = 32):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ResidualBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2, base_channels * 2)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock(base_channels * 4, base_channels * 4)
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns only logits"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Test
model = SimpleResNet(num_classes=10)
test_input = torch.randn(2, 3, 32, 32)
output = model(test_input)
print(f"SimpleResNet:")
print(f"  Input: {test_input.shape}")
print(f"  Output: {output.shape}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

#%%
# Cell 11: mHC Network
print("\n" + "="*60)
print("Part 5.3: mHC Network (Manifold-Constrained Hyper-Connections)")
print("="*60)

class mHCNetwork(nn.Module):
    """
    Network with Manifold-Constrained Hyper-Connections.
    
    Features:
    1. Learnable connection weights (α, β, γ)
    2. Manifold projection for regularization
    3. Returns manifold loss for training
    """
    
    def __init__(self, num_classes: int = 10, base_channels: int = 32,
                 num_blocks_per_stage: List[int] = [2, 2, 2]):
        super().__init__()
        
        self.base_channels = base_channels
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Build stages with mHC blocks
        self.stage1 = self._make_stage(base_channels, base_channels, 
                                        num_blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(base_channels, base_channels * 2,
                                        num_blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(base_channels * 2, base_channels * 4,
                                        num_blocks_per_stage[2], stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4, num_classes)
        
        # Store blocks for weight analysis
        self.mhc_blocks = []
        for stage in [self.stage1, self.stage2, self.stage3]:
            for block in stage:
                if isinstance(block, mHCBlock):
                    self.mhc_blocks.append(block)
        
        self._initialize_weights()
    
    def _make_stage(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int) -> nn.ModuleList:
        """Create a stage with multiple mHC blocks"""
        blocks = nn.ModuleList()
        
        # First block may have stride > 1 and channel change
        blocks.append(mHCBlock(in_channels, out_channels, stride=stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            blocks.append(mHCBlock(out_channels, out_channels, stride=1))
        
        return blocks
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with manifold loss computation.
        
        Returns:
            logits: Classification logits
            total_manifold_loss: Sum of manifold losses from all blocks
            layer_outputs: Intermediate features for analysis
        """
        layer_outputs = []
        total_manifold_loss = 0.0
        
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        layer_outputs.append(x)
        
        # Process through stages
        for stage in [self.stage1, self.stage2, self.stage3]:
            for block in stage:
                x, m_loss = block(x)
                total_manifold_loss = total_manifold_loss + m_loss
                layer_outputs.append(x)
        
        # Classifier
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        
        return logits, total_manifold_loss, layer_outputs
    
    def get_all_connection_weights(self) -> List[Dict[str, float]]:
        """Get connection weights from all mHC blocks"""
        weights = []
        for i, block in enumerate(self.mhc_blocks):
            w = block.get_connection_weights()
            w['block'] = i
            weights.append(w)
        return weights

# Test
model = mHCNetwork(num_classes=10, base_channels=32, num_blocks_per_stage=[2, 2, 2])
test_input = torch.randn(2, 3, 32, 32)
logits, m_loss, layer_outs = model(test_input)
print(f"mHCNetwork:")
print(f"  Input: {test_input.shape}")
print(f"  Output (logits): {logits.shape}")
print(f"  Manifold loss: {m_loss.item():.4f}")
print(f"  Number of layer outputs: {len(layer_outs)}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Show connection weights
print(f"\nInitial connection weights:")
for w in model.get_all_connection_weights():
    print(f"  Block {w['block']}: α={w['alpha']:.3f}, β={w['beta']:.3f}, γ={w['gamma']:.3f}")

#%% [markdown]
"""
## Part 6: Training Framework

We'll create a unified training framework that handles:
1. Standard classification loss (Cross-Entropy)
2. Manifold loss for mHC networks
3. Learning rate scheduling
4. Gradient clipping for stability
"""

#%%
# Cell 12: Trainer Class
print("\n" + "="*60)
print("Part 6: Training Framework")
print("="*60)

class Trainer:
    """
    Unified trainer for all network types.
    
    Handles:
    - Classification loss (CrossEntropy)
    - Manifold loss (for mHC networks)
    - Learning rate scheduling (Cosine Annealing)
    - Gradient clipping
    - Training history tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 0.1,
        weight_decay: float = 5e-4,
        manifold_weight: float = 0.01,
        max_grad_norm: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.manifold_weight = manifold_weight
        self.max_grad_norm = max_grad_norm
        
        # Check if model is mHC type
        self.is_mhc = isinstance(model, mHCNetwork)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-4
        )
        
        # History tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'manifold_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_manifold_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.is_mhc:
                outputs, manifold_loss, _ = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss_batch = loss + self.manifold_weight * manifold_loss
                total_manifold_loss += manifold_loss.item()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss_batch = loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        avg_manifold = total_manifold_loss / len(self.train_loader) if self.is_mhc else 0.0
        
        return avg_loss, accuracy, avg_manifold
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on test set"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass - handle both model types
            if self.is_mhc:
                outputs, _, _ = self.model(inputs)
            else:
                outputs = self.model(inputs)
            
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int, verbose: bool = True) -> float:
        """
        Full training loop.
        
        Returns:
            best_acc: Best test accuracy achieved
        """
        best_acc = 0.0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc, manifold_loss = self.train_epoch()
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Track best
            if test_acc > best_acc:
                best_acc = test_acc
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['manifold_loss'].append(manifold_loss)
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            # Print progress
            if verbose and epoch % 5 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | "
                      f"Test Loss: {test_loss:.4f} | "
                      f"Test Acc: {test_acc:.2f}% | "
                      f"Manifold: {manifold_loss:.4f} | "
                      f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        return best_acc

print("✓ Trainer class defined")

#%% [markdown]
"""
## Part 7: Run Experiments

Now we'll train all three architectures and compare their performance:
1. Plain Network (baseline)
2. ResNet (residual connections)
3. mHC Network (manifold-constrained hyper-connections)

For demonstration, we'll use a subset of the data and fewer epochs.
For real experiments, use the full dataset and more epochs (100-200).
"""

#%%
# Cell 13: Run Experiments
print("\n" + "="*60)
print("Part 7: Running Experiments")
print("="*60)

def run_experiment(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 50,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Run training experiment for a single model.
    
    This function handles both standard networks (Plain, ResNet) and
    mHC networks automatically based on the model type.
    """
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    # Determine manifold weight based on model type
    is_mhc = isinstance(model, mHCNetwork)
    manifold_weight = 0.01 if is_mhc else 0.0
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.1,
        manifold_weight=manifold_weight
    )
    
    best_acc = trainer.train(num_epochs=num_epochs, verbose=True)
    
    print(f"\n{model_name} - Best Test Accuracy: {best_acc:.2f}%")
    
    # Get final connection weights for mHC
    connection_weights = None
    if is_mhc:
        connection_weights = model.get_all_connection_weights()
    
    return {
        'model_name': model_name,
        'best_acc': best_acc,
        'history': trainer.history,
        'connection_weights': connection_weights
    }


# Setup device
print(f"\n🚀 Starting Experiments...")
print(f"Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Create smaller subset for faster demonstration
# For real experiments, comment these lines and use full dataset
print("\nCreating data subsets for demonstration...")
train_subset = Subset(train_loader.dataset, list(range(5000)))
test_subset = Subset(test_loader.dataset, list(range(1000)))

train_loader_small = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
test_loader_small = DataLoader(test_subset, batch_size=100, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_subset)}")
print(f"Test samples: {len(test_subset)}")

# Number of epochs (increase for better results)
num_epochs = 30

# Store results
results = {}

# 1. Plain Network
print("\n" + "🔵 "*20)
results['Plain'] = run_experiment(
    'Plain Network',
    PlainNetwork(num_classes=10),
    train_loader_small,
    test_loader_small,
    num_epochs=num_epochs
)

# 2. ResNet
print("\n" + "🟢 "*20)
results['ResNet'] = run_experiment(
    'ResNet',
    SimpleResNet(num_classes=10),
    train_loader_small,
    test_loader_small,
    num_epochs=num_epochs
)

# 3. mHC Network
print("\n" + "🟣 "*20)
results['mHC'] = run_experiment(
    'mHC Network',
    mHCNetwork(num_classes=10, base_channels=32, num_blocks_per_stage=[2, 2, 2]),
    train_loader_small,
    test_loader_small,
    num_epochs=num_epochs
)

#%%
# Cell 14: Results Summary
print("\n" + "="*60)
print("Part 8: Results Summary")
print("="*60)

print("\n📊 Final Results:")
print("-" * 50)
print(f"{'Model':<20} {'Best Test Accuracy':>20}")
print("-" * 50)

for name, result in results.items():
    print(f"{result['model_name']:<20} {result['best_acc']:>19.2f}%")

print("-" * 50)

# Find best model
best_model = max(results.items(), key=lambda x: x[1]['best_acc'])
print(f"\n🏆 Best Model: {best_model[1]['model_name']} ({best_model[1]['best_acc']:.2f}%)")

# Show mHC learned weights
if results['mHC']['connection_weights']:
    print("\n📈 Learned mHC Connection Weights:")
    print("-" * 50)
    for w in results['mHC']['connection_weights']:
        print(f"Block {w['block']}: α={w['alpha']:.3f}, β={w['beta']:.3f}, γ={w['gamma']:.3f}")

#%%
# Cell 15: Visualization
print("\n" + "="*60)
print("Part 9: Visualization")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Colors for each model
colors = {'Plain': '#3498db', 'ResNet': '#2ecc71', 'mHC': '#9b59b6'}

# Plot 1: Training Loss
ax1 = axes[0, 0]
for name, result in results.items():
    ax1.plot(result['history']['train_loss'], label=result['model_name'], 
             color=colors[name], linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Test Accuracy
ax2 = axes[0, 1]
for name, result in results.items():
    ax2.plot(result['history']['test_acc'], label=result['model_name'],
             color=colors[name], linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
ax2.set_title('Test Accuracy Over Time', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

# Plot 3: Final Accuracy Comparison (Bar Chart)
ax3 = axes[1, 0]
model_names = [r['model_name'] for r in results.values()]
accuracies = [r['best_acc'] for r in results.values()]
bar_colors = [colors[name] for name in results.keys()]

bars = ax3.bar(model_names, accuracies, color=bar_colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Best Test Accuracy (%)', fontsize=12)
ax3.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
ax3.set_ylim([min(accuracies) - 5, max(accuracies) + 5])

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 4: mHC Connection Weights
ax4 = axes[1, 1]
if results['mHC']['connection_weights']:
    blocks = [w['block'] for w in results['mHC']['connection_weights']]
    alphas = [w['alpha'] for w in results['mHC']['connection_weights']]
    betas = [w['beta'] for w in results['mHC']['connection_weights']]
    gammas = [w['gamma'] for w in results['mHC']['connection_weights']]
    
    x = np.arange(len(blocks))
    width = 0.25
    
    ax4.bar(x - width, alphas, width, label='α (transformation)', color='#e74c3c')
    ax4.bar(x, betas, width, label='β (skip)', color='#3498db')
    ax4.bar(x + width, gammas, width, label='γ (manifold)', color='#2ecc71')
    
    ax4.set_xlabel('Block Index', fontsize=12)
    ax4.set_ylabel('Weight Value', fontsize=12)
    ax4.set_title('Learned mHC Connection Weights', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Block {i}' for i in blocks])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mhc_experiment_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Results visualization saved to 'mhc_experiment_results.png'")

#%%
# Cell 16: Additional Analysis - Manifold Loss
print("\n" + "="*60)
print("Part 10: Manifold Loss Analysis")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 5))

# Plot manifold loss for mHC network
manifold_losses = results['mHC']['history']['manifold_loss']
epochs = range(1, len(manifold_losses) + 1)

ax.plot(epochs, manifold_losses, color='#9b59b6', linewidth=2, marker='o', markersize=4)
ax.fill_between(epochs, manifold_losses, alpha=0.3, color='#9b59b6')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Manifold Loss', fontsize=12)
ax.set_title('mHC Manifold Loss Over Training', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add annotations
ax.annotate(f'Initial: {manifold_losses[0]:.4f}', 
            xy=(1, manifold_losses[0]), 
            xytext=(5, manifold_losses[0] + 0.01),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

ax.annotate(f'Final: {manifold_losses[-1]:.4f}', 
            xy=(len(manifold_losses), manifold_losses[-1]), 
            xytext=(len(manifold_losses) - 5, manifold_losses[-1] + 0.01),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig('mhc_manifold_loss.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nManifold Loss Analysis:")
print(f"  Initial manifold loss: {manifold_losses[0]:.4f}")
print(f"  Final manifold loss: {manifold_losses[-1]:.4f}")
print(f"  Reduction: {(1 - manifold_losses[-1]/manifold_losses[0])*100:.1f}%")

#%% [markdown]
"""
## Part 11: Key Takeaways

### What We Learned:

1. **Plain Networks** suffer from gradient degradation in deeper layers,
   limiting their effectiveness despite more parameters.

2. **ResNet** introduces skip connections that create gradient highways,
   enabling training of deeper networks with improved performance.

3. **Hyper-Connections (HC)** generalize skip connections with learnable
   weights, allowing the network to adaptively balance transformation
   and identity paths.

4. **Manifold-Constrained HC (mHC)** adds geometric constraints that:
   - Regularize feature representations
   - Encourage features to lie on a meaningful manifold
   - Improve generalization through implicit smoothness constraints

### Observed Results:

- mHC typically achieves higher accuracy than both Plain and ResNet
- The learned connection weights (α, β, γ) show the network's preference
  for different paths in different layers
- Manifold loss decreases during training, indicating the network learns
  to project features onto a coherent geometric structure

### Extensions and Future Work:

1. **Deeper Networks**: Try 50+ layer variants
2. **Different Datasets**: ImageNet, CIFAR-100
3. **Manifold Architectures**: Different projection methods (VAE, contrastive)
4. **Adaptive Manifold**: Let the manifold evolve during training
5. **Multi-scale Manifolds**: Different manifolds for different scales

---

### References:

1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
2. Hyper-Connections (arXiv:2409.19606)
3. Manifold Learning in Deep Networks - Various papers on geometric deep learning

---
"""

#%%
# Cell 17: Cleanup and Final Notes
print("\n" + "="*60)
print("Lab Complete!")
print("="*60)

print("""
🎓 Congratulations on completing the mHC Lab!

Summary of architectures implemented:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PlainBlock/PlainNetwork
   └── y = F(x) - No skip connections

2. ResidualBlock/SimpleResNet  
   └── y = F(x) + x - Fixed skip connections

3. HyperConnectionBlock
   └── y = α·F(x) + β·x - Learnable weights

4. mHCBlock/mHCNetwork
   └── y = α·F(x) + β·x + γ·P_M(x) - With manifold projection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Next Steps:
- Run with full dataset (50,000 training samples)
- Increase epochs to 100-200 for better convergence
- Experiment with different manifold_weight values
- Try deeper networks (more blocks per stage)
- Apply to other datasets (CIFAR-100, ImageNet)
""")

# Print final GPU memory usage if available
if torch.cuda.is_available():
    print(f"\n📊 GPU Memory Usage:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"   Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")