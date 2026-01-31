# %% [markdown]
# # 🎯 Qwen3-TTS Fine-tuning Lab: Train Your Own Voice Model
#
# **Course:** Machine Learning / Deep Learning  
# **Topic:** Fine-tuning Text-to-Speech Models  
# **Model:** Qwen3-TTS (Alibaba, January 2025)
#
# ---
#
# ## Learning Objectives
#
# By the end of this lab, you will be able to:
#
# 1. Understand the Qwen3-TTS fine-tuning architecture
# 2. Prepare audio datasets for TTS training
# 3. Configure training hyperparameters
# 4. Fine-tune Qwen3-TTS on custom voice data
# 5. Evaluate and test the fine-tuned model
# 6. Export and deploy your trained model
#
# ---
#
# ## Prerequisites
#
# - GPU with at least 24GB VRAM (A100/A10/RTX 4090 recommended)
# - Basic understanding of PyTorch and Transformers
# - Familiarity with audio processing concepts
#
# ---
#
# ## Fine-tuning Overview
#
# ### Why Fine-tune?
#
# | Use Case | Benefit |
# |----------|---------|
# | **Custom Voice** | Train on specific speaker's voice |
# | **Domain Adaptation** | Improve on technical/medical terms |
# | **Language Expansion** | Better support for unsupported languages |
# | **Style Transfer** | Train specific speaking styles |
#
# ### Training Approaches
#
# | Approach | Data Required | VRAM | Quality |
# |----------|---------------|------|---------|
# | **Full Fine-tune** | 10+ hours | 80GB | Best |
# | **LoRA** | 1-5 hours | 24GB | Good |
# | **Prompt Tuning** | 10-30 mins | 16GB | Moderate |

# %%

# %% [markdown]
# ---
# ## Part 1: Environment Setup
#
# ติดตั้ง packages ที่จำเป็นสำหรับการ train model

# %%
# Cell 1.1: Install Required Packages
# Uncomment to install:

# !pip install -U transformers==4.57.3
# !pip install -U accelerate
# !pip install -U datasets
# !pip install -U peft  # For LoRA
# !pip install -U qwen-tts
# !pip install -U soundfile librosa
# !pip install -U tensorboard
# !pip install -U bitsandbytes  # For 8-bit training
# !pip install -U wandb  # Optional: for experiment tracking

# %%
# Cell 1.2: Import Libraries

import os
import json
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Transformers & Training
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset, DatasetDict, Audio
from peft import LoraConfig, get_peft_model, TaskType

# Qwen TTS
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

# Visualization
from IPython.display import Audio as IPAudio, display
import matplotlib.pyplot as plt

# %%
# Cell 1.3: Utility Functions

def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.2f}GB total")
    else:
        print("No GPU available")

def format_duration(seconds: float) -> str:
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# %%
# Cell 1.4: Check System Configuration

print_section("System Configuration")

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    
    if gpu_memory >= 24:
        print("✅ Sufficient GPU memory for LoRA fine-tuning")
    elif gpu_memory >= 16:
        print("⚠️ Limited memory - use smaller batch size or 8-bit training")
    else:
        print("❌ Insufficient GPU memory for fine-tuning")
else:
    print("❌ No CUDA GPU detected - fine-tuning requires GPU")

print_gpu_memory()

# %%
# Cell 1.5: Create Directory Structure

# Project directories
PROJECT_DIR = Path("./qwen3_tts_training")
DATA_DIR = PROJECT_DIR / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_DIR / "outputs"
LOG_DIR = PROJECT_DIR / "logs"

# Create all directories
for dir_path in [RAW_AUDIO_DIR, PROCESSED_DIR, CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created: {dir_path}")

print(f"\n📁 Project Structure:")
print(f"   {PROJECT_DIR}/")
print(f"   ├── data/")
print(f"   │   ├── raw_audio/     # Original audio files")
print(f"   │   └── processed/     # Processed dataset")
print(f"   ├── checkpoints/       # Model checkpoints")
print(f"   ├── outputs/           # Generated audio")
print(f"   └── logs/              # Training logs")

# %% [markdown]
# ---
# ## Part 2: Dataset Preparation
#
# การเตรียม dataset สำหรับ TTS training ประกอบด้วย:
# 1. Audio files (WAV format, 16kHz or 24kHz)
# 2. Transcriptions (text matching audio)
# 3. Speaker information (optional)
#
# ### Dataset Format
#
# ```
# data/
# ├── audio/
# │   ├── 001.wav
# │   ├── 002.wav
# │   └── ...
# └── metadata.json
# ```
#
# ### metadata.json format:
# ```json
# [
#     {"audio": "001.wav", "text": "Hello world", "speaker": "speaker_1"},
#     {"audio": "002.wav", "text": "How are you", "speaker": "speaker_1"}
# ]
# ```

# %%
# Cell 2.1: Audio Processing Functions

class AudioProcessor:
    """Audio processing utilities for TTS dataset preparation"""
    
    def __init__(self, target_sr: int = 24000):
        self.target_sr = target_sr
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return waveform and sample rate"""
        wav, sr = librosa.load(audio_path, sr=None)
        return wav, sr
    
    def resample(self, wav: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr != self.target_sr:
            wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=self.target_sr)
        return wav
    
    def normalize(self, wav: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.abs(wav).max()
        if max_val > 0:
            wav = wav / max_val * 0.95  # Leave some headroom
        return wav
    
    def trim_silence(self, wav: np.ndarray, top_db: int = 20) -> np.ndarray:
        """Trim silence from beginning and end"""
        wav_trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
        return wav_trimmed
    
    def get_duration(self, wav: np.ndarray) -> float:
        """Get audio duration in seconds"""
        return len(wav) / self.target_sr
    
    def process_audio(self, audio_path: str, 
                      normalize: bool = True,
                      trim: bool = True) -> Tuple[np.ndarray, float]:
        """Full audio processing pipeline"""
        # Load
        wav, sr = self.load_audio(audio_path)
        
        # Resample
        wav = self.resample(wav, sr)
        
        # Trim silence
        if trim:
            wav = self.trim_silence(wav)
        
        # Normalize
        if normalize:
            wav = self.normalize(wav)
        
        duration = self.get_duration(wav)
        return wav, duration
    
    def save_audio(self, wav: np.ndarray, output_path: str):
        """Save audio to file"""
        sf.write(output_path, wav, self.target_sr)

# Initialize processor
audio_processor = AudioProcessor(target_sr=24000)
print("✅ AudioProcessor initialized (target_sr=24000)")

# %%
# Cell 2.2: Create Sample Dataset (Demo)

def create_sample_dataset():
    """
    Create a sample dataset for demonstration
    In real usage, replace with your own audio data
    """
    print_section("Creating Sample Dataset")
    
    # Sample texts for demonstration
    sample_data = [
        {
            "id": "001",
            "text": "Welcome to the machine learning course.",
            "speaker": "teacher_1",
            "language": "English"
        },
        {
            "id": "002", 
            "text": "Today we will learn about neural networks.",
            "speaker": "teacher_1",
            "language": "English"
        },
        {
            "id": "003",
            "text": "Deep learning has revolutionized artificial intelligence.",
            "speaker": "teacher_1",
            "language": "English"
        },
        {
            "id": "004",
            "text": "Let's start with the basics of PyTorch.",
            "speaker": "teacher_1",
            "language": "English"
        },
        {
            "id": "005",
            "text": "Practice is essential for mastering these concepts.",
            "speaker": "teacher_1",
            "language": "English"
        },
    ]
    
    # Save metadata
    metadata_path = DATA_DIR / "sample_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created sample metadata: {metadata_path}")
    print(f"   Total samples: {len(sample_data)}")
    
    return sample_data

# Create sample dataset
sample_data = create_sample_dataset()

# %%
# Cell 2.3: Generate Training Audio with Base Model

def generate_training_audio(metadata: List[Dict], 
                           output_dir: Path,
                           model: Qwen3TTSModel,
                           speaker_description: str):
    """
    Generate audio files using Qwen3-TTS for training data
    This creates synthetic data for demonstration
    In real usage, use actual recorded audio
    """
    print_section("Generating Training Audio")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    total_duration = 0
    
    for i, item in enumerate(metadata):
        print(f"\n[{i+1}/{len(metadata)}] Generating: {item['text'][:50]}...")
        
        try:
            # Generate audio
            wavs, sr = model.generate_voice_design(
                text=item['text'],
                language=item.get('language', 'English'),
                instruct=speaker_description,
            )
            
            # Process audio
            wav = wavs[0]
            wav, duration = audio_processor.process_audio_array(wav)
            
            # Save
            audio_filename = f"{item['id']}.wav"
            audio_path = output_dir / audio_filename
            audio_processor.save_audio(wav, str(audio_path))
            
            # Update metadata
            item['audio_path'] = str(audio_path)
            item['duration'] = duration
            total_duration += duration
            
            generated_files.append(item)
            print(f"   ✅ Saved: {audio_filename} ({duration:.2f}s)")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    print(f"\n📊 Generation Summary:")
    print(f"   Total files: {len(generated_files)}")
    print(f"   Total duration: {format_duration(total_duration)}")
    
    return generated_files

# Add method to AudioProcessor for array processing
def process_audio_array(self, wav: np.ndarray) -> Tuple[np.ndarray, float]:
    """Process audio array (already loaded)"""
    wav = self.normalize(wav)
    wav = self.trim_silence(wav)
    duration = self.get_duration(wav)
    return wav, duration

AudioProcessor.process_audio_array = process_audio_array

# %%
# Cell 2.4: Load or Generate Training Data

print_section("Preparing Training Data")

# Check if we have existing audio or need to generate
existing_audio_files = list(RAW_AUDIO_DIR.glob("*.wav"))

if len(existing_audio_files) > 0:
    print(f"Found {len(existing_audio_files)} existing audio files")
    print("Using existing data...")
    
    # Load existing metadata if available
    metadata_path = DATA_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            training_data = json.load(f)
        print(f"Loaded metadata: {len(training_data)} samples")
    else:
        print("⚠️ No metadata.json found. Please create one.")
        training_data = []
else:
    print("No existing audio found.")
    print("For real training, add your audio files to:")
    print(f"  {RAW_AUDIO_DIR}")
    print("\nFor demo, we'll use the sample metadata (no audio generation)")
    
    # Use sample metadata
    training_data = sample_data
    print(f"\n📋 Sample training data: {len(training_data)} items")

# %%
# Cell 2.5: Create HuggingFace Dataset

def create_hf_dataset(metadata: List[Dict], 
                      audio_dir: Optional[Path] = None) -> Dataset:
    """
    Convert metadata to HuggingFace Dataset format
    """
    print_section("Creating HuggingFace Dataset")
    
    # Prepare data
    data_dict = {
        "id": [],
        "text": [],
        "speaker": [],
        "language": [],
    }
    
    # Add audio paths if available
    if audio_dir and audio_dir.exists():
        data_dict["audio"] = []
        has_audio = True
    else:
        has_audio = False
    
    for item in metadata:
        data_dict["id"].append(item["id"])
        data_dict["text"].append(item["text"])
        data_dict["speaker"].append(item.get("speaker", "default"))
        data_dict["language"].append(item.get("language", "English"))
        
        if has_audio:
            audio_path = audio_dir / f"{item['id']}.wav"
            if audio_path.exists():
                data_dict["audio"].append(str(audio_path))
            else:
                data_dict["audio"].append(None)
    
    # Create dataset
    dataset = Dataset.from_dict(data_dict)
    
    # Cast audio column if available
    if has_audio and any(data_dict["audio"]):
        dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))
    
    print(f"✅ Dataset created:")
    print(f"   Samples: {len(dataset)}")
    print(f"   Columns: {dataset.column_names}")
    print(f"   Has audio: {has_audio}")
    
    return dataset

# Create dataset (without audio for demo)
# For real training, pass audio_dir=RAW_AUDIO_DIR
demo_dataset = create_hf_dataset(training_data, audio_dir=None)
print(demo_dataset)

# %%
# Cell 2.6: Split Dataset

def split_dataset(dataset: Dataset, 
                  train_ratio: float = 0.9,
                  seed: int = 42) -> DatasetDict:
    """Split dataset into train and validation sets"""
    
    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    
    split_idx = int(len(dataset) * train_ratio)
    
    train_dataset = dataset.select(range(split_idx))
    val_dataset = dataset.select(range(split_idx, len(dataset)))
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    print(f"📊 Dataset Split:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    
    return dataset_dict

# Split dataset
dataset_splits = split_dataset(demo_dataset, train_ratio=0.8)

# %% [markdown]
# ---
# ## Part 3: Model Configuration
#
# การตั้งค่า model สำหรับ fine-tuning
#
# ### LoRA Configuration
#
# LoRA (Low-Rank Adaptation) ช่วยลด memory และเวลาในการ train:
#
# | Parameter | Description | Recommended |
# |-----------|-------------|-------------|
# | `r` | Rank of update matrices | 8-64 |
# | `lora_alpha` | Scaling factor | 16-32 |
# | `lora_dropout` | Dropout probability | 0.05-0.1 |
# | `target_modules` | Layers to adapt | attention layers |

# %%
# Cell 3.1: Load Base Model

print_section("Loading Base Model")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Device: {DEVICE}")
print(f"Dtype: {DTYPE}")
print("\nLoading model... (this may take a few minutes)")

# Load base model for fine-tuning
base_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",  # Use smaller model for demo
    device_map=DEVICE,
    dtype=DTYPE,
    attn_implementation="eager",  # Use eager for training compatibility
)

print("\n✅ Base model loaded!")
print_gpu_memory()

# %%
# Cell 3.2: Configure LoRA

print_section("Configuring LoRA")

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # Rank
    lora_alpha=32,                 # Scaling
    lora_dropout=0.05,             # Dropout
    target_modules=[               # Target attention layers
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
    ],
    bias="none",
    inference_mode=False,
)

print("📋 LoRA Configuration:")
print(f"   Rank (r): {lora_config.r}")
print(f"   Alpha: {lora_config.lora_alpha}")
print(f"   Dropout: {lora_config.lora_dropout}")
print(f"   Target modules: {lora_config.target_modules}")

# %%
# Cell 3.3: Apply LoRA to Model

print_section("Applying LoRA")

# Note: This is a simplified example
# Actual Qwen3-TTS may require specific adapter configuration

# For demonstration, we'll show the structure
print("⚠️ Note: Qwen3-TTS fine-tuning requires specific configuration")
print("   This demo shows the general approach")
print("\n📋 Steps for actual fine-tuning:")
print("   1. Extract the language model component")
print("   2. Apply LoRA adapters")
print("   3. Configure audio tokenizer training (if needed)")
print("   4. Set up the training loop")

# Count trainable parameters (example)
def count_parameters(model):
    """Count trainable parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

# For actual implementation:
# peft_model = get_peft_model(base_model.language_model, lora_config)
# trainable, total = count_parameters(peft_model)
# print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# %% [markdown]
# ---
# ## Part 4: Training Configuration
#
# ตั้งค่า hyperparameters สำหรับการ train

# %%
# Cell 4.1: Training Arguments

print_section("Training Arguments")

# Training configuration
training_config = {
    # Basic settings
    "output_dir": str(CHECKPOINT_DIR),
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    
    # Learning rate
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    
    # Optimization
    "optim": "adamw_torch",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    
    # Precision
    "bf16": torch.cuda.is_available(),
    "fp16": False,
    
    # Logging
    "logging_dir": str(LOG_DIR),
    "logging_steps": 10,
    "logging_first_step": True,
    
    # Checkpointing
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,
    
    # Evaluation
    "eval_strategy": "steps",
    "eval_steps": 100,
    
    # Other
    "seed": 42,
    "dataloader_num_workers": 2,
    "remove_unused_columns": False,
    "report_to": "tensorboard",
}

# Create TrainingArguments
training_args = TrainingArguments(**training_config)

print("📋 Training Configuration:")
for key, value in training_config.items():
    print(f"   {key}: {value}")

# %%
# Cell 4.2: Calculate Training Estimates

print_section("Training Estimates")

# Estimates based on configuration
num_samples = len(dataset_splits["train"])
batch_size = training_config["per_device_train_batch_size"]
grad_accum = training_config["gradient_accumulation_steps"]
epochs = training_config["num_train_epochs"]

effective_batch_size = batch_size * grad_accum
steps_per_epoch = num_samples // effective_batch_size
total_steps = steps_per_epoch * epochs

print(f"📊 Training Estimates:")
print(f"   Training samples: {num_samples}")
print(f"   Effective batch size: {effective_batch_size}")
print(f"   Steps per epoch: {steps_per_epoch}")
print(f"   Total steps: {total_steps}")
print(f"   Checkpoints: ~{total_steps // training_config['save_steps']}")

# Time estimate (rough)
# Assuming ~1 second per step on A100
estimated_time_seconds = total_steps * 1.0
print(f"\n⏱️ Estimated training time: {format_duration(estimated_time_seconds)}")
print("   (Actual time depends on hardware and data)")

# %% [markdown]
# ---
# ## Part 5: Custom Training Loop
#
# สำหรับ TTS model อาจต้องใช้ custom training loop
# เพื่อจัดการกับ audio encoding/decoding

# %%
# Cell 5.1: Custom Data Collator

class TTSDataCollator:
    """Custom data collator for TTS training"""
    
    def __init__(self, tokenizer, audio_tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.max_length = max_length
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of features"""
        
        texts = [f["text"] for f in features]
        
        # Tokenize text
        text_encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        batch = {
            "input_ids": text_encodings["input_ids"],
            "attention_mask": text_encodings["attention_mask"],
        }
        
        # Process audio if available
        if "audio" in features[0] and features[0]["audio"] is not None:
            audio_arrays = [f["audio"]["array"] for f in features]
            # Encode audio to tokens
            # audio_tokens = self.audio_tokenizer.encode(audio_arrays)
            # batch["labels"] = audio_tokens
        
        return batch

print("✅ TTSDataCollator defined")

# %%
# Cell 5.2: Custom Trainer Class

class TTSTrainer(Trainer):
    """Custom trainer for TTS models"""
    
    def __init__(self, audio_tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.audio_tokenizer = audio_tokenizer
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation for TTS
        Combines text-to-audio prediction loss
        """
        # Standard forward pass
        outputs = model(**inputs)
        
        # TTS specific loss computation
        # This would include:
        # 1. Audio token prediction loss
        # 2. Duration prediction loss (optional)
        # 3. Pitch prediction loss (optional)
        
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(self, *args, **kwargs):
        """Custom evaluation with audio generation"""
        output = super().evaluation_loop(*args, **kwargs)
        
        # Add custom metrics
        # - MOS (Mean Opinion Score) estimation
        # - Speaker similarity
        # - Intelligibility metrics
        
        return output

print("✅ TTSTrainer class defined")

# %%
# Cell 5.3: Training Function

def train_tts_model(
    model,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: TrainingArguments,
    audio_tokenizer=None,
):
    """
    Main training function for TTS model
    """
    print_section("Starting Training")
    
    # Initialize trainer
    trainer = TTSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        audio_tokenizer=audio_tokenizer,
    )
    
    # Training info
    print(f"📋 Training Info:")
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Eval samples: {len(eval_dataset)}")
    print(f"   Output dir: {training_args.output_dir}")
    
    # Start training
    print("\n🚀 Starting training...")
    train_result = trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model()
    
    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    print("\n✅ Training complete!")
    return trainer, metrics

print("✅ Training function defined")

# %% [markdown]
# ---
# ## Part 6: Run Training (Demo)
#
# ⚠️ **Note:** This section demonstrates the training workflow.
# For actual training, you need:
# 1. Real audio data (several hours)
# 2. Proper GPU resources (24GB+ VRAM)
# 3. Qwen3-TTS specific training code

# %%
# Cell 6.1: Training Demo (Dry Run)

print_section("Training Demo (Dry Run)")

print("⚠️ This is a demonstration of the training workflow")
print("   Actual training requires real audio data and sufficient GPU memory")
print("\n📋 What would happen in real training:")
print("   1. Load audio files and transcriptions")
print("   2. Encode audio to discrete tokens")
print("   3. Train model to predict audio tokens from text")
print("   4. Save checkpoints periodically")
print("   5. Evaluate on validation set")

# Simulate training steps
print("\n🔄 Simulated Training Progress:")
for epoch in range(1, 4):
    print(f"\n   Epoch {epoch}/3:")
    for step in range(0, 100, 25):
        loss = 5.0 - (epoch * 0.5 + step * 0.01)  # Simulated decreasing loss
        print(f"      Step {step:3d} | Loss: {loss:.4f}")

print("\n✅ Training simulation complete")

# %%
# Cell 6.2: Monitor Training (TensorBoard)

print_section("Training Monitoring")

print("📊 To monitor training with TensorBoard:")
print(f"\n   tensorboard --logdir={LOG_DIR}")
print("\n   Then open: http://localhost:6006")
print("\n📋 Metrics to monitor:")
print("   - Training loss")
print("   - Validation loss")
print("   - Learning rate")
print("   - GPU memory usage")

# %% [markdown]
# ---
# ## Part 7: Model Evaluation
#
# ประเมินผล model หลังการ train

# %%
# Cell 7.1: Evaluation Metrics

class TTSEvaluator:
    """Evaluation utilities for TTS models"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_samples(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """Generate audio samples from text"""
        samples = []
        for text in texts:
            wav, sr = self.model.generate(text, **kwargs)
            samples.append(wav)
        return samples
    
    def compute_rtf(self, text: str, num_runs: int = 5) -> float:
        """
        Compute Real-Time Factor (RTF)
        RTF < 1 means faster than real-time
        """
        import time
        
        total_time = 0
        total_duration = 0
        
        for _ in range(num_runs):
            start = time.time()
            wav, sr = self.model.generate(text)
            elapsed = time.time() - start
            
            audio_duration = len(wav) / sr
            total_time += elapsed
            total_duration += audio_duration
        
        rtf = total_time / total_duration
        return rtf
    
    def speaker_similarity(self, ref_audio: np.ndarray, 
                          gen_audio: np.ndarray) -> float:
        """
        Compute speaker similarity score
        (Requires speaker embedding model)
        """
        # Placeholder - would use speaker verification model
        return 0.85
    
    def intelligibility_score(self, text: str, audio: np.ndarray) -> float:
        """
        Compute intelligibility score using ASR
        (Requires ASR model)
        """
        # Placeholder - would use ASR model
        return 0.92

print("✅ TTSEvaluator class defined")

# %%
# Cell 7.2: Run Evaluation

print_section("Model Evaluation")

# Test texts for evaluation
eval_texts = [
    "Hello, this is a test of the text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning enables computers to learn from data.",
]

print("📋 Evaluation Tests:")
print("\n1. Generation Quality:")
for i, text in enumerate(eval_texts, 1):
    print(f"   Test {i}: {text[:50]}...")

print("\n2. Real-Time Factor (RTF):")
print("   RTF measures inference speed relative to audio duration")
print("   RTF < 1.0 = faster than real-time")
print("   Target: RTF < 0.5 for production")

print("\n3. Speaker Consistency:")
print("   Measures how consistent the voice is across generations")

print("\n4. Intelligibility:")
print("   Measures how well the speech can be understood (ASR-based)")

# %% [markdown]
# ---
# ## Part 8: Export and Deploy
#
# Export trained model สำหรับ deployment

# %%
# Cell 8.1: Save Trained Model

def save_trained_model(model, output_path: Path, config: dict = None):
    """Save trained model and configuration"""
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = output_path / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved model weights: {model_path}")
    
    # Save configuration
    if config:
        config_path = output_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Saved config: {config_path}")
    
    # Save LoRA adapters (if applicable)
    # adapter_path = output_path / "adapter"
    # model.save_pretrained(adapter_path)
    
    return output_path

print("✅ Save function defined")

# %%
# Cell 8.2: Export for Inference

def export_for_inference(model_path: Path, export_path: Path):
    """Export model optimized for inference"""
    
    export_path.mkdir(parents=True, exist_ok=True)
    
    print("📋 Export options:")
    print("\n1. PyTorch (standard):")
    print("   - Full precision (FP32)")
    print("   - Half precision (FP16/BF16)")
    
    print("\n2. ONNX Export:")
    print("   - Cross-platform compatibility")
    print("   - Optimized inference")
    
    print("\n3. TensorRT (NVIDIA):")
    print("   - Maximum GPU performance")
    print("   - Requires NVIDIA GPU")
    
    print("\n4. Quantization:")
    print("   - INT8 for smaller size")
    print("   - Faster inference on CPU")

print("✅ Export options defined")

# %%
# Cell 8.3: Inference Script Template

inference_script = '''
"""
Inference script for fine-tuned Qwen3-TTS model
"""
import torch
from qwen_tts import Qwen3TTSModel
import soundfile as sf

class FineTunedTTS:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        
        # Load base model
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map=device,
        )
        
        # Load fine-tuned weights
        state_dict = torch.load(f"{model_path}/model.pt")
        self.model.load_state_dict(state_dict, strict=False)
        
        self.model.eval()
    
    def generate(self, text: str, output_path: str = None):
        """Generate speech from text"""
        with torch.no_grad():
            wav, sr = self.model.generate(text)
        
        if output_path:
            sf.write(output_path, wav, sr)
        
        return wav, sr

# Usage
if __name__ == "__main__":
    tts = FineTunedTTS("./checkpoints/final")
    wav, sr = tts.generate(
        "Hello, this is my fine-tuned voice!",
        output_path="output.wav"
    )
    print(f"Generated {len(wav)/sr:.2f} seconds of audio")
'''

# Save inference script
script_path = OUTPUT_DIR / "inference.py"
with open(script_path, 'w') as f:
    f.write(inference_script)

print(f"✅ Saved inference script: {script_path}")

# %% [markdown]
# ---
# ## Part 9: Summary and Best Practices
#
# ### Training Checklist
#
# | Step | Description | Status |
# |------|-------------|--------|
# | 1 | Collect high-quality audio | ⬜ |
# | 2 | Transcribe accurately | ⬜ |
# | 3 | Preprocess and normalize | ⬜ |
# | 4 | Configure training | ⬜ |
# | 5 | Monitor training | ⬜ |
# | 6 | Evaluate results | ⬜ |
# | 7 | Export model | ⬜ |
#
# ### Best Practices
#
# 1. **Data Quality**
#    - Use clean, noise-free recordings
#    - Consistent microphone and environment
#    - Accurate transcriptions
#
# 2. **Training**
#    - Start with small learning rate
#    - Use gradient clipping
#    - Monitor for overfitting
#
# 3. **Evaluation**
#    - Listen to generated samples
#    - Compare with reference audio
#    - Test on diverse texts

# %%
# Cell 9.1: Final Summary

print_section("Training Lab Summary")

print("📚 What we covered:")
print("   1. Dataset preparation for TTS")
print("   2. Audio processing pipeline")
print("   3. LoRA configuration for efficient fine-tuning")
print("   4. Training hyperparameters")
print("   5. Custom training loop for TTS")
print("   6. Model evaluation metrics")
print("   7. Export and deployment")

print("\n📁 Generated Files:")
for f in OUTPUT_DIR.glob("*"):
    print(f"   {f.name}")

print("\n🔗 Resources:")
print("   - Qwen3-TTS GitHub: https://github.com/QwenLM/Qwen3-TTS")
print("   - HuggingFace PEFT: https://huggingface.co/docs/peft")
print("   - Training Tips: https://huggingface.co/docs/transformers/training")

print_gpu_memory()

# %%
# Cell 9.2: Cleanup

def cleanup():
    """Clean up resources"""
    global base_model
    
    if 'base_model' in globals():
        del base_model
        print("✅ Deleted base model")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ Cleared GPU cache")

# Uncomment to cleanup
# cleanup()

print("To free GPU memory, run: cleanup()")

# %% [markdown]
# ---
# ## 📝 Lab Exercises
#
# ### Exercise 1: Data Preparation
# สร้าง dataset จากไฟล์เสียงของคุณเอง:
# - บันทึกเสียงอ่าน 10-20 ประโยค
# - สร้าง metadata.json
# - Process audio files
#
# ### Exercise 2: Hyperparameter Tuning
# ทดลองปรับค่า hyperparameters:
# - เปลี่ยน LoRA rank (r = 8, 16, 32, 64)
# - ปรับ learning rate
# - เปลี่ยน batch size
#
# ### Exercise 3: Evaluation
# สร้าง evaluation pipeline:
# - Generate samples จาก test set
# - คำนวณ RTF
# - เปรียบเทียบกับ base model
#
# ### Exercise 4: Deployment
# สร้าง inference API:
# - FastAPI endpoint
# - Streaming audio response
# - Error handling

# %%
# Exercise Space

print_section("Exercise Space")

# Your code here!
print("Write your exercise solutions here")

# %% [markdown]
# ---
# ## 📚 Additional Resources
#
# - **Qwen3-TTS Paper**: https://arxiv.org/abs/2601.15621
# - **LoRA Paper**: https://arxiv.org/abs/2106.09685
# - **HuggingFace Transformers**: https://huggingface.co/docs/transformers
# - **PEFT Documentation**: https://huggingface.co/docs/peft
# - **Audio Processing**: https://librosa.org/doc/latest/
#
# ---
#
# **End of Lab**
