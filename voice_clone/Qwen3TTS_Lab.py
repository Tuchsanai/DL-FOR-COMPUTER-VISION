# %% [markdown]
# # 🎙️ Qwen3-TTS Lab: Design Custom Voices with Text Descriptions
#
# **Course:** Machine Learning / Deep Learning  
# **Topic:** Text-to-Speech with Voice Design  
# **Model:** Qwen3-TTS (Alibaba, January 2025)
#
# ---
#
# ## Learning Objectives
#
# By the end of this lab, you will be able to:
#
# 1. Understand the Qwen3-TTS architecture and model variants
# 2. Set up the environment for TTS generation
# 3. Create custom voices using natural language descriptions
# 4. Use pre-built premium voices with emotion control
# 5. Clone voices from reference audio samples
# 6. Combine voice design and cloning for reusable characters
# 7. Use the speech tokenizer for audio encoding/decoding
#
# ---
#
# ## What is Qwen3-TTS?
#
# **Qwen3-TTS** is a state-of-the-art text-to-speech model released by Alibaba's Qwen team 
# in January 2025. It represents a major advancement in controllable speech synthesis.
#
# ### Key Capabilities:
#
# | Feature | Description |
# |---------|-------------|
# | **Voice Design** | Create new voices from text descriptions |
# | **Voice Cloning** | Clone any voice from 3-second audio |
# | **10 Languages** | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian |
# | **Emotion Control** | Natural language control over tone and emotion |
# | **Streaming** | Ultra-low latency (97ms first packet) |
# | **Open Source** | Apache 2.0 license |

# %% [markdown]
# ---
# ## Part 1: Environment Setup
#
# First, let's install the required packages and verify our system configuration.

# %%
# Cell 1.1: Install Required Packages
# Run this cell if packages are not installed
# Uncomment to install:
# !uv pip install -U transformers==4.57.3 numba qwen-tts soundfile 
# For better performance with FlashAttention 2 (requires compatible GPU):
# #!pip install -U flash-attn --no-build-isolation

# %%
# Reinstall torchvision ให้ match กับ PyTorch + CUDA 12.8
# !uv pip uninstall torchvision 
# !uv pip install torchvision --index-url https://download.pytorch.org/whl/cu128

# %%
# Cell 1.2: Import Libraries and Check System

import torch
import soundfile as sf
import os
import numpy as np
from IPython.display import Audio, display, HTML

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

print_section("System Information")

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    
    # Check if enough memory for 1.7B model
    if gpu_memory >= 8:
        print("✅ Sufficient GPU memory for 1.7B models")
    else:
        print("⚠️ Limited GPU memory - consider using 0.6B models")
else:
    print("⚠️ No CUDA GPU detected - will use CPU (slower)")

# %%
# Cell 1.3: Create Output Directory

OUTPUT_DIR = "./qwen3_tts_lab_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ Output directory: {OUTPUT_DIR}")

# Helper function to save and play audio
def save_and_play(wav, sr, filename, description=""):
    """Save audio file and display player"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    sf.write(filepath, wav, sr)
    print(f"\n📁 Saved: {filepath}")
    if description:
        print(f"   {description}")
    display(Audio(wav, rate=sr))
    return filepath

# %% [markdown]
# ---
# ## Part 2: Understanding the Model Family
#
# Qwen3-TTS provides several model variants for different use cases:
#
# | Model | Size | Purpose | Use Case |
# |-------|------|---------|----------|
# | **VoiceDesign** | 1.7B | Create voices from descriptions | Character creation, custom personas |
# | **CustomVoice** | 0.6B/1.7B | Pre-built premium voices | Quick high-quality TTS |
# | **Base** | 0.6B/1.7B | Voice cloning | Clone real voices |
# | **Tokenizer** | - | Audio encoding/decoding | Compression, analysis |
#
# ### Model Selection Guide:
#
# - **1.7B models**: Higher quality, requires ~8GB VRAM
# - **0.6B models**: Faster, requires ~4GB VRAM
# - **12Hz tokenizer**: Better compression, newer architecture

# %% [markdown]
# ---
# ## Part 3: Voice Design - Creating Voices from Descriptions
#
# The **VoiceDesign** model is the most exciting feature - it allows you to create 
# entirely new voices just by describing them in natural language!
#
# ### How Voice Design Works:
#
# 1. You provide a **text** to be spoken
# 2. You provide an **instruction** describing the voice characteristics
# 3. The model generates speech with a voice matching your description
#
# ### Effective Voice Descriptions Include:
#
# - **Demographics**: Age, gender
# - **Voice Quality**: Deep, bright, raspy, smooth, warm
# - **Emotion/Tone**: Cheerful, serious, calm, excited
# - **Speaking Style**: Fast, slow, measured, energetic
# - **Context**: News anchor, storyteller, teacher, character type
#
# ### ⚠️ Language Support Note / หมายเหตุเรื่องภาษา:
#
# **Qwen3-TTS รองรับ 10 ภาษา:**
# - Chinese, English, Japanese, Korean
# - German, French, Russian, Portuguese, Spanish, Italian
#
# **ภาษาไทย (Thai) ยังไม่รองรับอย่างเป็นทางการ** - แต่เราจะทดลองใช้งานเพื่อดูผลลัพธ์
# และเปรียบเทียบกับภาษาที่รองรับ
#
# ### Supported Languages Parameter:
# ```python
# language = "English"   # ✅ Supported
# language = "Chinese"   # ✅ Supported  
# language = "Japanese"  # ✅ Supported
# language = "Korean"    # ✅ Supported
# # Thai ไม่มีใน official list แต่สามารถทดลองใช้ "Chinese" หรือ "Korean" ได้
# ```

# %%
# Cell 3.1: Load the VoiceDesign Model

from qwen_tts import Qwen3TTSModel

print_section("Loading VoiceDesign Model")
print("This may take a few minutes on first run (downloading weights)...")

# Model configuration
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

voice_design_model = Qwen3TTSModel.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE,
    dtype=DTYPE,
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
)

print(f"\n✅ Model loaded successfully!")
print(f"   Model: {MODEL_NAME}")
print(f"   Device: {DEVICE}")
print(f"   Dtype: {DTYPE}")

# %% [markdown]
# ### 3.2 Basic Examples - English Voice Design
#
# เริ่มต้นด้วยตัวอย่างพื้นฐานภาษาอังกฤษก่อน

# %%
# Cell 3.2: Example 1 - Professional News Anchor (English)

print_section("Example 1: Professional News Anchor (English)")

text_en = "Good evening. Tonight's top story: Scientists have made a groundbreaking discovery in renewable energy that could revolutionize how we power our cities."

instruction_news = """
A professional male news anchor in his 40s with a deep, authoritative voice. 
Clear articulation, measured pace, and confident delivery. 
The kind of voice you'd hear on a major evening news broadcast.
"""

print(f"🇺🇸 Text (English): {text_en[:80]}...")
print(f"\n📝 Voice Description:\n{instruction_news}")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_en,
    language="English",
    instruct=instruction_news,
)

save_and_play(wavs[0], sr, "01_news_anchor_en.wav", "Professional news anchor voice (English)")

# %%
# Cell 3.3: Example 2 - Warm Storyteller (English)

print_section("Example 2: Warm Storyteller / Narrator (English)")

text_en = "Once upon a time, in a small village nestled between rolling hills and ancient forests, there lived a young girl with a curious mind and an adventurous spirit."

instruction_storyteller = """
A warm, gentle female storyteller voice, like a grandmother telling bedtime stories.
Soft and soothing with natural pauses. Speaks slowly and deliberately, 
drawing listeners into the narrative. Voice carries wisdom and kindness.
"""

print(f"🇺🇸 Text (English): {text_en[:80]}...")
print(f"\n📝 Voice Description:\n{instruction_storyteller}")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_en,
    language="English",
    instruct=instruction_storyteller,
)

save_and_play(wavs[0], sr, "02_storyteller_en.wav", "Warm storytelling voice (English)")

# %% [markdown]
# ### 3.3 Thai Language Experiments 🇹🇭
#
# ทดลองใช้ข้อความภาษาไทย - แม้ภาษาไทยจะไม่ได้รับการรองรับอย่างเป็นทางการ
# แต่เราสามารถทดลองดูผลลัพธ์ได้
#
# **สิ่งที่ควรสังเกต:**
# - การออกเสียงวรรณยุกต์
# - การแบ่งคำและวรรค
# - ความชัดเจนของเสียงพยัญชนะและสระ

# %%
# Cell 3.4: Thai Storyteller Voice (Experimental)

print_section("Example 3: Thai Storyteller 🇹🇭 (Experimental)")

text_th = """กาลครั้งหนึ่งนานมาแล้ว ในหมู่บ้านเล็กๆ ที่ซ่อนตัวอยู่ระหว่างเนินเขาและป่าโบราณ 
มีเด็กหญิงตัวน้อยคนหนึ่ง เธอมีความอยากรู้อยากเห็นและหัวใจที่รักการผจญภัย"""

instruction_th_storyteller = """
A warm, gentle Thai female storyteller voice, like a grandmother telling bedtime stories.
Soft and soothing with natural pauses. Speaks slowly and deliberately.
Voice carries wisdom and kindness. Clear Thai pronunciation.
"""

print(f"🇹🇭 ข้อความ (Thai): {text_th[:60]}...")
print(f"\n📝 คำอธิบายเสียง:\n{instruction_th_storyteller}")

# ทดลองใช้ language="Chinese" เพราะมีโครงสร้างเสียงคล้ายภาษาไทยมากกว่า
print("\n⚠️ Note: Testing with language='Chinese' as Thai is not officially supported")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_th,
    language="Chinese",  # ทดลองใช้ Chinese
    instruct=instruction_th_storyteller,
)

save_and_play(wavs[0], sr, "03_storyteller_th.wav", "Thai storyteller voice (Experimental)")

# %%
# Cell 3.5: Thai News Anchor (Experimental)

print_section("Example 4: Thai News Anchor 🇹🇭 (Experimental)")

text_th_news = """สวัสดีครับ ข่าวเด่นวันนี้ นักวิทยาศาสตร์ค้นพบวิธีการใหม่ในการผลิตพลังงานสะอาด 
ที่อาจเปลี่ยนแปลงอนาคตของประเทศไทยและโลกใบนี้"""

instruction_th_news = """
A professional Thai male news anchor in his 40s with a clear, authoritative voice.
Speaks with confidence and measured pace. Clear pronunciation of Thai tones.
Professional broadcast quality like Thai evening news.
"""

print(f"🇹🇭 ข้อความ (Thai): {text_th_news[:60]}...")
print(f"\n📝 คำอธิบายเสียง:\n{instruction_th_news}")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_th_news,
    language="Chinese",  # ทดลองใช้ Chinese
    instruct=instruction_th_news,
)

save_and_play(wavs[0], sr, "04_news_anchor_th.wav", "Thai news anchor voice (Experimental)")

# %%
# Cell 3.6: Bilingual Comparison - Same Voice, Two Languages

print_section("Example 5: Bilingual Comparison 🇺🇸🇹🇭")
print("เปรียบเทียบเสียงเดียวกัน พูดสองภาษา\n")

# ใช้ voice description เดียวกัน
instruction_bilingual = """
A friendly young female teacher in her late 20s. Clear, warm voice with 
encouraging tone. Speaks at moderate pace with good articulation.
Perfect for educational content.
"""

# English version
text_en_edu = "Welcome to today's machine learning class. We will learn about neural networks and how they work."

# Thai version  
text_th_edu = "ยินดีต้อนรับสู่ชั้นเรียน Machine Learning วันนี้ เราจะเรียนรู้เกี่ยวกับ Neural Networks และวิธีการทำงาน"

print(f"📝 Voice Description (same for both):\n{instruction_bilingual}")

# Generate English
print("\n🇺🇸 English Version:")
print(f"   Text: {text_en_edu}")

wavs_en, sr = voice_design_model.generate_voice_design(
    text=text_en_edu,
    language="English",
    instruct=instruction_bilingual,
)
save_and_play(wavs_en[0], sr, "05_bilingual_en.wav", "Teacher voice - English")

# Generate Thai
print("\n🇹🇭 Thai Version:")
print(f"   Text: {text_th_edu}")

wavs_th, sr = voice_design_model.generate_voice_design(
    text=text_th_edu,
    language="Chinese",  # ทดลอง
    instruct=instruction_bilingual,
)
save_and_play(wavs_th[0], sr, "05_bilingual_th.wav", "Teacher voice - Thai (Experimental)")

# %% [markdown]
# ### 3.4 Voice Design - Emotional Variations
#
# ทดลองสร้างเสียงที่มีอารมณ์แตกต่างกัน

# %%
# Cell 3.7: Energetic YouTuber (English)

print_section("Example 6: Energetic YouTuber 🎬")

text_yt = "Hey everyone! Welcome back to the channel! Today we're diving into something absolutely incredible - you're not gonna believe what we discovered!"

instruction_youtuber = """
An energetic young male YouTuber in his early 20s. Enthusiastic and slightly fast-paced.
Expressive intonation with excitement peaks. Natural, conversational style like 
talking to friends. High energy but genuine, not forced.
"""

print(f"🇺🇸 Text: {text_yt[:80]}...")
print(f"\n📝 Voice Description:\n{instruction_youtuber}")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_yt,
    language="English",
    instruct=instruction_youtuber,
)

save_and_play(wavs[0], sr, "06_youtuber_en.wav", "Energetic YouTuber voice")

# %%
# Cell 3.8: Thai YouTuber (Experimental)

print_section("Example 7: Thai YouTuber 🇹🇭🎬 (Experimental)")

text_yt_th = """สวัสดีครับทุกคน! ยินดีต้อนรับกลับมาที่ช่องอีกครั้ง! 
วันนี้เรามีเรื่องสุดเจ๋งมาเล่าให้ฟัง พวกคุณจะไม่เชื่อสิ่งที่เราค้นพบแน่นอน!"""

instruction_yt_th = """
An energetic young Thai male YouTuber in his early 20s. Enthusiastic and fast-paced.
Very expressive with excitement. Natural Thai speaking style.
High energy like popular Thai tech YouTubers.
"""

print(f"🇹🇭 ข้อความ: {text_yt_th[:60]}...")
print(f"\n📝 คำอธิบายเสียง:\n{instruction_yt_th}")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_yt_th,
    language="Chinese",
    instruct=instruction_yt_th,
)

save_and_play(wavs[0], sr, "07_youtuber_th.wav", "Thai YouTuber voice (Experimental)")

# %%
# Cell 3.9: Calm Meditation Guide (English)

print_section("Example 8: Calm Meditation Guide 🧘")

text_meditation = "Close your eyes. Take a deep breath in... and slowly release. Feel the tension leaving your body with each exhale. You are safe. You are calm."

instruction_meditation = """
A serene, peaceful female voice for guided meditation. Very slow, measured pace
with intentional pauses between phrases. Soft, almost whispered tone. 
Each word flows gently like a calm stream. Deeply relaxing and tranquil.
"""

print(f"🇺🇸 Text: {text_meditation[:80]}...")
print(f"\n📝 Voice Description:\n{instruction_meditation}")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_meditation,
    language="English",
    instruct=instruction_meditation,
)

save_and_play(wavs[0], sr, "08_meditation_en.wav", "Calm meditation guide voice")

# %%
# Cell 3.10: Thai Meditation Guide (Experimental)

print_section("Example 9: Thai Meditation Guide 🇹🇭🧘 (Experimental)")

text_meditation_th = """หลับตาลง... หายใจเข้าลึกๆ... แล้วค่อยๆ ปล่อยลมหายใจออก 
รู้สึกถึงความตึงเครียดที่ค่อยๆ ละลายหายไปทุกครั้งที่หายใจออก 
คุณปลอดภัย... คุณสงบ..."""

instruction_meditation_th = """
A serene, peaceful Thai female voice for guided meditation. 
Very slow pace with long pauses. Soft, gentle, almost whispered tone.
Deeply relaxing. Clear Thai pronunciation with calming intonation.
"""

print(f"🇹🇭 ข้อความ: {text_meditation_th[:60]}...")
print(f"\n📝 คำอธิบายเสียง:\n{instruction_meditation_th}")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_meditation_th,
    language="Korean",  # ทดลอง Korean ดูบ้าง
    instruct=instruction_meditation_th,
)

save_and_play(wavs[0], sr, "09_meditation_th.wav", "Thai meditation guide (Experimental)")

# %% [markdown]
# ### 3.5 Character Voices - Creative Examples
#
# สร้างเสียงตัวละครแบบต่างๆ

# %%
# Cell 3.11: Nature Documentary Narrator

print_section("Example 10: Nature Documentary Narrator 🎬🦁")

text_documentary = "Deep in the heart of the Amazon rainforest, where sunlight barely reaches the forest floor, a remarkable creature emerges from the shadows."

instruction_documentary = """
A mature, authoritative male voice perfect for nature documentaries.
Deep, resonant baritone with gravitas. Measured pacing that builds anticipation.
Like David Attenborough - conveys wonder and respect for nature.
Rich, warm timbre that commands attention.
"""

print(f"🇺🇸 Text: {text_documentary[:80]}...")
print(f"\n📝 Voice Description:\n{instruction_documentary}")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_documentary,
    language="English",
    instruct=instruction_documentary,
)

save_and_play(wavs[0], sr, "10_documentary_en.wav", "Documentary narrator voice")

# %%
# Cell 3.12: Friendly Robot AI

print_section("Example 11: Friendly Robot AI 🤖")

text_robot = "Greetings, human. I am your artificial assistant. How may I help you today? I am programmed to be helpful, harmless, and honest."

instruction_robot = """
A friendly robot AI assistant voice. Slightly synthetic quality but warm and approachable.
Precise articulation with subtle mechanical undertones. Not cold or menacing -
think helpful companion robot. Clear enunciation, moderate pace.
"""

print(f"🇺🇸 Text: {text_robot[:80]}...")
print(f"\n📝 Voice Description:\n{instruction_robot}")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_robot,
    language="English",
    instruct=instruction_robot,
)

save_and_play(wavs[0], sr, "11_robot_en.wav", "Friendly robot voice")

# %%
# Cell 3.13: Emotional Delivery - Panic

print_section("Example 12: Emotional Delivery - Panic 😰")

text_panic = "Wait, where is it? I put it right here! It was in this drawer, I'm absolutely certain! Oh no, no, no... this can't be happening!"

instruction_panic = """
A young woman in her 20s experiencing rising panic. Voice starts confused,
then escalates to genuine distress. Breathing becomes faster, pitch rises.
Natural emotional progression from confusion to panic. Authentic fear.
"""

print(f"🇺🇸 Text: {text_panic[:80]}...")
print(f"\n📝 Voice Description:\n{instruction_panic}")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_panic,
    language="English",
    instruct=instruction_panic,
)

save_and_play(wavs[0], sr, "12_panic_en.wav", "Panicked emotional voice")

# %%
# Cell 3.14: Thai Character - Wise Elder

print_section("Example 13: Thai Wise Elder 🇹🇭👴 (Experimental)")

text_elder_th = """ลูกเอ๋ย ความรู้นั้นสำคัญก็จริง แต่ปัญญาที่แท้จริงนั้นมาจากประสบการณ์ 
จงเรียนรู้จากความผิดพลาด แล้วเจ้าจะเติบโตขึ้น"""

instruction_elder_th = """
An elderly Thai male voice, like a wise grandfather giving life advice.
Slow, deliberate pace with natural pauses for reflection. 
Warm, kind tone with wisdom in every word. Gentle but carries weight.
Traditional Thai elder speaking style.
"""

print(f"🇹🇭 ข้อความ: {text_elder_th[:60]}...")
print(f"\n📝 คำอธิบายเสียง:\n{instruction_elder_th}")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_elder_th,
    language="Chinese",
    instruct=instruction_elder_th,
)

save_and_play(wavs[0], sr, "13_elder_th.wav", "Thai wise elder voice (Experimental)")

# %%
# Cell 3.15: Thai Customer Service

print_section("Example 14: Thai Customer Service 🇹🇭📞 (Experimental)")

text_service_th = """สวัสดีค่ะ ขอบคุณที่โทรหาศูนย์บริการลูกค้าค่ะ 
ดิฉันชื่อสมศรี ยินดีให้บริการค่ะ มีอะไรให้ดิฉันช่วยเหลือคะ?"""

instruction_service_th = """
A professional Thai female customer service representative.
Polite, warm, and helpful tone. Clear pronunciation with proper Thai 
polite particles (ค่ะ, คะ). Moderate pace, easy to understand.
Professional but friendly, like good Thai customer service.
"""

print(f"🇹🇭 ข้อความ: {text_service_th[:60]}...")
print(f"\n📝 คำอธิบายเสียง:\n{instruction_service_th}")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_service_th,
    language="Chinese",
    instruct=instruction_service_th,
)

save_and_play(wavs[0], sr, "14_customer_service_th.wav", "Thai customer service voice (Experimental)")

# %% [markdown]
# ### 3.6 Contrast Demo - Same Text, Different Voices
#
# การเปรียบเทียบเสียงที่แตกต่างกันด้วยข้อความเดียวกัน
# เพื่อให้เห็นว่า Voice Description มีผลต่อเสียงอย่างไร

# %%
# Cell 3.16: Contrast Demo - English

print_section("Example 15: Contrast Demo - Same Text, Different Voices 🎭")

text_contrast = "The results are in. We've achieved a fifteen percent increase in efficiency."

voices_contrast = [
    {
        "name": "Excited Startup CEO",
        "instruct": "An excited young male startup CEO announcing great news. Enthusiastic, slightly fast, building energy. Voice rises with excitement."
    },
    {
        "name": "Serious Corporate Executive",
        "instruct": "A serious middle-aged female corporate executive in a board meeting. Professional, measured, matter-of-fact. No emotion, just facts."
    },
    {
        "name": "Bored Employee",
        "instruct": "A tired, bored office worker reading numbers. Monotone, low energy, disinterested. Like it's the end of a long day."
    },
    {
        "name": "Nervous Intern",
        "instruct": "A nervous young intern presenting for the first time. Slightly shaky voice, uncertain pauses, lacks confidence but trying hard."
    }
]

print(f"📝 Text (same for all): \"{text_contrast}\"\n")
print("=" * 60)

for i, voice in enumerate(voices_contrast):
    print(f"\n🎤 Voice {i+1}: {voice['name']}")
    print(f"   Description: {voice['instruct'][:70]}...")
    
    wavs, sr = voice_design_model.generate_voice_design(
        text=text_contrast,
        language="English",
        instruct=voice['instruct'],
    )
    
    filename = f"15_contrast_{i+1}_{voice['name'].lower().replace(' ', '_')}.wav"
    save_and_play(wavs[0], sr, filename)

# %%
# Cell 3.17: Contrast Demo - Thai

print_section("Example 16: Contrast Demo - Thai 🇹🇭🎭 (Experimental)")

text_contrast_th = "ผลการทดสอบออกมาแล้วครับ เราสามารถเพิ่มประสิทธิภาพได้ถึงสิบห้าเปอร์เซ็นต์"

voices_contrast_th = [
    {
        "name": "Excited Manager",
        "instruct": "An excited Thai male manager announcing good news to his team. Very enthusiastic, fast-paced, voice full of energy and pride."
    },
    {
        "name": "Calm Scientist", 
        "instruct": "A calm Thai female scientist presenting research findings. Neutral, professional, measured pace. Just stating facts objectively."
    },
    {
        "name": "Tired Student",
        "instruct": "A tired Thai university student presenting a project. Low energy, monotone, clearly exhausted from studying all night."
    }
]

print(f"📝 ข้อความ (เหมือนกันทั้งหมด): \"{text_contrast_th}\"\n")
print("=" * 60)

for i, voice in enumerate(voices_contrast_th):
    print(f"\n🎤 เสียงที่ {i+1}: {voice['name']}")
    print(f"   คำอธิบาย: {voice['instruct'][:70]}...")
    
    wavs, sr = voice_design_model.generate_voice_design(
        text=text_contrast_th,
        language="Chinese",  # Experimental for Thai
        instruct=voice['instruct'],
    )
    
    filename = f"16_contrast_th_{i+1}_{voice['name'].lower().replace(' ', '_')}.wav"
    save_and_play(wavs[0], sr, filename)

# %% [markdown]
# ### 3.7 Technical Terminology Examples
#
# ทดลองกับข้อความที่มีคำศัพท์ทางเทคนิค - ทั้งภาษาอังกฤษและภาษาไทยผสม

# %%
# Cell 3.18: Technical Content - ML Course Introduction

print_section("Example 17: ML Course Introduction 📚")

# English version
text_ml_en = """Today we'll learn about Convolutional Neural Networks, or CNNs.
These networks are especially powerful for image recognition tasks.
We'll implement one using PyTorch and train it on the CIFAR-10 dataset."""

instruction_ml = """
A clear, articulate male university professor in his 30s teaching computer science.
Patient and educational tone. Speaks clearly and at moderate pace.
Enunciates technical terms carefully. Encouraging and approachable.
"""

print("🇺🇸 English Version:")
print(f"   Text: {text_ml_en[:80]}...")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_ml_en,
    language="English",
    instruct=instruction_ml,
)
save_and_play(wavs[0], sr, "17_ml_course_en.wav", "ML Course - English")

# %%
# Cell 3.19: Thai Technical Content with English Terms

print_section("Example 18: Thai-English Mixed Technical Content 🇹🇭💻")

text_ml_th = """วันนี้เราจะเรียนรู้เกี่ยวกับ Convolutional Neural Networks หรือ CNNs กันครับ
เครือข่ายประเภทนี้มีประสิทธิภาพมากสำหรับงาน Image Recognition
เราจะ implement ด้วย PyTorch และ train บน CIFAR-10 dataset"""

instruction_ml_th = """
A clear Thai male university professor teaching computer science.
Comfortable mixing Thai and English technical terms naturally.
Patient and educational. Moderate pace for student understanding.
"""

print("🇹🇭 Thai Version (with English terms):")
print(f"   ข้อความ: {text_ml_th[:80]}...")

wavs, sr = voice_design_model.generate_voice_design(
    text=text_ml_th,
    language="Chinese",  # Experimental
    instruct=instruction_ml_th,
)
save_and_play(wavs[0], sr, "18_ml_course_th.wav", "ML Course - Thai (Experimental)")

# %% [markdown]
# ### 3.8 Voice Description Best Practices Summary
#
# จากตัวอย่างทั้งหมด นี่คือหลักการเขียน Voice Description ที่ดี:
#
# | หมวดหมู่ | ตัวอย่างที่ดี ✅ | ตัวอย่างที่ไม่ดี ❌ |
# |---------|----------------|-------------------|
# | **อายุ/เพศ** | "A 45-year-old male" | "A person" |
# | **คุณภาพเสียง** | "Deep, resonant baritone" | "Nice voice" |
# | **อารมณ์** | "Excited with rising pitch" | "Happy" |
# | **บริบท** | "Like a BBC news anchor" | "Professional" |
# | **ความเร็ว** | "Slow, measured with pauses" | "Normal speed" |
# | **สไตล์** | "Speaks in short, punchy sentences" | "Good speaker" |
#
# ### Tips for Thai Content:
# 1. ระบุให้ชัดว่าต้องการ "clear Thai pronunciation"
# 2. ใส่บริบทที่คนไทยคุ้นเคย เช่น "like Thai news anchor"
# 3. ทดลองใช้ `language="Chinese"` หรือ `"Korean"` สำหรับภาษาไทย
# 4. สังเกตและบันทึกผลลัพธ์เพื่อเปรียบเทียบ

# %%
# Cell 3.20: Summary Statistics

print_section("Part 3 Summary - Generated Audio Files")

# List all files generated in Part 3
part3_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
part3_files.sort()

print(f"\n📁 Total files generated: {len(part3_files)}")
print("\n" + "-" * 60)

english_files = [f for f in part3_files if '_en' in f or 'contrast_' in f]
thai_files = [f for f in part3_files if '_th' in f]

print(f"\n🇺🇸 English examples: {len(english_files)}")
print(f"🇹🇭 Thai examples (experimental): {len(thai_files)}")

print("\n" + "-" * 60)
print("\nAll generated files:")
for f in part3_files:
    filepath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(filepath) / 1024
    print(f"  • {f:<50} {size:>6.1f} KB")

# %% [markdown]
# ---
# ## Part 4: CustomVoice - Pre-built Premium Speakers
#
# If you don't need to design a custom voice, Qwen3-TTS provides **9 pre-built premium voices**
# that are professionally crafted and ready to use.
#
# ### Available Speakers:
#
# | Speaker | Description | Native Language |
# |---------|-------------|-----------------|
# | **Vivian** | Bright, slightly edgy young female | Chinese |
# | **Serena** | Warm, gentle young female | Chinese |
# | **Uncle_Fu** | Seasoned male, low mellow timbre | Chinese |
# | **Dylan** | Youthful Beijing male, clear natural | Chinese (Beijing) |
# | **Eric** | Lively Chengdu male, slightly husky | Chinese (Sichuan) |
# | **Ryan** | Dynamic male, strong rhythmic drive | English |
# | **Aiden** | Sunny American male, clear midrange | English |
# | **Ono_Anna** | Playful Japanese female, light nimble | Japanese |
# | **Sohee** | Warm Korean female, rich emotion | Korean |
#
# ### Note: 
# Each speaker can speak **any supported language**, but sounds best in their native language.

# %%
# Cell 4.1: Load CustomVoice Model

print_section("Loading CustomVoice Model")

custom_voice_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map=DEVICE,
    dtype=DTYPE,
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
)

print("✅ CustomVoice Model loaded!")
print(f"\n📋 Supported Speakers: {custom_voice_model.get_supported_speakers()}")
print(f"📋 Supported Languages: {custom_voice_model.get_supported_languages()}")

# %%
# Cell 4.2: Basic CustomVoice - English Speakers

print_section("CustomVoice: English Speakers")

text = "Hello! Welcome to our artificial intelligence course. Today we'll explore the fascinating world of neural networks."

# Ryan - Dynamic male
print("\n🎤 Speaker: Ryan (Dynamic male, strong rhythmic drive)")
wavs, sr = custom_voice_model.generate_custom_voice(
    text=text,
    language="English",
    speaker="Ryan",
)
save_and_play(wavs[0], sr, "19_ryan_normal.wav", "Ryan - Normal")

# Aiden - Sunny American male  
print("\n🎤 Speaker: Aiden (Sunny American male, clear midrange)")
wavs, sr = custom_voice_model.generate_custom_voice(
    text=text,
    language="English",
    speaker="Aiden",
)
save_and_play(wavs[0], sr, "19_aiden_normal.wav", "Aiden - Normal")

# %%
# Cell 4.3: CustomVoice with Emotion Instructions

print_section("CustomVoice with Emotion Control")

text = "I can't believe it actually worked! After all these months of trying, we finally did it!"

emotions = [
    ("Normal", ""),
    ("Very excited", "Speak with extreme excitement and joy, like winning the lottery!"),
    ("Exhausted relief", "Speak with exhausted relief, like finally finishing a marathon."),
    ("Skeptical", "Speak with skepticism, like you're not quite sure it's real."),
]

print(f"Text: \"{text}\"\n")
print("Speaker: Ryan\n")

for emotion_name, instruction in emotions:
    print(f"🎭 Emotion: {emotion_name}")
    
    wavs, sr = custom_voice_model.generate_custom_voice(
        text=text,
        language="English",
        speaker="Ryan",
        instruct=instruction if instruction else None,
    )
    
    filename = f"20_ryan_{emotion_name.lower().replace(' ', '_')}.wav"
    save_and_play(wavs[0], sr, filename)

# %%
# Cell 4.4: Batch Generation with Multiple Speakers

print_section("Batch Generation: Multiple Speakers")

# Same text, different speakers
text = "Welcome to the international AI conference. We're excited to have participants from around the world."

speakers_to_demo = ["Ryan", "Aiden", "Vivian", "Ono_Anna"]

print(f"Text: \"{text}\"\n")

for speaker in speakers_to_demo:
    print(f"\n🎤 Speaker: {speaker}")
    
    wavs, sr = custom_voice_model.generate_custom_voice(
        text=text,
        language="English",  # All speakers can speak English
        speaker=speaker,
    )
    
    save_and_play(wavs[0], sr, f"21_batch_{speaker.lower()}.wav")

# %% [markdown]
# ---
# ## Part 5: Voice Cloning
#
# The **Base** model allows you to clone any voice from a short audio sample (3+ seconds).
# This is incredibly powerful for:
#
# - Creating consistent character voices for games/animations
# - Personalizing TTS with a specific voice
# - Audiobook narration matching original author's voice
#
# ### How Voice Cloning Works:
#
# 1. Provide a **reference audio** (3+ seconds)
# 2. Provide the **transcript** of the reference audio
# 3. Provide the **new text** you want spoken
# 4. The model generates the new text in the cloned voice

# %%
# Cell 5.1: Load Base Model for Cloning

print_section("Loading Base Model for Voice Cloning")

clone_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map=DEVICE,
    dtype=DTYPE,
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
)

print("✅ Base Model (Voice Cloning) loaded!")

# %%
# Cell 5.2: Basic Voice Cloning from URL

print_section("Voice Cloning Example")


# Reference audio from Qwen's repository
ref_audio_url = "lisa3.wav"
ref_text = "สวัสดีค่ะ วู้ดดี้ไทยแลนด์ ลิซ่านะคะ ตอนนี้พวกเราอยู่ที่ ALTER EGO POP-UP IN BANGKOK ค่ะ เครียด เหมือนมันมองไม่เห็นอนาคตอันใกล้หละ แค่แบบเดือนหน้า เราก็ไม่รู้ว่ามันจะออกมาเป็นรูปแบบไหน เราไม่มีคนที่จะถาม ปรึกษา เรามีแค่ทีมเรา แค่นั้น แต่มันไม่มีการันตีว่ามันจะโอเค มันจะเวิร์คใช่ไหม สรุปแล้ว"

# New text to generate with cloned voice
new_text = "ลิซ่านะคะ ทุกๆคน เสาร์นี้เราจะได้ทำการบ้านที่ สนุกสนาน และ ท้าทาย มาลุ้นกันดีกว่าการบ้านของวันเสาร์นี้คืออะไร"


print(f"Reference Audio: {ref_audio_url}")
print(f"Reference Text: \"{ref_text}\"")
print(f"\nNew Text to Generate: \"{new_text}\"")

wavs, sr = clone_model.generate_voice_clone(
    text=new_text,
    language="Chinese",
    ref_audio=ref_audio_url,
    ref_text=ref_text,
)

save_and_play(wavs[0], sr, "22_voice_clone_basic.wav", "Cloned voice speaking new text")

# %%
# Cell 5.3: Reusable Voice Clone Prompt

print_section("Creating Reusable Voice Clone Prompt")

print("Creating a voice prompt that can be reused for multiple generations...")
print("(This avoids re-computing features for each generation)\n")

# Create the prompt once
voice_prompt = clone_model.create_voice_clone_prompt(
    ref_audio=ref_audio_url,
    ref_text=ref_text,
)

print("✅ Voice prompt created!")

# Generate multiple sentences with the same voice
sentences = [
    "Chapter one. Introduction to Neural Networks.",
    "In the beginning, researchers tried to mimic the human brain.",
    "Today, we have models with billions of parameters.",
]

print("\n📖 Generating multiple sentences with reusable prompt:\n")

for i, sentence in enumerate(sentences):
    print(f"Sentence {i+1}: \"{sentence}\"")
    
    wavs, sr = clone_model.generate_voice_clone(
        text=sentence,
        language="Chinese",
        voice_clone_prompt=voice_prompt,  # Reuse the prompt
    )
    
    save_and_play(wavs[0], sr, f"23_clone_reuse_{i+1}.wav")

# %% [markdown]
# ### 5.4 Voice Cloning - Multiple Reference Scenarios
#
# Let's explore more advanced voice cloning scenarios including:
# - Cloning from different audio sources
# - Cross-language voice cloning
# - Emotional variation with cloned voices
# - Creating character voice variations

# %%
# Cell 5.5: Story Narration with Cloned Voice

print_section("Voice Cloning: Story Narration")

print("Creating a short story narration with consistent cloned voice\n")

story_segments = [
    "มีเด็กหญิงคนหนึ่ง เธอชอบเดินเล่นในป่าทุกวัน",
    "วันหนึ่ง เธอพบกระต่ายตัวน้อยที่บาดเจ็บ",
    "เธอจึงพากระต่ายกลับบ้านและดูแลมันอย่างดี",
    "หลังจากนั้นไม่นาน กระต่ายก็หายเป็นปกติ และกลายเป็นเพื่อนที่ดีที่สุดของเธอ",
]

print("📖 Story: The Girl and the Rabbit\n")

for i, segment in enumerate(story_segments):
    print(f"Part {i+1}: {segment}")
    
    wavs, sr = clone_model.generate_voice_clone(
        text=segment,
        language="Chinese",
        voice_clone_prompt=voice_prompt,
        instruct="Speak like a gentle storyteller, with warm tone and natural pauses.",
    )
    
    save_and_play(wavs[0], sr, f"29_story_part_{i+1}.wav")

# %%
# Cell 5.6: Educational Content with Cloned Voice

print_section("Voice Cloning: Educational Tutorial")

print("Creating an educational tutorial with cloned voice\n")

tutorial_steps = [
    "สวัสดีค่ะ วันนี้เราจะมาเรียนรู้เกี่ยวกับ Machine Learning กันนะคะ",
    "ขั้นตอนแรก เราต้องเตรียม dataset ให้พร้อม",
    "ขั้นตอนที่สอง เราจะทำการ train model ด้วย PyTorch",
    "ขั้นตอนสุดท้าย เราจะ evaluate ผลลัพธ์และปรับปรุง model",
    "ขอให้ทุกคนโชคดีกับการเรียนรู้นะคะ",
]

print("🎓 Tutorial: Machine Learning Basics\n")

for i, step in enumerate(tutorial_steps):
    print(f"Step {i+1}: {step}")
    
    wavs, sr = clone_model.generate_voice_clone(
        text=step,
        language="Chinese",
        voice_clone_prompt=voice_prompt,
        instruct="Speak like a clear, patient teacher explaining to students. Moderate pace, encouraging tone.",
    )
    
    save_and_play(wavs[0], sr, f"30_tutorial_step_{i+1}.wav")

# %%
# Cell 5.7: Podcast Introduction with Cloned Voice

print_section("Voice Cloning: Podcast Style")

print("Creating podcast segments with cloned voice\n")

podcast_segments = [
    {
        "type": "intro",
        "text": "สวัสดีค่ะทุกคน ยินดีต้อนรับสู่ Podcast เทคโนโลยีและนวัตกรรม",
        "instruction": "Speak with energy and enthusiasm like a podcast host welcoming listeners."
    },
    {
        "type": "topic",
        "text": "วันนี้เรามีหัวข้อน่าสนใจมาก เราจะมาคุยกันเรื่อง AI และอนาคตของมนุษยชาติ",
        "instruction": "Speak with curiosity and intrigue, building interest in the topic."
    },
    {
        "type": "question",
        "text": "คุณเคยสงสัยไหมคะว่า AI จะเปลี่ยนแปลงชีวิตเราอย่างไรในอีก 10 ปีข้างหน้า?",
        "instruction": "Ask the question thoughtfully, encouraging listeners to reflect."
    },
    {
        "type": "outro",
        "text": "ขอบคุณที่รับฟังนะคะ พบกันใหม่ใน Episode หน้า สวัสดีค่ะ",
        "instruction": "Speak warmly like saying goodbye to friends, with gratitude."
    },
]

print("🎙️ Podcast: Technology and Innovation\n")

for i, segment in enumerate(podcast_segments):
    print(f"\n{segment['type'].upper()}: {segment['text'][:60]}...")
    
    wavs, sr = clone_model.generate_voice_clone(
        text=segment['text'],
        language="Chinese",
        voice_clone_prompt=voice_prompt,
        instruct=segment['instruction'],
    )
    
    filename = f"31_podcast_{i+1}_{segment['type']}.wav"
    save_and_play(wavs[0], sr, filename)

# %%
# Cell 5.8: Character Dialogue with Multiple Cloned Voices

print_section("Voice Cloning: Multi-Character Dialogue")

print("Creating a dialogue between two characters using voice cloning\n")

# For this example, we'll create two different "versions" of the same voice
# In practice, you would clone two different reference voices

dialogue = [
    {
        "character": "Lisa (Excited)",
        "text": "เฮ้! คุณเห็นข่าวเมื่อเช้านี้ไหม? มันน่าตื่นเต้นมากเลย!",
        "instruction": "Speak with high energy and excitement, very enthusiastic!"
    },
    {
        "character": "Lisa (Curious)",
        "text": "ข่าวอะไรหรอ? เล่าหน่อยสิ ฉันอยากรู้จัง",
        "instruction": "Speak with curiosity and interest, asking questions naturally."
    },
    {
        "character": "Lisa (Excited)",
        "text": "เขาประกาศแล้วว่าจะมี AI ใหม่ที่สามารถเข้าใจอารมณ์มนุษย์ได้!",
        "instruction": "Speak with amazement and share exciting news enthusiastically."
    },
    {
        "character": "Lisa (Thoughtful)",
        "text": "โอ้ว... น่าสนใจจัง แต่มันจะปลอดภัยไหมนะ?",
        "instruction": "Speak thoughtfully and contemplatively, showing some concern."
    },
]

print("🎭 Dialogue: AI News Discussion\n")

for i, line in enumerate(dialogue):
    print(f"\n{line['character']}: {line['text']}")
    
    wavs, sr = clone_model.generate_voice_clone(
        text=line['text'],
        language="Chinese",
        voice_clone_prompt=voice_prompt,
        instruct=line['instruction'],
    )
    
    filename = f"32_dialogue_{i+1}_{line['character'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.wav"
    save_and_play(wavs[0], sr, filename)

# %%
# Cell 5.9: News Broadcast with Cloned Voice

print_section("Voice Cloning: News Broadcast Style")

print("Creating news segments with cloned voice\n")

news_segments = [
    {
        "type": "headline",
        "text": "ข่าวด่วน ประเทศไทยประสบความสำเร็จในการพัฒนา AI ภาษาไทย",
        "instruction": "Speak like a professional news anchor delivering breaking news, serious and authoritative."
    },
    {
        "type": "detail",
        "text": "นักวิจัยจากมหาวิทยาลัยชั้นนำได้พัฒนาระบบ AI ที่สามารถเข้าใจภาษาไทยได้อย่างแม่นยำ",
        "instruction": "Speak clearly and informatively, providing details professionally."
    },
    {
        "type": "quote",
        "text": "ผู้อำนวยการโครงการกล่าวว่า นี่เป็นก้าวสำคัญสำหรับประเทศไทย",
        "instruction": "Speak in reporting style, quoting someone else's words."
    },
    {
        "type": "closing",
        "text": "รายงานข่าวจาก ลิซ่า ขอบคุณครับ",
        "instruction": "Speak professionally like signing off from a news report."
    },
]

print("📺 News Report: Thai AI Development\n")

for i, segment in enumerate(news_segments):
    print(f"\n[{segment['type'].upper()}]")
    print(f"Text: {segment['text']}")
    
    wavs, sr = clone_model.generate_voice_clone(
        text=segment['text'],
        language="Chinese",
        voice_clone_prompt=voice_prompt,
        instruct=segment['instruction'],
    )
    
    filename = f"33_news_{i+1}_{segment['type']}.wav"
    save_and_play(wavs[0], sr, filename)

# %%
# Cell 5.10: Voice Cloning Speed Variations

print_section("Voice Cloning: Speed & Pace Variations")

print("Testing different speaking speeds with the same cloned voice\n")

test_text = "การเรียนรู้ Machine Learning นั้นสนุกและท้าทายในเวลาเดียวกัน"

speed_variations = [
    ("Very Slow", "Speak very slowly and deliberately, with long pauses between words."),
    ("Slow", "Speak slowly and carefully, giving time for understanding."),
    ("Normal", None),
    ("Fast", "Speak quickly and energetically, with rapid delivery."),
    ("Very Fast", "Speak very rapidly like in a fast-paced advertisement."),
]

print(f"📝 Test Text: \"{test_text}\"\n")

for speed_name, instruction in speed_variations:
    print(f"⏱️ Speed: {speed_name}")
    
    wavs, sr = clone_model.generate_voice_clone(
        text=test_text,
        language="Chinese",
        voice_clone_prompt=voice_prompt,
        instruct=instruction,
    )
    
    filename = f"34_speed_{speed_name.lower().replace(' ', '_')}.wav"
    save_and_play(wavs[0], sr, filename)

# %%
# Cell 5.11: Cloned Voice for Product Advertisement

print_section("Voice Cloning: Product Advertisement")

print("Creating advertisement copy with cloned voice\n")

ad_script = [
    {
        "segment": "hook",
        "text": "คุณกำลังมองหาคอร์สเรียน AI ที่ดีที่สุดอยู่ใช่ไหมคะ?",
        "instruction": "Speak with curiosity and engagement, asking a question that captures attention."
    },
    {
        "segment": "feature",
        "text": "เรามีคอร์สที่จะพาคุณไปจากเริ่มต้นจนถึงระดับ Expert",
        "instruction": "Speak with confidence and promise, highlighting the benefit."
    },
    {
        "segment": "benefit",
        "text": "เรียนรู้แบบ hands-on กับโปรเจคจริง ไม่ใช่แค่ทฤษฎี",
        "instruction": "Speak persuasively, emphasizing the unique value proposition."
    },
    {
        "segment": "cta",
        "text": "ลงทะเบียนวันนี้ รับส่วนลด 50% สำหรับ 100 คนแรกเท่านั้น!",
        "instruction": "Speak with urgency and excitement, encouraging immediate action!"
    },
]

print("📢 Advertisement: AI Course Promotion\n")

for i, segment in enumerate(ad_script):
    print(f"\n[{segment['segment'].upper()}]")
    print(f"{segment['text']}")
    
    wavs, sr = clone_model.generate_voice_clone(
        text=segment['text'],
        language="Chinese",
        voice_clone_prompt=voice_prompt,
        instruct=segment['instruction'],
    )
    
    filename = f"35_ad_{i+1}_{segment['segment']}.wav"
    save_and_play(wavs[0], sr, filename)

# %%
# Cell 5.12: Voice Clone Summary Statistics

print_section("Voice Cloning Summary")

# Count files generated in Part 5
clone_files = [f for f in os.listdir(OUTPUT_DIR) 
               if f.endswith('.wav') and any(x in f for x in ['28_', '29_', '30_', '31_', '32_', '33_', '34_', '35_'])]
clone_files.sort()

print(f"\n📊 Voice Cloning Examples Generated: {len(clone_files)}")
print("\nCategories:")
print(f"  🎭 Emotional Variations: {len([f for f in clone_files if '28_clone_emotion' in f])}")
print(f"  📖 Story Narration: {len([f for f in clone_files if '29_story' in f])}")
print(f"  🎓 Educational Content: {len([f for f in clone_files if '30_tutorial' in f])}")
print(f"  🎙️ Podcast Style: {len([f for f in clone_files if '31_podcast' in f])}")
print(f"  🎭 Character Dialogue: {len([f for f in clone_files if '32_dialogue' in f])}")
print(f"  📺 News Broadcast: {len([f for f in clone_files if '33_news' in f])}")
print(f"  ⏱️ Speed Variations: {len([f for f in clone_files if '34_speed' in f])}")
print(f"  📢 Advertisement: {len([f for f in clone_files if '35_ad' in f])}")

print("\n" + "=" * 70)
print("💡 Key Takeaways from Voice Cloning:")
print("=" * 70)
print("""
1. ✅ One voice can express many emotions through instructions
2. ✅ Cloned voices maintain consistency across long content
3. ✅ Speaking speed and style can be controlled via instructions
4. ✅ Same voice works for different contexts (education, news, ads)
5. ✅ Voice cloning is powerful for creating character consistency
""")

# %% [markdown]
# ### 5.13 Advanced Tips for Voice Cloning
#
# **Best Practices:**
#
# 1. **Reference Audio Quality**
#    - Use clean audio without background noise
#    - At least 3 seconds of continuous speech
#    - Clear pronunciation and good recording quality
#
# 2. **Reference Text Accuracy**
#    - Transcript must match the audio exactly
#    - Include punctuation for natural pauses
#    - Match the language of the audio
#
# 3. **Instruction Effectiveness**
#    - Be specific about emotion and style
#    - Reference concrete examples ("like a news anchor")
#    - Combine multiple attributes (pace + emotion + tone)
#
# 4. **Reusable Prompts**
#    - Create prompts once for multiple generations
#    - Saves computation time
#    - Ensures consistency across all outputs
#
# 5. **Language Mixing**
#    - For Thai content, experiment with `language="Chinese"` or `"Korean"`
#    - English instructions work even for non-English content
#    - Test different language parameters for best results

# %%
# Cell 5.13: Bonus - Batch Processing Multiple Texts

print_section("Bonus: Batch Voice Cloning")

print("Efficiently generating multiple audio files with the same cloned voice\n")

batch_texts = [
    "ข้อความที่หนึ่ง: AI กำลังเปลี่ยนแปลงโลก",
    "ข้อความที่สอง: Machine Learning เป็นอนาคตของเทคโนโลยี",
    "ข้อความที่สาม: Deep Learning ทำให้ AI ฉลาดขึ้น",
    "ข้อความที่สี่: Neural Networks เลียนแบบสมองมนุษย์",
    "ข้อความที่ห้า: ขอบคุณที่รับฟังนะคะ",
]

print(f"Generating {len(batch_texts)} audio files in batch mode...\n")

batch_instruction = "Speak clearly and professionally, like presenting to an audience."

for i, text in enumerate(batch_texts):
    print(f"[{i+1}/{len(batch_texts)}] {text[:50]}...")
    
    wavs, sr = clone_model.generate_voice_clone(
        text=text,
        language="Chinese",
        voice_clone_prompt=voice_prompt,
        instruct=batch_instruction,
    )
    
    save_and_play(wavs[0], sr, f"36_batch_{i+1}.wav")

print(f"\n✅ Batch processing complete! Generated {len(batch_texts)} files.")


# %% [markdown]
# ---
# ## Part 6: Advanced Workflow - Voice Design → Clone
#
# A powerful technique is to combine **Voice Design** and **Voice Cloning**:
#
# 1. **Design** a unique voice using natural language description
# 2. **Clone** that designed voice for consistent reuse
#
# This is perfect for creating consistent character voices from scratch!

# %%
# Cell 6.1: Create Character Voice with Design

print_section("Advanced: Voice Design → Clone Workflow")

print("Step 1: Design a character voice\n")

# Define the character
character_name = "Professor Maxwell"
character_ref_text = "Welcome, students. Today we embark on a journey through the fundamentals of quantum mechanics."
character_description = """
An elderly British male professor in his 60s. Distinguished, intellectual voice with 
a slight Oxford accent. Warm but authoritative. Speaks deliberately with natural 
academic cadence. Think of a beloved university professor who makes complex 
topics accessible.
"""

print(f"Character: {character_name}")
print(f"Description: {character_description}")

# Generate reference audio with VoiceDesign
ref_wavs, sr = voice_design_model.generate_voice_design(
    text=character_ref_text,
    language="English",
    instruct=character_description,
)

save_and_play(ref_wavs[0], sr, "24_character_reference.wav", f"{character_name} - Reference Audio")

# %%
# Cell 6.2: Create Clone Prompt from Designed Voice

print("\nStep 2: Create reusable clone prompt from designed voice\n")

character_prompt = clone_model.create_voice_clone_prompt(
    ref_audio=(ref_wavs[0], sr),  # Pass the numpy array directly
    ref_text=character_ref_text,
)

print("✅ Character clone prompt created!")

# %%
# Cell 6.3: Generate Multiple Lines as Character

print("\nStep 3: Generate dialogue lines with character voice\n")

lecture_lines = [
    "Now, let's consider Schrödinger's famous thought experiment with the cat.",
    "The beauty of quantum mechanics lies in its counterintuitive nature.",
    "Don't worry if this seems confusing at first. Even Einstein struggled with these concepts.",
    "For your homework, please read chapters three and four by next Tuesday.",
]

print(f"Generating {len(lecture_lines)} lines as {character_name}:\n")

for i, line in enumerate(lecture_lines):
    print(f"Line {i+1}: \"{line[:50]}...\"")
    
    wavs, sr = clone_model.generate_voice_clone(
        text=line,
        language="English",
        voice_clone_prompt=character_prompt,
    )
    
    save_and_play(wavs[0], sr, f"25_professor_line_{i+1}.wav")

# %% [markdown]
# ---
# ## Part 7: Speech Tokenizer
#
# The **Qwen3-TTS-Tokenizer** converts audio to/from discrete tokens.
# This is useful for:
#
# - **Audio compression**: Represent audio as compact token sequences
# - **Model training**: Prepare data for fine-tuning
# - **Audio analysis**: Study the acoustic representation
#
# ### Tokenizer Specifications:
#
# - **12Hz**: 12 tokens per second of audio
# - **Multi-codebook**: 16 layers of codes for high fidelity
# - **Efficient**: Extreme compression with good quality

# %%
# Cell 7.1: Load Tokenizer

from qwen_tts import Qwen3TTSTokenizer

print_section("Loading Speech Tokenizer")

tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device_map=DEVICE,
)

print("✅ Tokenizer loaded!")

# %%
# Cell 7.2: Encode and Decode Audio

print_section("Audio Encoding/Decoding")

# Use sample audio
sample_audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/tokenizer_demo_1.wav"

print(f"Original audio: {sample_audio_url}\n")

# Encode
print("Encoding audio to tokens...")
encoded = tokenizer.encode(sample_audio_url)
print(f"✅ Encoded shape: {encoded.shape}")
print(f"   Sequence length: {encoded.shape[1]} frames")
print(f"   At 12Hz = {encoded.shape[1]/12:.2f} seconds of audio")

# Decode
print("\nDecoding tokens back to audio...")
decoded_wavs, sr = tokenizer.decode(encoded)
print(f"✅ Decoded audio: {len(decoded_wavs[0])} samples at {sr}Hz")

save_and_play(decoded_wavs[0], sr, "26_tokenizer_roundtrip.wav", "Encoded → Decoded audio")

# %%
# Cell 7.3: Encode Generated Audio

print_section("Tokenize Our Generated Audio")

# Encode one of our generated files
generated_file = os.path.join(OUTPUT_DIR, "10_documentary_en.wav")

if os.path.exists(generated_file):
    print(f"Encoding: {generated_file}\n")
    
    # Encode
    encoded = tokenizer.encode(generated_file)
    print(f"Encoded shape: {encoded.shape}")
    print(f"Duration: {encoded.shape[1]/12:.2f} seconds")
    
    # Decode
    decoded_wavs, sr = tokenizer.decode(encoded)
    save_and_play(decoded_wavs[0], sr, "27_documentary_roundtrip.wav", "Documentary voice after encode/decode")
else:
    print(f"File not found: {generated_file}")
    print("Run the Voice Design examples first!")

# %% [markdown]
# ---
# ## Part 8: Summary and Best Practices
#
# ### Voice Description Tips:
#
# | Aspect | Good Example | Bad Example |
# |--------|--------------|-------------|
# | **Age/Gender** | "A 45-year-old male" | "A person" |
# | **Voice Quality** | "Deep, resonant baritone" | "Nice voice" |
# | **Emotion** | "Excited with rising pitch" | "Happy" |
# | **Context** | "Like a BBC news anchor" | "Professional" |
# | **Pace** | "Slow, measured with pauses" | "Normal speed" |
#
# ### Model Selection Guide:
#
# | Need | Model | Why |
# |------|-------|-----|
# | Create unique voices | VoiceDesign | Natural language → Voice |
# | Quick quality TTS | CustomVoice | Pre-built premium voices |
# | Match existing voice | Base | Clone from audio |
# | Multiple lines, same character | Design → Clone | Consistent reusable voice |
#
# ### Performance Tips:
#
# 1. **Use bfloat16** for GPU (saves memory, maintains quality)
# 2. **FlashAttention 2** for faster inference
# 3. **Batch processing** for multiple generations
# 4. **Reusable prompts** for voice cloning
#
# ### Thai Language Tips:
#
# 1. ภาษาไทยยังไม่ได้รับการรองรับอย่างเป็นทางการ
# 2. ทดลองใช้ `language="Chinese"` หรือ `"Korean"` 
# 3. ผลลัพธ์อาจไม่สมบูรณ์แบบ โดยเฉพาะวรรณยุกต์
# 4. รอติดตาม updates จาก Qwen team สำหรับการรองรับภาษาไทยในอนาคต

# %%
# Cell 8.1: Memory Usage Summary

print_section("Memory and File Summary")

if torch.cuda.is_available():
    print("GPU Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

print(f"\nGenerated Files in {OUTPUT_DIR}:")
print("-" * 60)

if os.path.exists(OUTPUT_DIR):
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')])
    total_size = 0
    for f in files:
        filepath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(filepath) / 1024
        total_size += size
        print(f"  {f:<50} {size:>8.1f} KB")
    print("-" * 60)
    print(f"  Total: {len(files)} files, {total_size/1024:.2f} MB")

# %%
# Cell 8.2: Cleanup Function (Optional)

def cleanup_models():
    """Free GPU memory by deleting models"""
    global voice_design_model, custom_voice_model, clone_model, tokenizer
    
    models_to_delete = [
        ('voice_design_model', 'VoiceDesign'),
        ('custom_voice_model', 'CustomVoice'),
        ('clone_model', 'Clone'),
        ('tokenizer', 'Tokenizer'),
    ]
    
    for var_name, display_name in models_to_delete:
        if var_name in globals():
            del globals()[var_name]
            print(f"✅ Deleted {display_name} model")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ GPU cache cleared")

# Uncomment to run cleanup:
# cleanup_models()

print("To free GPU memory, run: cleanup_models()")

# %% [markdown]
# ---
# ## 📝 Lab Exercises
#
# Complete the following exercises to practice what you've learned:
#
# ### Exercise 1: Design Your Own Voice (English)
# Create a unique voice for one of these scenarios:
# - A pirate captain
# - A sci-fi spaceship AI
# - A sports commentator
# - A children's cartoon character
#
# ### Exercise 2: Thai Voice Experiment 🇹🇭
# ทดลองสร้างเสียงภาษาไทยในบริบทต่างๆ:
# - พิธีกรรายการทีวี
# - ครูสอนภาษาไทย
# - ดีเจวิทยุ
# - พระสงฆ์เทศน์
#
# ### Exercise 3: Emotion Comparison
# Take one sentence and generate it with 5 different emotions using CustomVoice.
#
# ### Exercise 4: Character Dialogue (Bilingual)
# Create two different characters using VoiceDesign and generate a short dialogue 
# between them - one speaks English, one speaks Thai.
#
# ### Exercise 5: Clone and Extend
# Use the Voice Design → Clone workflow to create a consistent character voice
# and generate a 5-sentence monologue.
#
# ### Exercise 6: Audio Analysis
# Use the tokenizer to encode several different voice types and compare the token statistics.

# %%
# Exercise Space - Write your code here!

print_section("Exercise Space")

# Exercise 1: Pirate Captain
print("=" * 60)
print("Exercise 1: Create a pirate captain voice")
print("=" * 60)

# Uncomment and modify:
# pirate_text = "Arrr! All hands on deck! We've spotted treasure on the horizon!"
# pirate_description = """
# A gruff male pirate captain with a raspy, weathered voice.
# Speaks with enthusiasm and a slight growl. Commanding presence.
# Think of a classic movie pirate - boisterous and larger than life.
# """
# 
# wavs, sr = voice_design_model.generate_voice_design(
#     text=pirate_text,
#     language="English",
#     instruct=pirate_description,
# )
# save_and_play(wavs[0], sr, "ex1_pirate.wav", "Pirate captain voice")

print("\n")

# Exercise 2: Thai Voice
print("=" * 60)
print("Exercise 2: Thai TV Host Voice 🇹🇭")
print("=" * 60)

# Uncomment and modify:
# tv_host_text = """สวัสดีค่ะทุกท่าน ยินดีต้อนรับเข้าสู่รายการของเรา 
# วันนี้เรามีเรื่องน่าสนใจมากมายมาเล่าให้ฟังค่ะ"""
# 
# tv_host_description = """
# A cheerful Thai female TV host in her 30s. Energetic and engaging.
# Clear pronunciation with proper polite particles.
# Warm smile in her voice. Professional but friendly.
# """
# 
# wavs, sr = voice_design_model.generate_voice_design(
#     text=tv_host_text,
#     language="Chinese",  # Experimental for Thai
#     instruct=tv_host_description,
# )
# save_and_play(wavs[0], sr, "ex2_thai_tv_host.wav", "Thai TV host voice")

print("Uncomment the examples above or write your own code!")

# %% [markdown]
# ---
# ## 📚 Resources
#
# - **GitHub Repository**: https://github.com/QwenLM/Qwen3-TTS
# - **Hugging Face Models**: https://huggingface.co/collections/Qwen/qwen3-tts
# - **Technical Paper**: https://arxiv.org/abs/2601.15621
# - **Online Demo**: https://huggingface.co/spaces/Qwen/Qwen3-TTS
# - **API Documentation**: https://help.aliyun.com/zh/model-studio/qwen-tts-realtime
#
# ---
#
# **End of Lab**
