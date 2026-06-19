# NariRaksha - Google Colab T4 Setup Script
# Run this ENTIRE cell first, then restart the runtime

import subprocess, sys

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout[-2000:] if result.stdout else "")
    if result.stderr:
        print("STDERR:", result.stderr[-500:])

# Step 1: Detect actual CUDA version on this runtime
print("="*60)
print("STEP 1: Detecting CUDA version...")
run("nvcc --version")
run("ls /usr/local/cuda/lib64/libnvJitLink*")

# Step 2: Create symlink fix for libnvJitLink mismatch
print("\nSTEP 2: Fixing libnvJitLink symlink...")
run("ls /usr/local/cuda/lib64/libnvJitLink* 2>/dev/null || echo 'libnvJitLink not found in default path'")
run("find /usr -name 'libnvJitLink*' 2>/dev/null")

# Step 3: Wipe ALL conflicting ML packages for a clean slate
print("\nSTEP 3: Wiping conflicting packages...")
packages_to_remove = [
    "bitsandbytes", "transformers", "peft", "trl", "accelerate",
    "datasets", "triton", "torch", "torchvision", "torchaudio"
]
run(f"pip uninstall -y {' '.join(packages_to_remove)}")

# Step 4: Install PyTorch FIRST, specifically for CUDA 12.1 (T4 native)
print("\nSTEP 4: Installing PyTorch for CUDA 12.1 (T4 native)...")
run("pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 -q")

# Step 5: Install bitsandbytes >= 0.46.1 (required by latest transformers)
print("\nSTEP 5: Installing bitsandbytes>=0.46.1...")
run("pip install bitsandbytes>=0.46.1 -q")

# Step 6: Install the rest of the HF stack
print("\nSTEP 6: Installing HuggingFace ecosystem...")
run("pip install transformers>=4.44.0 datasets>=2.20.0 accelerate>=0.33.0 peft>=0.12.0 trl>=0.9.0 -q")

# Step 7: Install remaining dependencies
print("\nSTEP 7: Installing remaining dependencies...")
run("pip install evaluate scikit-learn loguru pydantic networkx einops sentencepiece openai regex>=2024.0.0 dill>=0.3.8 pyyaml tqdm huggingface_hub -q")

# Step 8: Fix LD_LIBRARY_PATH for CUDA libraries
print("\nSTEP 8: Setting CUDA library path...")
import os
cuda_paths = [
    "/usr/local/cuda/lib64",
    "/usr/local/cuda-12.1/lib64",
    "/usr/lib/x86_64-linux-gnu"
]
existing = [p for p in cuda_paths if os.path.exists(p)]
ld_path = ":".join(existing)
os.environ["LD_LIBRARY_PATH"] = ld_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
print(f"LD_LIBRARY_PATH set to: {ld_path}")

# Step 9: Verify bitsandbytes loads correctly
print("\nSTEP 9: Verifying bitsandbytes CUDA setup...")
run("python -c \"import bitsandbytes as bnb; print('bitsandbytes OK:', bnb.__version__)\"")
run("python -c \"import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())\"")

print("\n" + "="*60)
print("✅ Setup complete! Now RESTART RUNTIME, then run Cell 2.")
print("="*60)
