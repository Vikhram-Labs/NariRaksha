import subprocess, sys, os

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-3000:])
    if result.returncode != 0 and result.stderr:
        print("STDERR:", result.stderr[-1000:])

print("=" * 60)
print("NariRaksha - Definitive Colab T4 Setup (Unsloth-based)")
print("=" * 60)

# Step 1: Wipe all conflicting packages cleanly
print("\nSTEP 1: Wiping conflicting packages...")
run("pip uninstall -y bitsandbytes transformers peft trl accelerate datasets torch torchvision torchaudio triton unsloth 2>/dev/null")

# Step 2: Install Unsloth — handles ALL CUDA/PyTorch/bitsandbytes compatibility automatically
print("\nSTEP 2: Installing Unsloth (bundles compatible PyTorch + bitsandbytes for T4)...")
run('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q')

# Step 3: Install TRL and PEFT (without deps to avoid overriding Unsloth's pinned versions)
print("\nSTEP 3: Installing TRL + PEFT...")
run("pip install --no-deps trl peft accelerate -q")

# Step 4: Install datasets and other HF utilities
print("\nSTEP 4: Installing datasets + HF utilities...")
run("pip install datasets>=2.20.0 huggingface_hub evaluate -q")

# Step 5: Install remaining project dependencies
print("\nSTEP 5: Installing project dependencies...")
run("pip install loguru pydantic scikit-learn networkx einops sentencepiece openai regex>=2024.0.0 dill>=0.3.8 pyyaml tqdm -q")

# Step 6: Verify the stack
print("\nSTEP 6: Verifying installation...")
run('python -c "import torch; print(\'PyTorch:\', torch.__version__, \'| CUDA:\', torch.cuda.is_available())"')
run('python -c "import bitsandbytes as bnb; print(\'bitsandbytes:\', bnb.__version__)"')
run('python -c "from unsloth import FastLanguageModel; print(\'Unsloth: OK\')"')

print("\n" + "=" * 60)
print("✅ Setup complete!")
print("⚠️  NOW DO: Runtime > Restart session")
print("   Then run Cell 3 (verify) and Cell 6 (training)")
print("=" * 60)
