import subprocess, sys, os

def run(cmd, check=False):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-3000:])
    if result.returncode != 0 and result.stderr:
        print("STDERR:", result.stderr[-1000:])
    return result

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
run("pip install 'datasets>=2.20.0' huggingface_hub evaluate -q")

# Step 5: Install remaining project dependencies
print("\nSTEP 5: Installing project dependencies...")
run("pip install loguru pydantic scikit-learn networkx einops sentencepiece openai 'regex>=2024.0.0' 'dill>=0.3.8' pyyaml tqdm -q")

# Step 6: Resolve CUDA library path PERMANENTLY
# bitsandbytes must find libnvJitLink at process startup — setting LD_LIBRARY_PATH
# inside Python after import is too late. We use ldconfig + /etc/environment to persist it.
print("\nSTEP 6: Resolving CUDA library path permanently...")

# Find all CUDA lib dirs that actually exist
cuda_candidates = [
    "/usr/local/cuda/lib64",
    "/usr/local/cuda-13.0/lib64",
    "/usr/local/cuda-12.6/lib64",
    "/usr/local/cuda-12.1/lib64",
    "/usr/lib/x86_64-linux-gnu",
]
# Also search dynamically
find_result = run("find /usr/local/cuda* -maxdepth 3 -name 'libnvJitLink*' 2>/dev/null")
found_dirs = set()
for line in find_result.stdout.strip().splitlines():
    found_dirs.add(os.path.dirname(line))

# Add known dirs that exist
for p in cuda_candidates:
    if os.path.exists(p):
        found_dirs.add(p)

ld_path = ":".join(sorted(found_dirs))
print(f"  Found CUDA lib dirs: {ld_path or '(none found)'}")

if found_dirs:
    # 1. Register with ldconfig (system-wide, survives subprocess launches)
    for d in found_dirs:
        run(f"echo '{d}' >> /etc/ld.so.conf.d/cuda-nariraksha.conf")
    run("ldconfig 2>/dev/null || true")

    # 2. Write to /etc/environment (survives shell restarts in same session)
    with open("/etc/environment", "a") as f:
        f.write(f'\nLD_LIBRARY_PATH={ld_path}:$LD_LIBRARY_PATH\n')

    # 3. Add to /etc/bash.bashrc (picked up by new bash subprocesses)
    with open("/etc/bash.bashrc", "a") as f:
        f.write(f'\nexport LD_LIBRARY_PATH={ld_path}:$LD_LIBRARY_PATH\n')

    # 4. Set for current process too
    os.environ["LD_LIBRARY_PATH"] = ld_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    print(f"  LD_LIBRARY_PATH configured ✅")
else:
    print("  ⚠️  No CUDA lib dirs found — bitsandbytes may fail. Try: find / -name 'libnvJitLink*' 2>/dev/null")

# Step 7: Write the training launcher shell script
# This ensures LD_LIBRARY_PATH is correct even after a runtime restart
launcher_content = f"""#!/bin/bash
# NariRaksha Training Launcher
# Ensures CUDA library path is set before Python starts (critical for bitsandbytes)
export LD_LIBRARY_PATH={ld_path}:$LD_LIBRARY_PATH
exec python /content/NariRaksha/training/train.py "$@"
"""
with open("/content/NariRaksha/scripts/run_training.sh", "w") as f:
    f.write(launcher_content)
run("chmod +x /content/NariRaksha/scripts/run_training.sh")
print("\nSTEP 7: Launcher script written to scripts/run_training.sh ✅")

# Step 8: Verify the full stack
print("\nSTEP 8: Verifying installation...")
import subprocess as _sp
_env = os.environ.copy()
if ld_path:
    _env["LD_LIBRARY_PATH"] = ld_path + ":" + _env.get("LD_LIBRARY_PATH", "")
_sp.run([sys.executable, "-c", "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"], env=_env)
_sp.run([sys.executable, "-c", "import bitsandbytes as bnb; print('bitsandbytes:', bnb.__version__)"], env=_env)
_sp.run([sys.executable, "-c", "from unsloth import FastLanguageModel; print('Unsloth: OK')"], env=_env)

print("\n" + "=" * 60)
print("✅ Setup complete!")
print("\n⚠️  IMPORTANT: Runtime > Restart session")
print("\nAfter restart, use the launcher to train:")
print("  !bash scripts/run_training.sh --config configs/qwen.yaml")
print("\nOR use the %%bash magic directly:")
print(f"  %%bash")
print(f"  export LD_LIBRARY_PATH={ld_path}:$LD_LIBRARY_PATH")
print(f"  python training/train.py --config configs/qwen.yaml")
print("=" * 60)
