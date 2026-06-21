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

# ──────────────────────────────────────────────────────────────
# STEP 0 (CRITICAL): Detect CUDA 12.x paths and set LD_LIBRARY_PATH
# BEFORE any pip installs so ldconfig is configured early.
#
# Colab 2025 CUDA layout (confirmed):
#   - /usr/local/cuda-12.8/targets/x86_64-linux/lib/  (NOT lib64!)
#   - /usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib/
#   - /usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib/
#   - DANGER: nvidia/cu13/lib/ also exists with libnvJitLink.so.13
#             → must be EXCLUDED or bitsandbytes 0.45.x will crash
# ──────────────────────────────────────────────────────────────
print("\nSTEP 0: Detecting CUDA version and pre-configuring library paths...")

# Detect the CUDA version present on this machine
cuda_ver_result = run("nvcc --version 2>/dev/null || nvidia-smi 2>/dev/null | head -5")
print(f"  CUDA info: {cuda_ver_result.stdout[:200]}")

import glob as _glob

found_dirs = set()

# 1. Standard layout: /usr/local/cuda-X.Y/lib64/
for pat in ["/usr/local/cuda*/lib64/libcudart.so*",
            "/usr/local/cuda*/lib64/libnvJitLink*.so*"]:
    for hit in _glob.glob(pat):
        found_dirs.add(os.path.dirname(hit))

# 2. Colab 2025 layout: /usr/local/cuda-X.Y/targets/arch/lib/
for pat in ["/usr/local/cuda*/targets/*/lib/libcudart.so*",
            "/usr/local/cuda*/targets/*/lib/libnvJitLink*.so*"]:
    for hit in _glob.glob(pat):
        found_dirs.add(os.path.dirname(hit))

# 3. pip-installed nvidia packages (site-packages/nvidia/<pkg>/lib/)
_site = "/usr/local/lib/python3.12/dist-packages/nvidia"
for pkg in ["cuda_runtime", "cudart", "nvjitlink"]:
    _candidate = os.path.join(_site, pkg, "lib")
    if os.path.isdir(_candidate):
        found_dirs.add(_candidate)

# 4. FILTER: remove directories that ONLY contain CUDA 13.x libs
#    nvidia/cu13/lib has libnvJitLink.so.13 but NOT .so.12 — exclude it
def _has_cuda12(d):
    return bool(
        _glob.glob(os.path.join(d, "lib*.so.12*")) or
        _glob.glob(os.path.join(d, "libcudart.so*"))
    )
found_dirs = {d for d in found_dirs if _has_cuda12(d)}

ld_path = ":".join(sorted(found_dirs))
print(f"  CUDA 12.x lib dirs: {ld_path or '(none found)'}")

if found_dirs:
    # Register with ldconfig (persistent for this session's subprocesses)
    with open("/etc/ld.so.conf.d/cuda-nariraksha.conf", "w") as f:
        f.write("\n".join(sorted(found_dirs)) + "\n")
    run("ldconfig 2>/dev/null || true")

    # Set for the current process immediately
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ld_path + (":" + current_ld if current_ld else "")

    # Persist for new bash sessions (resolved value, not a variable reference)
    bash_export = f'export LD_LIBRARY_PATH={ld_path}:${{LD_LIBRARY_PATH:-}}'
    for rc_file in ["/etc/bash.bashrc", "/root/.bashrc"]:
        try:
            with open(rc_file, "a") as f:
                f.write(f"\n# NariRaksha CUDA path\n{bash_export}\n")
        except Exception:
            pass
    print(f"  LD_LIBRARY_PATH pre-configured ✅")

else:
    print("  ⚠️  No libnvJitLink found — will attempt install anyway.")
    ld_path = ""

# ──────────────────────────────────────────────────────────────
# STEP 1: Wipe conflicting packages
# ──────────────────────────────────────────────────────────────
print("\nSTEP 1: Wiping conflicting packages...")
run("pip uninstall -y bitsandbytes transformers peft trl accelerate datasets "
    "torch torchvision torchaudio triton unsloth 2>/dev/null || true")

# ──────────────────────────────────────────────────────────────
# STEP 2: Install bitsandbytes pinned to CUDA 12.x compatible version
#
# ROOT CAUSE FIX: bitsandbytes >= 0.46.0 ships binaries compiled
# against CUDA 13.x (libnvJitLink.so.13). Colab T4 runs CUDA 12.x.
# Pinning to 0.45.5 is the last stable CUDA 12.x build.
# ──────────────────────────────────────────────────────────────
print("\nSTEP 2: Installing bitsandbytes pinned for CUDA 12.x (T4 compatible)...")
run("pip install 'bitsandbytes==0.45.5' -q")

# ──────────────────────────────────────────────────────────────
# STEP 3: Install Unsloth with compatible PyTorch for T4
# ──────────────────────────────────────────────────────────────
print("\nSTEP 3: Installing Unsloth (T4-optimized)...")
run('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q')

# ──────────────────────────────────────────────────────────────
# STEP 4: Install TRL + PEFT without overriding Unsloth's pinned versions
# ──────────────────────────────────────────────────────────────
print("\nSTEP 4: Installing TRL + PEFT (no-deps to preserve Unsloth pins)...")
run("pip install --no-deps trl peft accelerate -q")

# ──────────────────────────────────────────────────────────────
# STEP 5: HuggingFace utilities
# ──────────────────────────────────────────────────────────────
print("\nSTEP 5: Installing HF utilities...")
run("pip install 'datasets>=2.20.0' huggingface_hub evaluate -q")

# ──────────────────────────────────────────────────────────────
# STEP 6: Remaining project dependencies
# ──────────────────────────────────────────────────────────────
print("\nSTEP 6: Installing project dependencies...")
run("pip install loguru pydantic scikit-learn networkx einops sentencepiece openai "
    "'regex>=2024.0.0' 'dill>=0.3.8' pyyaml tqdm -q")

# ──────────────────────────────────────────────────────────────
# STEP 7: Write the training launcher shell script
# This ensures LD_LIBRARY_PATH is set BEFORE Python starts,
# which is the only reliable way for bitsandbytes to find libnvJitLink.
# ──────────────────────────────────────────────────────────────
print("\nSTEP 7: Writing training launcher script...")
launcher_content = """#!/bin/bash
# NariRaksha Training Launcher
# LD_LIBRARY_PATH is injected HERE, before Python starts.
# This is the correct fix for bitsandbytes CUDA library errors.
export LD_LIBRARY_PATH="{ld_path}:${{LD_LIBRARY_PATH:-}}"
exec python /content/NariRaksha/training/train.py "$@"
""".format(ld_path=ld_path)

os.makedirs("/content/NariRaksha/scripts", exist_ok=True)
with open("/content/NariRaksha/scripts/run_training.sh", "w") as f:
    f.write(launcher_content)
run("chmod +x /content/NariRaksha/scripts/run_training.sh")
print("  Launcher written to scripts/run_training.sh ✅")

# ──────────────────────────────────────────────────────────────
# STEP 8: Verify the full stack in a subprocess with correct env
# ──────────────────────────────────────────────────────────────
print("\nSTEP 8: Verifying installation stack...")
import subprocess as _sp
_env = os.environ.copy()  # already has LD_LIBRARY_PATH from Step 0

print("  Testing PyTorch...")
_sp.run([sys.executable, "-c",
    "import torch; print('  PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"],
    env=_env)

print("  Testing bitsandbytes...")
bnb_test = _sp.run([sys.executable, "-c",
    "import bitsandbytes as bnb; print('  bitsandbytes:', bnb.__version__)"],
    env=_env, capture_output=True, text=True)
if bnb_test.returncode == 0:
    print(bnb_test.stdout)
else:
    print(f"  ⚠️  bitsandbytes import failed:\n{bnb_test.stderr[-500:]}")
    print("  → Try: pip install 'bitsandbytes==0.44.1' and retry")

print("  Testing Unsloth...")
unsloth_test = _sp.run([sys.executable, "-c",
    "from unsloth import FastLanguageModel; print('  Unsloth: OK')"],
    env=_env, capture_output=True, text=True)
if unsloth_test.returncode == 0:
    print(unsloth_test.stdout)
else:
    print(f"  ⚠️  Unsloth import failed:\n{unsloth_test.stderr[-500:]}")

print("\n" + "=" * 60)
print("✅ Setup complete!")
print("\n⚠️  MANDATORY: Runtime > Restart session before training!")
print("   (bitsandbytes reads LD_LIBRARY_PATH at Python startup, not mid-run)")
print("\nAfter restart, run training with the launcher:")
print("  !bash /content/NariRaksha/scripts/run_training.sh --config configs/qwen.yaml")
print("\nOR set LD_LIBRARY_PATH manually in a %%bash cell first:")
if ld_path:
    print(f"  %%bash")
    print(f"  export LD_LIBRARY_PATH={ld_path}:$LD_LIBRARY_PATH")
    print(f"  python training/train.py --config configs/qwen.yaml")
print("=" * 60)
