import platform
import shutil
import subprocess
import sys

print("=" * 72)
print("CrackSSL Environment Checker")
print("=" * 72)
print("Python:", sys.version.replace("\n", " "))
print("Executable:", sys.executable)
print("Platform:", platform.platform())

nvidia_smi = shutil.which("nvidia-smi")
if nvidia_smi:
    try:
        r = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        print("nvidia-smi:", r.stdout.strip() if r.returncode == 0 else "available but query failed")
    except Exception as e:
        print("nvidia-smi query error:", repr(e))
else:
    print("nvidia-smi: not found in PATH")

try:
    import torch
except ModuleNotFoundError:
    print("PyTorch: NOT INSTALLED (expected at this stage)")
else:
    print("PyTorch:", torch.__version__)
    print("CUDA available in PyTorch:", torch.cuda.is_available())
    print("PyTorch CUDA runtime:", torch.version.cuda)
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(f"GPU {idx}: {torch.cuda.get_device_name(idx)} | {props.total_memory / 1024**3:.2f} GB")

print("=" * 72)
