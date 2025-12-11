import torch

print("\nChecking CUDA availability...\n")

print("CUDA available :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name       :", torch.cuda.get_device_name(0))
    print("CUDA Version   :", torch.version.cuda)
    print("PyTorch CUDA   :", torch.backends.cudnn.version())
else:
    print("❌ CUDA not available.")
    print("If you see this, you installed CPU-only PyTorch or venv not activated.")
