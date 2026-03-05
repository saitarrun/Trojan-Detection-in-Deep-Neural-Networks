import torch
import sys
import os

def check_env():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU device count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Check if we are in the Nautilus workspace
    workspace = "/workspace"
    if os.path.exists(workspace):
        print(f"Workspace {workspace} is accessible.")
        print(f"Files in {workspace}: {os.listdir(workspace)}")
    else:
        print(f"Workspace {workspace} not found. Are you running locally?")

if __name__ == "__main__":
    check_env()
