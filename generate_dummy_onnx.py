import torch
from torchvision.models import resnet18
import os

def create_dummy_onnx():
    print("Generating dummy ONNX model...")
    model = resnet18(num_classes=10)
    model.eval()

    # Create dummy input matching CIFAR-10 shape
    dummy_input = torch.randn(1, 3, 32, 32)
    
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/dummy_model.onnx"
    
    # Export the model, forcing weights to be embedded inside the file
    # PyTorch dynamically offloads weights to .data for anything > ~some MB threshold.
    # To force a single file, we disable the external data format.
    torch.onnx.export(
        model,               # model being run
        dummy_input,         # model input (or a tuple for multiple inputs)
        onnx_path,           # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=14,    # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
    )
    print(f"ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    create_dummy_onnx()
