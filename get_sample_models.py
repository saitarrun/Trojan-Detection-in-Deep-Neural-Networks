import torch
import os

def download_samples():
    print("Downloading external community models for testing...")
    
    # Create the directory if it doesn't exist
    sample_dir = "sample_external_models"
    os.makedirs(sample_dir, exist_ok=True)
    
    # 1. Download a "Clean" community model
    print("\nFetching Clean ResNet20 from chenyaofo...")
    try:
        model1 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
        torch.save(model1.state_dict(), os.path.join(sample_dir, "community_clean_resnet20.pth"))
        print(f"✅ Saved 'community_clean_resnet20.pth' to {sample_dir}/")
    except Exception as e:
        print(f"Error fetching model: {e}")
        
    # 2. Duplicate a local poisoned model to act as a "Malicious" external download
    print("\nPreparing a Malicious sample model...")
    try:
        import shutil
        if os.path.exists("models/blended_model.pth"):
            shutil.copy("models/blended_model.pth", os.path.join(sample_dir, "unknown_malicious_model.pth"))
            print(f"✅ Saved 'unknown_malicious_model.pth' to {sample_dir}/")
        else:
            print("Local blended_model.pth not found. Try running the training script first.")
    except Exception as e:
         print(f"Error copying model: {e}")
         
    print("\n==============================================")
    print(f"Testing samples are ready in the '{sample_dir}/' folder.")
    print("You can now upload these files through the Streamlit UI!")

if __name__ == "__main__":
    download_samples()
