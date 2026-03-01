import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TrojAIDataset(Dataset):
    """
    Universal PyTorch Dataset for loading high-resolution images from the TrojAI dataset.
    Rounds like Round 1 use polygon/stop-sign images with custom label lists.
    """
    def __init__(self, data_dir, image_size=(224, 224), transform=None):
        self.data_dir = data_dir
        
        # Standard TrojAI images are 224x224 (DenseNet/ResNet50) or 299x299 (Inception)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size), # Dynamic resizing for TrojAI
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        self.image_paths = []
        self.labels = []
        
        # In a real TrojAI folder, ground_truth.csv exists. 
        # For this script we will walk the directory to simulate.
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.png') or file.endswith('.jpg'):
                    self.image_paths.append(os.path.join(data_dir, file))
                    # Simplified parsing (e.g., class_0_image1.png)
                    label_str = file.split('_')[1] if '_' in file else "0"
                    self.labels.append(int(label_str) if label_str.isdigit() else 0)

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

def get_trojai_dataloader(data_dir, batch_size=32, image_size=(224, 224)):
    dataset = TrojAIDataset(data_dir, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
