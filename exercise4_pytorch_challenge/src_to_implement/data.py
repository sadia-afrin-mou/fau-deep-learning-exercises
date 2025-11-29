from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        """
        Initialize the dataset.
        
        Args:
            data (pandas.DataFrame): DataFrame containing the dataset information from data.csv
            mode (str): Either 'train' or 'val' to specify the dataset mode
        """
        self.data = data
        self.mode = mode
        
        # Create different transforms based on mode
        if mode == 'train':
            # Training transforms with data augmentation
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.RandomVerticalFlip(p=0.5),
                # tv.transforms.RandomRotation(degrees=10),
                tv.transforms.RandomAffine(degrees=(-3, 3), translate=(0.02, 0.02)),
                tv.transforms.RandomResizedCrop((300, 300), scale=(0.98, 1.0), ratio=(1.0, 1.0)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:
            # Validation transforms without augmentation
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
    
    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Get item at the specified index.
        
        Args:
            index (int): Index of the item to retrieve
            
        Returns:
            tuple: (image, label) where image is a tensor and label is a tensor with shape (2,)
        """
        # Get the row from the dataframe
        row = self.data.iloc[index]
        
        # Load the image
        image_path = row['filename']
        image = imread(image_path)
        
        # Convert grayscale to RGB
        if len(image.shape) == 2:  # If grayscale
            image = gray2rgb(image)
        
        # Get labels (crack and inactive)
        crack_label = row['crack']
        inactive_label = row['inactive']
        labels = torch.tensor([crack_label, inactive_label], dtype=torch.float32)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, labels
