import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset 
import os
from PIL import Image
import numpy as np
import kagglehub

#--------------------------------------------------------------------------------------------------------------
# Shallow Feature Extractor - Architecture
#--------------------------------------------------------------------------------------------------------------

class Shallow_Feature_Extractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=180):
        super(Shallow_Feature_Extractor, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                  nn.Conv2d(out_channels,64, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(inplace=True),
                                  nn.PixelShuffle(upscale_factor=2),
                                  nn.Conv2d(16,64, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(inplace=True),
                                  nn.PixelShuffle(upscale_factor=2),
                                  nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.body(x)
        return x


#--------------------------------------------------------------------------------------------------------------
# Shallow Feature Extractor - Training
#--------------------------------------------------------------------------------------------------------------

model = Shallow_Feature_Extractor()
optimizer = optim.Adam(model.parameters(), lr=2e-4)
loss_fn = nn.L1Loss()

def train_step(model, optimizer, loss_fn, input_data, target_data):
    optimizer.zero_grad()
    output = model(input_data)
    loss = loss_fn(output, target_data)
    loss.backward()
    optimizer.step()
    return loss.item()

#--------------------------------------------------------------------------------------------------------------
# Shallow Feature Extractor - Data Preparation
#--------------------------------------------------------------------------------------------------------------
class SRDataset(Dataset):
    def __init__(self, hr_dir, patch_size=256, scale_factor=4, file_list=None):
        # data loading
        super(SRDataset, self).__init__()
        self.hr_dir = hr_dir
        self.patch_size_hr = patch_size
        self.patch_size_lr = patch_size // scale_factor
        self.scale_factor = scale_factor
        self.file_list = os.listdir(hr_dir) if file_list is None else file_list
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        # len(dataset) returns the number of samples in the dataset
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # dataset[idx] returns the sample at index idx
        hr_image_path = os.path.join(self.hr_dir, self.file_list[idx])
        hr_image = Image.open(hr_image_path).convert('RGB')
        hr_patch, lr_patch = self.get_patch(hr_image)
        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)
    
    def get_patch(self, hr_image):
        hr_w, hr_h = hr_image.size
        lr_w, lr_h = hr_w // self.scale_factor, hr_h // self.scale_factor

        patch_start_lr = np.random.randint(0, lr_w - self.patch_size_lr + 1), np.random.randint(0, lr_h - self.patch_size_lr + 1)
        patch_start_hr = patch_start_lr[0] * self.scale_factor, patch_start_lr[1] * self.scale_factor

        hr_patch = hr_image.crop((patch_start_hr[0], patch_start_hr[1], patch_start_hr[0] + self.patch_size_hr, patch_start_hr[1] + self.patch_size_hr))
        lr_patch = hr_patch.resize((self.patch_size_lr, self.patch_size_lr), Image.BICUBIC)
        return hr_patch, lr_patch
    
# Dataset Directories
DIV2K_train_HR_dir = kagglehub.dataset_download("takihasan/div2k-dataset-for-super-resolution/DIV2K_train_HR")
DIV2K_valid_HR_dir = kagglehub.dataset_download("takihasan/div2k-dataset-for-super-resolution/DIV2K_valid_HR")

# Dataset Preparation
train_dataset = SRDataset(hr_dir=DIV2K_train_HR_dir, patch_size=256, scale_factor=4)
valid_dataset = SRDataset(hr_dir=DIV2K_valid_HR_dir, patch_size=256, scale_factor=4)

# DatLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)


#--------------------------------------------------------------------------------------------------------------
# Shallow Feature Extractor - Main
#--------------------------------------------------------------------------------------------------------------