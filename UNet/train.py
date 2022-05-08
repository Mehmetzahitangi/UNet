# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:17:49 2022
"""

import torch 
import torch.nn as nn
import torch.optim as optimizer
from tqdm import tqdm
from models.model import UNet
from utils.dataset import CustomDataset

def train(epochs, trainLoader):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 1, factor = 0.1, verbose = True)
    
    mean_losses = []
     
    print("Training is started.")
    for epoch in range(epochs):
        running_loss = []
        loop = tqdm(enumerate(trainLoader), total = len(trainLoader))
        
        for idx,(image,mask) in loop:
            image,mask = image.to(device), mask.to(device)
            outputs = model(image)
            loss = criterion(outputs,mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item())
            mean_loss = sum(running_loss) / len(running_loss)
            
            loop.set_description("Epoch: ", [{epoch + 1 }/{epochs}])
            loop.set_postfix(batch_loss = loss.item(), mean_loss = mean_loss, lr = optimizer.param_groups[0]["lr"])
            
        if len(mean_losses) >= 1:
            if mean_loss < min(mean_losses):
                print("Model Saved")
                torch.save(model.state_dict(), "model_v2.pth")
                
        mean_losses.append(mean_loss)
        scheduler.step(mean_loss)
        
        
if __name__ == "__main__":
    img_paths = "./DATA/train/images"
    mask_paths = "./Data/train/masks"
    
    train_dataset = CustomDataset(img_paths, mask_paths, input_size=(256,256))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True, num_workers= 4)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=2)
    model = model.to(device)
    
    criterion = nn.NLLoss()
    optimizer = optimizer.Adam(model.parameters(), lr = 1e-3 )
    train(epochs = 350, trainLoader = train_loader)
    print("Training is ended")
    