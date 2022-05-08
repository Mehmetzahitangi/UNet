# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:46:44 2022
"""

import os
import torch
from torchvision.transforms import transforms
from natsort import natsorted
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import cv2
import PIL
from models.model import UNet

def predict_mask(output,image):
    prediction = torch.argmax(output,1)
    print("Prediction shape: ", prediction.shape)
    prediction = prediction.permute(1,2,0) # CHW ==> Height,Width,Channel, (1024,1024,1
    print("Prediction shape after premute : ", prediction.shape)
    prediction = prediction.cpu().numpy()
    
    mask = prediction.astype("uint8")
    
    # reverse label_color map
    label_color_map = {
        0:0,
        1:255
        }
    
    for k in label_color_map:
        mask[mask==k] = label_color_map[k]
        
        
    cv2.imshow("Image: ", image.astype(np.uint8))
    cv2.imshow("Mask: " , mask)
    cv2.waitkey(0)
    
def show_probability_map(output):
    # slice output channels of prediction, show probability mao for each class
    output = output.cpu()
    #prob = F.softmax(output,1)
    prob = torch.exp(output) # using log_softmax in model, so torch.exp to get probabilities
    prob_imgs = torchvision.utils.make_grid(prob.permute(1,0,2,3))
    plt.imshow(prob_imgs.permute(1,2,0))
    plt.show()
    
    
if __name__ == "__main__":
    to_tensor = transforms.ToTensor()
    to_gray = transforms.GrayScale()
    
    images_path = "./DATA/test/images/"
    images = natsorted(os.listdir(images_path))

    for image in images:
        image = Image.open(images_path + image)
        image = image.resize((256,256), resample = PIL.Image.NEAREST)
        image = to_gray(image)
        image = np.array(image)
        t_image= to_tensor(image)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = UNet(num_classes = 3)
        model = model.to(device)
        model.load_state_dict(torch.load("./model.pth"))
        model.eval()
        
        with torch.no_grad():
            t_image = t_image.unsquueze(0) # (1,1,1024,1024)
            t_image = t_image.to(device)
            print(t_image.shape)
            output = model(t_image) # (1,3,1024,1024) 3 is num. of classes, mapping must be number_classes => 0,1,2
            
        predict_mask(output,image)
        show_probability_map(output)