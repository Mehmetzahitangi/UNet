from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as func

def double_conv(in_channel_size, out_channel_size):
    conv = nn.Sequential(
        nn.Conv2d(in_channels = in_channel_size, out_channels = out_channel_size, kernel_size = 3),
        nn.ReLU(),
        nn.Conv2d(in_channels = out_channel_size, out_channels = out_channel_size, kernel_size = 3),
        nn.ReLU()
    )

    return conv

def double_up_conv(in_channel_size, out_channel_size):
    up_conv = nn.Sequential(
        nn.Conv2d(in_channels = in_channel_size, out_channels = out_channel_size, kernel_size = 3),
        nn.ReLU(),
        nn.Conv2d(in_channels = out_channel_size, out_channels = out_channel_size, kernel_size = 3),
        nn.ReLU()
    )

    return up_conv

def crop_tensor(target_tensor, change_tensor):
    # from 1,512,64,64 to 1,512,56,56
    target_size = target_tensor.size()[2]
    tensor_size = change_tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2 
    return change_tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()

        #encoder
        self.maxpool2x2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv1 = double_conv(1,64)
        self.conv2 = double_conv(64,128)
        self.conv3 = double_conv(128,256)
        self.conv4 = double_conv(256,512)
        self.conv5 = double_conv(512,1024)

        #decoder
        self.up_conv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,kernel_size=2,stride=2)
        self.up_conv1_3x3 = double_up_conv(in_channel_size= 1024, out_channel_size=512)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256,kernel_size=2,stride=2)
        self.up_conv2_3x3 = double_up_conv(in_channel_size= 512, out_channel_size=256)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,kernel_size=2,stride=2)
        self.up_conv3_3x3 = double_up_conv(in_channel_size= 256, out_channel_size=128)

        self.up_conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64,kernel_size=2,stride=2)
        self.up_conv4_3x3 = double_up_conv(in_channel_size= 128, out_channel_size=64)

    def forward(self,image):
        #encoder
        layer1 = self.conv1(image) #1,64,568,568
        print("layer1: ",layer1.shape)
        layer2 = self.maxpool2x2(layer1)#1,64,284,284
        print("layer2: ",layer2.shape)

        layer3 = self.conv2(layer2)#1,128,280,280
        print("layer3: ",layer3.shape)        
        layer4 = self.maxpool2x2(layer3)#1,128,140,140
        print("layer4: ",layer4.shape)

        layer5 = self.conv3(layer4)#1,256,136,136
        print("layer5: ",layer5.shape)        
        layer6 = self.maxpool2x2(layer5)#1,256,68,68
        print("layer6: ",layer6.shape)

        layer7 = self.conv4(layer6)#1,512,64,64
        print("layer7: ",layer7.shape)
        layer8 = self.maxpool2x2(layer7)#1,512,32,32
        print("layer8: ",layer8.shape)

        layer9 = self.conv5(layer8)#1,1024,28,28
        print("layer9: ",layer9.shape)

        #decoder
        layer10 = self.up_conv1(layer9) # 1,512,56,56
        print("layer10: ",layer10.shape)
        layer11 = torch.cat((layer10,crop_tensor(target_tensor = layer10, change_tensor = layer7)),dim=1)
        print("layer11: ",layer11.shape)
        layer12 = self.up_conv1_3x3(layer11)
        print("layer12: ",layer12.shape)    

        layer13 = self.up_conv2(layer12)
        print("layer13: ",layer13.shape)    
        #layer14 = torch.cat((layer13,layer5),dim=1)
        layer14 = torch.cat((layer13,crop_tensor(target_tensor = layer13, change_tensor = layer5)),dim=1)
        print("layer14: ",layer14.shape) 
        layer15 = self.up_conv2_3x3(layer14)
        print("layer15: ",layer15.shape)  

        layer16 = self.up_conv3(layer15)
        print("layer16: ",layer16.shape)  
        #layer17 = torch.cat((layer16,layer3),dim=1)
        layer17 = torch.cat((layer16,crop_tensor(target_tensor = layer16, change_tensor = layer3)),dim=1)
        print("layer17: ",layer17.shape)    
        layer18 = self.up_conv3_3x3(layer17)
        print("layer18: ",layer18.shape)  

        layer19 = self.up_conv4(layer18)
        print("layer19: ",layer19.shape)    
        #layer20 = torch.cat((layer19,layer1),dim=1)
        layer20 = torch.cat((layer19,crop_tensor(target_tensor = layer19, change_tensor = layer1)),dim=1)
        print("layer20: ",layer20.shape)    
        layer21 = self.up_conv4_3x3(layer20)
        print("layer21: ",layer21.shape)  

        return layer21

model = UNet()

aug_data = torch.rand(size=(1,1,572,572))

output = model(aug_data) 