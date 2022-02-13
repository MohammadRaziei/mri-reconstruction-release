import numpy as np
import torch
import torch.nn.functional as F
from skimage.restoration import estimate_sigma

class Sobel2d(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        sobel_x_kernel = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/8 
        self.sobel_x_kernel = sobel_x_kernel.repeat(num_channels,1,1,1)
        sobel_y_kernel = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/8
        self.sobel_y_kernel = sobel_y_kernel.repeat(num_channels,1,1,1)
        
    def forward(self, inputs):
        grad_sobel_x = F.conv2d(inputs, self.sobel_x_kernel, stride=1, 
                                padding=1, groups=self.num_channels)
        grad_sobel_y = F.conv2d(inputs, self.sobel_y_kernel, stride=1, 
                                padding=1, groups=self.num_channels)
        grad_sobel = torch.hypot(grad_sobel_x, grad_sobel_y)
        return grad_sobel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sobel2d = Sobel2d(1).to(device)


def LMGNR(target, kernel_size=20, eps=None, version=2):
    image = torch.tensor(target).unsqueeze(0).unsqueeze(0)
    grad_image = sobel2d(image)
    LMG = torch.nn.MaxPool2d(kernel_size, stride=kernel_size//2)(grad_image)
    LMG = LMG[0,0,1:-1, 1:-1]
    E_lmg = torch.mean(LMG**2).item()
    sigma_x = image.std().item()
    sigma_n = estimate_sigma(target)
    sigma2_x = sigma_x**2 

    if eps is None:
        if version == 1:
            eps = 0.001
        elif version == 2:
            eps = 0.025
        else: 
            raise Exception("ERROR: INVALID VERSION")


    if version == 1:
        return E_lmg / (sigma2_x*eps + sigma_n*sigma_n) 
    elif version == 2:
        return E_lmg / (sigma2_x*eps + sigma_x*sigma_n)
    else: 
        raise Exception("ERROR: INVALID VERSION")
