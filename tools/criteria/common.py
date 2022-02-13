import numpy as np
from skimage.metrics import structural_similarity as ssim


def PSNR(output, target):
    return 20*np.log10(np.abs(target).max() / np.sqrt(np.mean((output-target)**2)))
def MSE(output, target):
    return np.mean((output-target)**2)
def MAE(output, target):
    return np.mean(np.abs(output-target))
def NMSE(output, target):
    return np.mean((output-target)**2) / np.mean(target**2)
def NMAE(output, target):
    return np.mean(np.abs(output-target)) / np.mean(np.abs(target))
def SSIM(output, target):
    return ssim(target, output, data_range=output.max() - output.min())