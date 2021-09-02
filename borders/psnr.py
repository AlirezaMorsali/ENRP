import numpy as np
import cv2

inImage = cv2.imread('in.jpg')
outImage = cv2.imread('out.jpg')

diff = inImage - outImage
meanS = np.mean(diff**2)
border_margin = 7
meanSb = (np.mean(diff[128-border_margin:130+border_margin,:]**2) + np.mean(diff[:, 128-border_margin:130+border_margin]**2))/2
PIXEL_MAX = 255.0
psnrS = 20 * np.log10(PIXEL_MAX / np.sqrt(meanS))
psnrSb = 20 * np.log10(PIXEL_MAX / np.sqrt(meanSb))
print(f'psnr total: {psnrS}, psnr borders: {psnrSb}')
