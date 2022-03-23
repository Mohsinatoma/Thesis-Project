# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:36:29 2022

@author: HP
"""



import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
import cv2
from empatches import EMPatches


large_image_stack = tiff.imread('E:/DRIVE/Training/images/21_training.tif')
large_mask_stack = cv2.imread('E:/DRIVE/Training/1st_manual/21_manual1.png')

large_mask_stack = cv2.imread('E:/Ground.jpg')


# = tiff.imread('E:/DRIVE/Training/1st_manual/21_manual1.png')

patches_img = patchify(large_mask_stack, (100,100,3), step=128)
print(patches_img.shape[0])
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        
        single_patch_img = patches_img[i,j,:,:]
        tiff.imwrite('E:/DRIVE/Training/patches/images/' + 'image_' + '_' + str(i)+str(j)+ ".tif", single_patch_img)

  
emp = EMPatches()

img_patches, indices = emp.extract_patches(large_mask_stack, patchsize=128, overlap=0)
print(img_patches)
for i in range(img_patches.shape[0]):
    for j in range(img_patches.shape[1]):
         
         single_patch_mask = img_patches[i,j,:,:]
         cv2.imwrite('E:/DRIVE/Training/1st_manual/patches/masks/' + 'mask_' + str(img_patches) + '_' + str(i)+str(j)+ ".tif", single_patch_mask)
        

"""
patches_mask = patchifyI(large_mask_stack, 128, 128)  #Step=256 for 256 patches means no overlap

for i in range(patches_mask.shape[0]):
    for j in range(patches_mask.shape[1]):
         
         single_patch_mask = patches_mask[i,j,:,:]
         cv2.imwrite('E:/DRIVE/Training/1st_manual/patches/masks/' + 'mask_' + str(patches_mask) + '_' + str(i)+str(j)+ ".tif", single_patch_mask)
         single_patch_mask = single_patch_mask / 255.
"""        
"""   

for msk in range(large_mask_stack.shape[0]):
     
    large_mask = large_mask_stack[msk]
    patches_mask = patchify(large_mask, (100, 100), step=100)  #Step=256 for 256 patches means no overlap

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            
            single_patch_mask = patches_mask[i,j,:,:]
            tiff.imwrite('E:/DRIVE/Training/1st_manual/patches/masks/' + 'mask_' + str(msk) + '_' + str(i)+str(j)+ ".tif", single_patch_mask)
            single_patch_mask = single_patch_mask / 255.
 """           