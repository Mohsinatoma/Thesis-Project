# -*- coding: utf-8 -*-
"""
@author: Mohoshina Toma
"""


import numpy as np
import tensorflow as tf

import cv2
import os

import glob
from skimage.transform import resize
from sklearn.utils import shuffle
from skimage.io import imread 
from patchify import patchify
from PIL import Image, ImageDraw, ImageFont 
import tifffile as tiff


os.environ['KRAS_BACKEND'] = 'tersorflow'

X_train = []
Y_train = []
X_test = []
mask =[]

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3



class ImageProcessing():
    """
    Image will be resized , processed and stored into a numpy array.
    properties:
        image_dir = string
    """


    def __init__(self, image_dir=None,height= None, width=None, training=1):
        
        
        # properties
        self.image_dir = image_dir
        self.height = height
        self.width = width
        self.x = None
        self.y = None
        self.training = training
        """
        name= glob.glob('E:\DRIVE\Test\images\*')
        print(name)
        """
        
        if training == 1:
            imag_path = os.path.join(self.image_dir, "images", "*" )
            self.x = glob.glob(imag_path)
            self.y = glob.glob(os.path.join(self.image_dir, "1st_manual", "*"))
            
        else:

            # return all the file of matched pattern
            imag_path = os.path.join(self.image_dir, "images", "*" )
            #print(imag_path)
            self.x = glob.glob(imag_path)
            #print(self.x)
 
     
        
    def git_to_png(self):
        for n, image_name in enumerate(self.y):
            img = Image.open(image_name)
            image_name = image_name.split('.')[0]
            mask = img.save(image_name+".png",'png', optimize=True, quality=70)
            print(mask)
        
        return
    def git_to_tif(self):
        for n, image_name in enumerate(self.y):
            img = Image.open(image_name)
            image_name = image_name.split('.')[0]
            mask = img.save(image_name+".tif",'tif', optimize=True, quality=70)
            print(mask)
        
        return
        
    def shuffling(self):
       image, mask = shuffle(self.x, self.y, random_state=42)
       return image, mask
   
    def normalize(self,input_image, input_mask):
         input_image = tf.cast(input_image, tf.float32) / 255.0
         input_mask -= 1
         return input_image, input_mask
    
    def resizeStore(self,train_x=None , train_y= None):
        """
        input: a png image from database
        output: a numpy array
        
        """
       
       
        #X_train = np.zeros(( IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        #Y_train = np.zeros(( IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

        print('Resizing training images and masks\n')
       # print(self.training)
        if self.training == 1:
           
            for n, image_name in enumerate(train_x):

                  x = cv2.imread(image_name, cv2.IMREAD_COLOR)
                  x = cv2.resize(x, (self.width, self.height))
                  x = x/255.0
                  x = x.astype(np.float32)
                  X_train.append(np.array(x))
                  
             
            for n, image_name in enumerate(train_y):    
                  y = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
                  y = cv2.resize(y, (self.width, self.height))
                  y = x/255.0
                  y = x.astype(np.float32)
                  Y_train.append(np.array(y))
                  
                  
           
            return (X_train, Y_train)
        
   
            
            
        elif self.training == 0:
              
                #print(self.x)
                #print('Resizing test images') 
                for n, image in enumerate(self.x):
                  #  print(image)
                    img = cv2.imread(image)
                    img = resize(img, (self.height,self.width), mode='constant', preserve_range=True)
                    X_test.append(np.array(img))
                    
                return (X_test)
           
      
 
        
    def imagepatching(image,path, size):
        patches_img = patchify(image, (size,size,3), step=size)
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                
                single_patch_img = patches_img[i,j,:,:]
                tiff.imwrite(path + '_' + str(i)+str(j)+ ".tif", single_patch_img)

        return image
    
        
    
    def dataAugmentation(self,image, mask ):
        """
        input: image and mask
        output: augmentated image
        
        """
        from keras.preprocessing.image import ImageDataGenerator

      
        train_datagen = ImageDataGenerator(rotation_range=45,  
            width_shift_range=0.3,  #try 0.1, 0.3
            height_shift_range=0.3, #try 0.1, 0.3
            zoom_range = 0.3, #try 0.1, 0.3
            vertical_flip=True,
            horizontal_flip = True,
            fill_mode="reflect")
        
        print(np.asarray(image).shape)  
        print(np.asarray(mask).shape)
        train_generator = train_datagen.flow(
            image,
            mask,
            batch_size = 16)  #images to generate in a batch
        
        #NOTE: When we use fit_generator, the number of samples processed 
        #for each epoch is batch_size * steps_per_epochs. 
        #should typically be equal to the number of unique samples in our 
        #dataset divided by the batch size.

        
        
        return train_generator
    
    



"""
image = input("Enter the Location of First image:\n") 
mask = input("Enter the Location of Second Class:\n")
heightofimage = input("Enter the size of the image:\n")
widthofimage = input("Enter the size of the image:\n")
sizeofimage = input("Enter the size of the image:\n")
"""
def image_pre_processing_for_training():
    
    image_path = "E:/DRIVE/Training/1st_manual/21_manual1.png"
   
    I1 = ImageProcessing(image_path,int(128),int(128), int(1))
    I1.git_to_tif()
    print("done")
   # train_x, train_y = I1.shuffling()
   
  #  X_train_data, Y_train_data =I1.resizeStore(train_x, train_y)
   
    """
    for item in X_train_data:
        print(item)
        if np.asarray(item).shape!=(20, 128, 128, 3):
           # X_train_data.remove(item)
           count = count+1
    print(count)
    
    for item in Y_train_data:
        if item.shape!=(20, 128, 128, 3):
            #Y_train_data.remove(item)
    print(len(Y_train_data))
   """ 

   
  #  train_gen = I1.dataAugmentation(X_train_data,Y_train_data)
    #return X_train_data, Y_train_data, train_gen

    #I1.git_to_png()
def image_pre_processing_for_testing():
    
    image_path = "E:/DRIVE/Test/"
    I1 = ImageProcessing(image_path,int(128),int(128), int(0))
    X_train_data =I1.resizeStore()
    return X_train_data

   


if __name__ == "__main__":
    image_pre_processing_for_training()  
    image_pre_processing_for_testing()