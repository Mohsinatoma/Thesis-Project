# -*- coding: utf-8 -*-
"""


@author: Mohoshina Toma
"""
from keras.utils import plot_model 
import tensorflow as tf


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
count = 0


class model():
    """
     CNN architecture desceibe
     #k_initializer = starting weight for nodes ; he_normal= guassian distribution
    """
    
    def __init__(self, dimentionofFilter= (3,3), act='relu' , k_initializer='he_normal', pad='same' ):
        
        
        # properties
  
        self.dimentionofFilter = dimentionofFilter
        self.act = act
        self.k_initializer = k_initializer
        self.pad = pad


    def encoder_block(self, inputs, num_filters):
        """
        

        Parameters
        ----------
        inputs : TYPE
            DESCRIPTION.
        num_filters : TYPE
            DESCRIPTION.

        Returns
        -------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        """
   
        #conv layers
        x = tf.keras.layers.Conv2D(num_filters, self.dimentionofFilter, activation=self.act, kernel_initializer= self.k_initializer, padding=self.pad)(inputs)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Conv2D(num_filters, self.dimentionofFilter, activation=self.act, kernel_initializer= self.k_initializer, padding=self.pad)(x)
        #pooling
        y = tf.keras.layers.MaxPooling2D((2, 2))(x)
        

        return x, y
   
    def decoder_block(self, inputs, concated_conv, num_filters, ax):
        x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs)
      
        if ax == 0:
          
            x = tf.keras.layers.concatenate([x, concated_conv])
        else:
        
            x = tf.keras.layers.concatenate([x, concated_conv], axis= ax)
        y = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        y = tf.keras.layers.Dropout(0.2)(y)
        y = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(y)
        
       
        return y
   
def build_unet():
    cnnmodel= model()
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    c1, p1 = cnnmodel.encoder_block(s, 16)
    c2, p2 = cnnmodel.encoder_block(p1, 32)
    c3, p3 = cnnmodel.encoder_block(p2, 64)
    c4, p4 = cnnmodel.encoder_block(p3, 128)
    c5, p5 = cnnmodel.encoder_block(p4, 256)
    
    
    
    c6 = cnnmodel.decoder_block(c5, c4, 128, 0)
    c7 = cnnmodel.decoder_block(c6, c3, 64, 0)
    c8 = cnnmodel.decoder_block(c7, c2, 32, 0)
    c9 = cnnmodel.decoder_block(c8, c1, 16, 3)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    modelUnet = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    modelUnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    modelUnet.summary()
    
   # FeatureSet = tf.keras.layers.Conv2D(1, (1, 1), activation='relu')(c9)
   # modelforFeature = tf.keras.Model(inputs=[inputs], outputs=[FeatureSet])
    return outputs

if __name__ == "__main__":
    build_unet() 



