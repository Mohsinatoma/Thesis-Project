
import numpy as np
np.random.seed(1000)


import os
import matplotlib.pyplot as plt


os.environ['KRAS_BACKEND'] = 'tersorflow'

dataset = []
label = []


class resultploting():

    def __init__(self, history= None):
        
        
        # properties
  
        self.history = history
        

    def result(self):

    
        

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        t = f.suptitle('CNN Performance', fontsize=12)
        f.subplots_adjust(top=0.85, wspace=0.3)


        max_epoch = len(self.history.history['accuracy'])+1
        epoch_list = list(range(1,max_epoch))
        ax1.plot(epoch_list, self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(epoch_list, self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_xticks(np.arange(1, max_epoch, 5))
        ax1.set_ylabel('Accuracy Value')
        ax1.set_xlabel('Epoch')
        ax1.set_title('Accuracy')
        l1 = ax1.legend(loc="best")

        ax2.plot(epoch_list, self.history.history['loss'], label='Train Loss')
        ax2.plot(epoch_list, self.history.history['val_loss'], label='Validation Loss')
        ax2.set_xticks(np.arange(1, max_epoch, 5))
        ax2.set_ylabel('Loss Value')
        ax2.set_xlabel('Epoch')
        ax2.set_title('Loss')
        l2 = ax2.legend(loc="best")
       
            
       




