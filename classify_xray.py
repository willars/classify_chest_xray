
# coding: utf-8

# In[1]:


import shutil
import os
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam, RMSprop
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import  BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from tqdm import tqdm

import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_files       
from keras.utils import np_utils
from glob import glob
from pathlib import Path
import random


train_path = 'chest_xray/train/'
valid_path = 'chest_xray/val/'
test_path =  'chest_xray/test/'



train_normal_path = (train_path + 'NORMAL')
train_pneumonia_path = (train_path + 'PNEUMONIA')

valid_normal_path = (valid_path + 'NORMAL')
valid_pneumonia_path = (valid_path + 'PNEUMONIA')

test_normal_path = (test_path + 'NORMAL')
test_pneumonia_path = (test_path + 'PNEUMONIA')




for f in os.listdir(valid_normal_path):
    shutil.move(valid_normal_path + '/' + f, train_normal_path + '/' + f)


for f in os.listdir(valid_pneumonia_path):
    shutil.move(valid_pneumonia_path + '/' + f, train_pneumonia_path + '/' + f)

    
new_valid_normal = random.sample(os.listdir(train_normal_path), 50)
new_valid_pneumonia = random.sample(os.listdir(train_pneumonia_path), 50)

for f in new_valid_normal:
    shutil.move(train_normal_path + '/' + f, valid_normal_path + '/' + f)

for f in new_valid_pneumonia:
    shutil.move(train_pneumonia_path + '/' + f, valid_pneumonia_path + '/' + f)


# In[2]:


#train_path = 'chest_xray/train'
#valid_path = 'chest_xray/val'
#test_path =  'chest_xray/test'

#must figure out steps and batch sizes

# from image prep for cnn image classifier
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['NORMAL', 'PNEUMONIA'], batch_size = 30)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['NORMAL', 'PNEUMONIA'], batch_size = 10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['NORMAL', 'PNEUMONIA'], batch_size = 1)


# In[3]:


vgg16_model = keras.applications.vgg16.VGG16()


# In[4]:


vgg16_model.summary()


# In[5]:


#convert from model type
modelvgg16 = Sequential()
for layer in vgg16_model.layers:
    modelvgg16.add(layer)


# In[6]:


#remove the last dense layer dealing with all 1000 classes of imagenet
modelvgg16.layers.pop()


# In[7]:


modelvgg16.summary()


# In[8]:


#freeze layers
for layer in modelvgg16.layers[:20]:
    layer.trainable = False


# In[9]:


#for converting to binary classification
modelvgg16.add(Dense(2, activation='sigmoid'))


# In[10]:


modelvgg16.summary()


# In[ ]:


modelvgg16.compile(Adam(lr=.00001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


modelvgg16.fit_generator(train_batches, steps_per_epoch=100, validation_data=valid_batches, validation_steps=5, epochs=3, verbose=2, class_weight={0:1., 1:0.5})


# In[ ]:


modelvgg16.save('vgg16_xray_model.h5')


# # import compile and train vgg19

# In[ ]:


vgg19_model = keras.applications.vgg19.VGG19()


# In[ ]:


vgg19_model.summary()


# In[ ]:


#convert from model type
modelvgg19 = Sequential()
for layer in vgg19_model.layers:
    modelvgg19.add(layer)


# In[ ]:


modelvgg19.summary()


# In[ ]:


#remove the last dense layer dealing with all 1000 classes of imagenet
modelvgg19.layers.pop()


# In[ ]:


modelvgg19.summary()


# In[ ]:


#freeze layers
for layer in modelvgg19.layers[:21]:
    layer.trainable = False


# In[ ]:


modelvgg19.summary()


# In[ ]:


#for converting to binary classification
modelvgg19.add(Dense(2, activation='sigmoid'))


# In[ ]:


modelvgg19.summary()


# In[ ]:


modelvgg19.compile(Adam(lr=.00001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


modelvgg19.fit_generator(train_batches, steps_per_epoch=100, validation_data=valid_batches, validation_steps=5, epochs=3, verbose=2, class_weight={0:1., 1:0.5})


# In[ ]:


modelvgg19.save('vgg19_xray_model.h5')


# # Plot Confusion Matrix

# cm = confusion_matrix(test_labels, whole_predictions)

# # Running test data

# In[ ]:


# array of test labels

def get_test_labels():
    test_labels = []
    
    for files in os.listdir('chest_xray/test/NORMAL'):
        test_labels.append(0)

    for files in os.listdir('chest_xray/test/PNEUMONIA'):
        test_labels.append(1)

    return np.array(test_labels)

test_labels = get_test_labels()
print (len(test_labels))
print (test_labels)


# In[ ]:


#an array of image files

def get_test_image_files():
    test_image_files = []

    for files in os.listdir('chest_xray/test/NORMAL'):
        test_image_files.append(files)

    for files in os.listdir('chest_xray/test/PNEUMONIA'):
        test_image_files.append(files)

    return np.array(test_image_files)

test_image_files = get_test_image_files()
print (len(test_image_files))


# In[ ]:


from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions


# In[ ]:


from keras.preprocessing import image

def get_data(folder):
    X = []
    y = []
    Z = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['NORMAL']:
                label = 0
                #path = 'NORMAL'
            elif folderName in ['PNEUMONIA']:
                label = 1
                #path = 'PNEUMONIA'
            for image_filename in tqdm(os.listdir(folder + folderName)):
                # loads RGB image as PIL.Image.Image type
                img = image.load_img(folder + folderName + '/' + image_filename, target_size=(224, 224))
                # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
                x = image.img_to_array(img)
                # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
                img_arr = np.expand_dims(x, axis=0)
                
                path = folder + folderName + '/' + image_filename
                
                print img_arr
                
                X.append(img_arr)
                y.append(label)
                Z.append(path)
                
    X = np.asarray(X)            
    X = np.vstack(X)         
    y = np.asarray(y)
    Z = np.asarray(Z)
    
    return X,y,Z

X_test, y_test, Z_test= get_data(test_path)

print (X_test.shape)
print (X_test)
print (y_test)
print (Z_test)
print (type(Z_test))
print (Z_test.shape)


# In[ ]:


predictions_vgg16 = np.array(modelvgg16.predict(X_test))
print (predictions_vgg16)


# In[ ]:


print (predictions_vgg16.shape)


# In[ ]:


predictions_vgg19 = np.array(modelvgg19.predict(X_test))
print (predictions_vgg19)


# In[ ]:


predictions_vgg16_categorical = np.argmax(predictions_vgg16, axis=-1)
print (predictions_vgg16_categorical)


# In[ ]:



from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# In[ ]:


cm = confusion_matrix(y_test, predictions_vgg16_categorical)


# In[ ]:


#plot confusion matrix from scikit learn


from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


cm_plt_labels = ['NORMAL', 'PNEUMONIA']
plot_confusion_matrix(cm, cm_plot_labels, title ='Pneumonia Confusion Matrix')


# # VGG16 output and Test

# test as normal, and confusion matrix as normal

# test_imgs, test_labels = next(test_batches)

# test_labels = test_labels[:,0]
# test_labels

# predictionsvgg16 = modelvgg16.predict_generator(test_batches, steps=624, verbose=1, workers=0)
# predictionsvgg16

# predictionsvgg16.shape

# #manual find actual prediction
# predicted_class = np.argmax(predictionsvgg16, axis=-1)
# predicted_class

# # Simple Average Ensemble predictions

# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# test_normal_path = (test_path + 'NORMAL')
# test_pneumonia_path = (test_path + 'PNEUMONIA')

# submitted_files = np.array(glob("images/Submitted_images/*"))
# 
# dog_or_human(submitted_files)
