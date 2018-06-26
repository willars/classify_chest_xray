
# coding: utf-8

# In[1]:


import shutil
import os
import numpy as np
from sklearn.datasets import load_files       
from keras.utils import np_utils
from glob import glob
from pathlib import Path
import random


train_dir = 'chest_xray/train/'
val_dir = 'chest_xray/val/'
test_dir =  'chest_xray/test/'



train_normal_dir = (train_dir + 'NORMAL')
train_pneumonia_dir = (train_dir + 'PNEUMONIA')

val_normal_dir = (val_dir + 'NORMAL')
val_pneumonia_dir = (val_dir + 'PNEUMONIA')

test_normal_dir = (test_dir + 'NORMAL')
test_pneumonia_dir = (test_dir + 'PNEUMONIA')




for f in os.listdir(val_normal_dir):
    shutil.move(val_normal_dir + '/' + f, train_normal_dir + '/' + f)


for f in os.listdir(val_pneumonia_dir):
    shutil.move(val_pneumonia_dir + '/' + f, train_pneumonia_dir + '/' + f)

    
new_val_normal = random.sample(os.listdir(train_normal_dir), 50)
new_val_pneumonia = random.sample(os.listdir(train_pneumonia_dir), 50)


for f in new_val_normal:
    shutil.move(train_normal_dir + '/' + f, val_normal_dir + '/' + f)

for f in new_val_pneumonia:
    shutil.move(train_pneumonia_dir + '/' + f, val_pneumonia_dir + '/' + f)


# In[ ]:


from keras.utils import np_utils
from tqdm import tqdm
import cv2
import skimage
from skimage.transform import resize


def get_data(folder):
    X = []
    y = []
    for folderName in os.listdir(folder):
        if folderName in ['NORMAL']:
            label = 0
        elif folderName in ['PNEUMONIA']:
            label = 1
        for image_filename in os.listdir(folder + folderName):
            img_file = cv2.imread(folder + folderName + '/' + image_filename)
            if img_file is not None:
                img_file = skimage.transform.resize(img_file, (150, 150, 3))
                img_arr = np.asarray(img_file)
                X.append(img_arr)
                y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y
X_train, y_train = get_data(train_dir)
X_test, y_test= get_data(test_dir)
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
y_trainHot = to_categorical(y_train, num_classes = 2)
y_testHot = to_categorical(y_test, num_classes = 2)


# put jpeg files into array with labels

# In[2]:


normal_cases = Path(train_normal_dir).glob('*.jpeg')
pneumonia_cases = Path(train_pneumonia_dir).glob('*.jpeg')


# In[3]:


import cv2
from keras.preprocessing import image 
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    z = np.vstack(list_of_tensors)
    return np.vstack(list_of_tensors)

for f in os.listdir(train_normal_dir):
    train_tensors = paths_to_tensor(normal_cases).astype('float32')/255


# In[ ]:


def data_label_set(folder):
    ls_data = []
    ls_label = []
    for i in folder:
        ls_data.append(i)
        if folder == 'NORMAL':
            ls_label.append(0)
        if folder == 'PNEUMONIA':
            ls_label.append(1)
    return ls_data, ls_label


# In[ ]:


from glob import glob

#data_dir = Path('chest_xray')
#train_dir = data_dir / 'train'
#normal_cases_dir = train_dir / 'NORMAL'
normal_cases = train_normal_dir.glob('*.jpeg')


# 
# ls_of_normal_cases = []
# 
# for i in normal_cases:
#     ls_of_normal_cases.append(i)
# 
# normal_cases = np.array(ls_of_normal_cases)
# 
# 
