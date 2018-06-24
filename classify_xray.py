import shutil
import os
import numpy as np
from sklearn.datasets import load_files       
from keras.utils import np_utils
from glob import glob
from pathlib import Path

val_normal_dir = ('chest_xray/val/NORMAL/')
val_pneumonia_dir = ('chest_xray/val/PNEUMONIA/')


train_normal_dir = 'chest_xray/train/NORMAL/'
train_pneumonia_dir = 'chest_xray/train/PNEUMONIA/'

files_normal = os.listdir(val_normal_dir)

for f in files_normal:
    shutil.move(val_normal_dir + f, train_normal_dir + f)

files_pneumonia = os.listdir(val_pneumonia_dir)

for f in files_pneumonia:
    shutil.move(source_pneumonia + f, train_pneumonia_dir + f)



data_dir = Path('chest_xray')

train_dir = data_dir / 'train'

normal_cases_dir = train_dir / 'NORMAL'

normal_cases = normal_cases_dir.glob('*.jpeg')



ls_of_normal_cases = []

for i in normal_cases:
    ls_of_normal_cases.append(i)

normal_cases = np.array(ls_of_normal_cases)



import cv2
from keras.preprocessing import image 

from tqdm import tqdm

#files_normal = os.listdir(normal_cases)
#files_pneumonia = os.listdir(train_pneumonia_dir)

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    print (x.shape)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    y = np.expand_dims(x, axis=0)
    print (y.shape)
    return y

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    z = np.vstack(list_of_tensors)
    print (z.shape)
    return z


train_tensors = paths_to_tensor(normal_cases).astype('float32')/255
