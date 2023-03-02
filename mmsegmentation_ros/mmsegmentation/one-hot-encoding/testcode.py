from cgi import test
from logging import raiseExceptions
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from tqdm import tqdm


# Directories
print("1")
root_data_dir = "/home/avalocal/Desktop/data for segmentation"

img_dir_train =os.path.join(root_data_dir, "leftImg8bit/train" )
img_dir_val=os.path.join(root_data_dir, "leftImg8bit/val")

label_train_path=os.path.join(root_data_dir, "gtFine/train" )
label_val_path=os.path.join(root_data_dir, "gtFine/val")


def dirfiles(d, pattern="*"):
    return glob.glob(os.path.join(d, pattern))


folders_train=dirfiles(img_dir_train)
folders_val=dirfiles(img_dir_val)
print(folders_train)  
print("2")



list_files1=[]
list_files2=[]


for folder in folders_train:

    img_files_train = dirfiles(folder, "*.png")
    for file in img_files_train:
       f=file.replace("_leftImg8bit","_gtFine_color")
       f=f.replace("leftImg8bit","gtFine")
       list_files1.append(f)


for folder in folders_val:

    img_files_val = dirfiles(folder, "*.png")
    for file in img_files_val:
       f=file.replace("_leftImg8bit","_gtFine_color")
       f=f.replace("leftImg8bit","gtFine")
       list_files2.append(f)
       
       


print('number of training images: ', len(list_files1))    
print('number of val images : ', len(list_files2)) 
count=0
dir1="/home/avalocal/Desktop/par/mmsegmentation/data/ann_dir"
dir_save_train=os.path.join(dir1, "train")
dir_save_val=os.path.join(dir1, "val")


for file in tqdm(list_files1):
    print(file)

    binary_label=[]
    maska= Image.open(file)
    mask =maska.convert('RGB')
    label_array = np.asarray(mask)
    road_label = np.array([128,64,128])

    binary_label = (np.all(label_array==road_label, axis=2)).astype(np.uint8)
    binary_label = (1-np.all(label_array!=road_label, axis=2)).astype(np.uint8)
    #print(binary_label.shape) #1024*2048
    finalImg=Image.fromarray(binary_label)
    name=os.path.basename(file)
    #print(name) : darmstadt_000063_000019_gtFine_color.png
    path=os.path.join(dir_save_train, name)
    #print(path) : /home/pardis/Desktop/road_mmsegmentation/binary/train/darmstadt_000063_000019_gtFine_color.png
    finalImg.save(str(path)) 
    #print(str(path))


for file in tqdm(list_files2):
    print(file)
    binary_label=[]
    maska= Image.open(file)
    mask =maska.convert('RGB')
    label_array = np.asarray(mask)
    #print(label_array.shape) #1024*2048*3
    # Current class label encoding
    road_label = np.array([128,64,128])

    binary_label = (np.all(label_array==road_label, axis=2)).astype(np.uint8)
    binary_label = (1-np.all(label_array!=road_label, axis=2)).astype(np.uint8)
    #print(binary_label.shape) #1024*2048
    finalImg=Image.fromarray(binary_label)
     
    name=os.path.basename(file)
    #print(name) : darmstadt_000063_000019_gtFine_color.png
    path=os.path.join(dir_save_val, name)
    #print(path) : /home/pardis/Desktop/road_mmsegmentation/binary/train/darmstadt_000063_000019_gtFine_color.png
    finalImg.save(str(path)) 
   
    #print(str(path))'''
