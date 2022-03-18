# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 20:31:10 2022

@author: Tao
"""
import shutil
import pandas as pd
import numpy as np
import os
import tarfile
import pandas as pd

#prepare parameters
##original data direction
orig_dir='X:/Python/ILSVRC-2012'
##target training date direction
dest_train_dir = './ILSVRC2012_img_train'
##target valid date direction
dest_val_dir = './ILSVRC2012_img_val'

#read txt file
list_class=pd.read_csv('X:/Python/ILSVRC-2012/caffe_ilsvrc12/synsets.txt',sep='\n',names=['class_dir'])

# with open('X:/Python/ILSVRC-2012/caffe_ilsvrc12/synsets.txt') as f:
#     classframe = f.readlines()
#     print(classframe)
#     f.close()

#2 random select 10 classes
'''
select_class = np.random.choice(list_class["class_dir"], size=10, replace=False, p=None)
print ("selected classes",select_class)
'''

select_class = np.array(['n01986214', 'n02086240', 'n02132136', 'n02138441', 'n02396427','n03444034', 'n03837869', 'n04033995', 'n04330267', 'n07873807'])



'''
#2.1copy 10 valid to new folder

for i in list(select_class):
    src = os.path.join("X:/Python/ILSVRC-2012/ILSVRC2012_img_val", i)
    dst = os.path.join('X:/Python/ILSVRC-2012/valid', i)
    dst1=dst.replace("\\", "/")
    src1=src.replace("\\", "/")
    print('src1:', src1)
    print('dst1:', dst1)
    shutil.copytree(src1, dst1)
'''

#4 copy 10 train classes  
for i in list(select_class):
    print(i)

    #unzip train
    zipfile=os.path.join('X:/Python/ILSVRC-2012/ILSVRC2012_img_train', i+".tar")
    tar = tarfile.open(zipfile)
    tar.extractall(os.path.join('X:/Python/ILSVRC-2012/train',i))
    print(i)
    tar.close()

#choose 8:2files
# files=os.listdir('X:/Python/ILSVRC-2012/train')
# filespath=os.path.join('X:/Python/ILSVRC-2012/train',files)

rate = 0.5
path_train = "X:/Python/ILSVRC-2012/train"
path_val = "X:/Python/ILSVRC-2012/valid"
for i in list(select_class):
    k=0
    
    select_file=os.listdir(os.path.join(path_train,i))
    print(len(select_file))
    select_img = np.random.choice(select_file, size=int(len(select_file)*rate),replace=False)#int(len(select_file)*rate)
    print(len(select_img))
    # train_dir=os.listdir('X:/Python/ILSVRC-2012/train')
    for img in select_img:
        print(img)
        src = os.path.join(path_train, i,img)
        print(os.path.exists(src))
        dst_path = os.path.join(path_val, i)
        os.makedirs(dst_path,exist_ok = True)
        dst = os.path.join(dst_path,img)
        dst1=dst.replace("\\", "/")
        src1=src.replace("\\", "/")
        shutil.move(src1, dst1)
        # print(k)
        # k+=1
# close()c
print('file move done')

        
# =============================================================================
#         
#     
# 
#     #5 claculate the number with the rate
#     num_train_img = len(list_unzip_clas)*(1-rate)
#     
#     #6 random select the items with a num of num_train_img
#     slt_train_img = 
#     
#     
#     #copy EACH image into the new folder, if you want to deal with EACH image, you may need for loop
#     #we check each images in the class, if it is inside the slt_train_img, we copy it to train folder,
#     #otherwise, we copy it to valid folder
#     for img in list_unzip_class:
#         #read the source path
#         src = os.path.join(orig_dir, img)
#         #if it is inside the slt_train_img
#         if img inslt_train_img:
#             #we set the destination path as following       
#             dst = 
#             
#         #otherwise 
#         else:
#             #we set the destination as following 
#             dst = 
#         #we mmove the image to the destination folder
#         shutil.copyfile(src, dst)
# 
# #DONE
# =============================================================================
    





