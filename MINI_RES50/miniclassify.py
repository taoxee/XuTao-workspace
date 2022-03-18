# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 21:08:08 2022

@author: Tao
"""
import shutil
from collections import Counter
from shutil import copyfile
import sys
import copy
import os
import json
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def read_csv_classes(csv_dir: str, csv_name: str):
    data = pd.read_csv(os.path.join(csv_dir, csv_name))
    # print(data.head(1))  # filename, label

    label_set = set(data["label"].drop_duplicates().values)

    print("{} have {} images and {} classes.".format(csv_name,
                                                     data.shape[0],
                                                     len(label_set)))
    return data, label_set


def calculate_split_info(path: str, label_dict: dict, rate: float = 0.2):
    # path = data_dir
    # label_dict = label_dict
    # read all images
    image_dir = os.path.join(path, "images")
    images_list = [i for i in os.listdir(image_dir) if i.endswith(".jpg")]
    print("find {} images in dataset.".format(len(images_list)))

    train_data, train_label = read_csv_classes(path, "train.csv")
    val_data, val_label = read_csv_classes(path, "val.csv")
    test_data, test_label = read_csv_classes(path, "test.csv")

    # Union operation
    oglabels = (train_label | val_label | test_label)
    oglabels = list(oglabels)
    oglabels.sort()
    print("all classes: {}".format(len(oglabels)))
    
    # labels = oglabels
    
    ###select 10 classes from all data
    labels=np.random.choice(oglabels, size=10, replace=True, p=None)
    print ("selected classes",labels.shape)
    # print(labels)

    # create classes_name.json
    classes_label = dict([(label, [index, label_dict[label]]) for index, label in enumerate(labels)])
    # print(classes_label)
    json_str = json.dumps(classes_label, indent=4)
    with open('classes_name.json', 'w') as json_file:
        json_file.write(json_str)

    # concat csv data
    data = pd.concat([train_data, val_data, test_data], axis=0)
    print("total data shape: {}".format(data.shape))
      

    # split data on every classes
    num_every_classes = []
    split_train_data = []
    split_val_data = []
    for label in labels:
        class_data = data[data["label"] == label]
        num_every_classes.append(class_data.shape[0])

        # shuffle
        shuffle_data = class_data.sample(frac=1, random_state=1)
        num_train_sample = int(class_data.shape[0] * (1 - rate))
        split_train_data.append(shuffle_data[:num_train_sample])
        split_val_data.append(shuffle_data[num_train_sample:])

        # imshow
        imshow_flag = False
        if imshow_flag:
            img_name, img_label = shuffle_data.iloc[0].values
            img = Image.open(os.path.join(image_dir, img_name))
            plt.imshow(img)
            plt.title("class: " + classes_label[img_label][1])
            plt.show()

    # plot classes distribution
    plot_flag = False
    if plot_flag:
        plt.bar(range(1, 101), num_every_classes, align='center')
        plt.show()

    # concatenate data
    new_train_data = pd.concat(split_train_data, axis=0)
    new_val_data = pd.concat(split_val_data, axis=0)

    # save new csv data
    new_train_data.to_csv(os.path.join(path, "new_train.csv"))
    new_val_data.to_csv(os.path.join(path, "new_val.csv"))



def main():
    data_dir = "../mini-imagenet"  # 指向数据集的根目录
    json_path = "./imagenet_class_index.json"  # 指向imagenet的索引标签文件

    # load imagenet labels
    label_dict = json.load(open(json_path, "r"))
    label_dict = dict([(v[0], v[1]) for k, v in label_dict.items()])

    calculate_split_info(data_dir, label_dict)
    # print(calculate_split_info)

if __name__ == '__main__':
    main()

#copy selected train_imgs to new folder
new_train_1= pd.read_csv('../mini-imagenet/new_train.csv')
filename1=new_train_1['filename']
print(filename1)
# def remove_file(old_path, new_path):
    
for i in list(filename1):
    src1 = os.path.join("../mini-imagenet/images", i)
    dst1 = os.path.join('../mini-imagenet/train', i)
    print('src1:', src1)
    print('dst1:', dst1)
    shutil.copyfile(src1, dst1)

#copy seleted test_imgs to newfolder
new_test_1= pd.read_csv('../mini-imagenet/new_val.csv')
filename2=new_test_1['filename']
print(filename2)
# def remove_file(old_path, new_path):
    
for i in list(filename2):
    src2 = os.path.join("../mini-imagenet/images", i)
    dst2 = os.path.join('../mini-imagenet/test', i)
    print('src2:', src2)
    print('dst2:', dst2)
    shutil.copyfile(src2, dst2)