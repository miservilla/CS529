import os
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import datasets

rootdir = '/home/michaelservilla/CS529/Project_3/train'
train_small = []
bad = []
count = 0
feature = datasets.Image(decode=False)
size = (32, 32)

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        count += 1
        path = subdir + '/' + file
        dir = subdir.split('/')[-1]
        img = Image.open(path)
        # img = img.resize((32, 32))
        img.save(path)
        w, h = img.size
        if w != h:
            print(path)
            print(w, h)
            print()
        new_img = {'img': feature.encode_example(img)}
        # img = cv2.imread(path)
        # cv2.imshow('OutputImage', img)
        # cv2.waitKey(0)
        train_small.append([new_img,dir])
        print(count, w, h)

# print(train_small)

# for i, j in train.items():
#     print(i, j)

train_small = pd.DataFrame(train_small, columns=['img', 'label'])
print(train_small)

# plants = ds.dataset(pa.Table.from_pandas(train))

# plants.save_to_disk('plants')
# print(plants)

# hg_plants_small = ds.dataset(pa.Table.from_pandas(train_small).to_batches())
# # print(type(hg_plants_small))

# hg_plants_small.save_to_disk('plants_small')

# df_train, df_test = train_test_split(train, test_size=0.2)

# dataset_train = ds.dataset(pa.Table.from_pandas(df_train))
# print(df_train)
# print(dataset_train)

# train = Dataset(pa.Table.from_pandas(df_train))
# print(type(train))

# dataset_test = ds.dataset(pa.Table.from_pandas(df_test))
# print(df_test)
# print(dataset_test)

# test = Dataset(pa.Table.from_pandas(df_test))
# print(type(test))
