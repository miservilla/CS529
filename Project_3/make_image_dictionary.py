import os
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.feather as ft
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import datasets

rootdir = '/home/michaelservilla/CS529/Project_3/train (copy)'
train = []
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
        # img.save(path)
        w, h = img.size
        if w != h:
            print(path)
            print(w, h)
            print()
        new_img = {'img': feature.encode_example(img)}
        # img = cv2.imread(path)
        # cv2.imshow('OutputImage', img)
        # cv2.waitKey(0)
        train.append([new_img,dir])
        print(count, w, h)

# print(train_small)

# for i, j in train.items():
#     print(i, j)

train = pd.DataFrame(train, columns=['img', 'label'])
print(train)
train.to_pickle("./Project_3/plants_df")

# plants = ds.dataset(pa.Table.from_pandas(train))

# plants.save_to_disk('plants')
# print(plants)

# hg_plants = ds.dataset(pa.Table.from_pandas(train).to_batches())
# print(type(hg_plants))

# ds.write_dataset(hg_plants, './Project_3/plants')

# hg_plants.save_to_disk('plants')

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
