import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import cv2
import matplotlib.pyplot as plt
import random



class IntelDataset(torch.utils.data.Dataset):  # inheritin from Dataset class
    def __init__(self, csv_file, root_dir="", transform=None):
        self.annotation_df = pd.read_csv(csv_file)
        # root directory of images, leave "" if using the image path column in the __getitem__ method
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # return length (numer of rows) of the dataframe
        return len(self.annotation_df)

    def __getitem__(self, idx):
        # use image path column (index = 1) in csv file
        image_path = os.path.join(
            self.root_dir, self.annotation_df.iloc[idx, 1])
        image = cv2.imread(image_path)  # read image by cv2
        # convert from BGR to RGB for matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # use class name column (index = 2) in csv file
        class_name = self.annotation_df.iloc[idx, 2]
        # use class index column (index = 3) in csv file
        class_index = self.annotation_df.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        return image, class_name, class_index

def build_csv(directory_string, output_csv_name):
    import csv
    directory = directory_string
    class_lst = os.listdir(directory)
    class_lst.sort()
    with open(output_csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['filename', 'file_path', 'class_name', 'class_index'])
        for class_name in class_lst:
            class_path = os.path.join(directory, class_name)
            file_lst = os.listdir(class_path)
            for file_name in file_lst:
                file_path = os.path.join(directory, class_name, file_name)
                writer.writerow([file_name, file_path, class_name, class_lst.index(class_name)])
    return


train_folder = '/home/michaelservilla/CS529/Project_3/data/seg_train'
test_folder = '/home/michaelservilla/CS529/Project_3/data/seg_test'

build_csv(train_folder, 'train.csv')
train_df = pd.read_csv('train.csv')

df_train, df_test = train_test_split(train_df, test_size=0.2)
df_train = df_train.sort_values(by=['class_index'])
df_test = df_test.sort_values(by=['class_index'])
df_train.to_csv('./Project_3/train.csv', index=False)
df_test.to_csv('./Project_3/test.csv', index=False)

class_names = list(df_train['class_name'].unique())

train_dataset_untransformed = IntelDataset(
    csv_file='./Project_3/train.csv', root_dir="", transform=None)

plt.figure(figsize=(12, 6))

image, class_name, class_index = train_dataset_untransformed[0]


for i in range(10):
    idx = random.randint(0, len(train_dataset_untransformed))
    image, class_name, class_index = train_dataset_untransformed[idx]
    ax = plt.subplot(2, 5, i+1)  # create an axis
    # create a name of the axis based on the img name
    ax.title.set_text(class_name + '-' + str(class_index))
    plt.imshow(image)  # show the img
plt.show()

