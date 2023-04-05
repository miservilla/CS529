import numpy as np
from datasets import load_metric
from transformers import ViTFeatureExtractor
import datasets
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from numpy import moveaxis
from numpy import asarray
from PIL import Image
import cv2

dataset = load_dataset("imagefolder", data_dir="/home/michaelservilla/CS529/Project_3/train",
                       split='train').train_test_split(test_size=0.2)
print(dataset.shape)
dataset_train = dataset['train']
dataset_test = dataset['test']

num_classes = len(set(dataset_train['label']))
labels = dataset_train.features['label']
print(num_classes, labels)

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


def preprocess(batch):
  inputs = feature_extractor(batch['image'], return_tensors='pt')
  inputs['label'] = batch['label']
  return inputs

prepared_train = dataset_train.with_transform(preprocess)
prepared_test = dataset_test.with_transform(preprocess)

training_subset = {}
v = 0

for i in range(dataset_train.shape[0]):

    try:
        training_subset[i-v] = prepared_train[i+v]
    except:
        print(i)
        v += 1

print(training_subset.keys())

testing_subset = {}
u = 0

for i in range(dataset_test.shape[0]):

    try:
        testing_subset[i-u] = prepared_test[i]
    except:
        print(i)
        u += 1

print(testing_subset.keys())

def collate_fn(batch):
    return {'pixel_values': torch.stack([x['pixel_values'] for x in batch]), 'labels': torch.tensor([x['label'] for x in batch])}

from huggingface_hub import evaluate

metric = evaluate.load("accuracy")


def compute_metrics(p):
  return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

