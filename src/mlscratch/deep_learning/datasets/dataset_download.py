import os
from pathlib import Path
import torchvision
import torchvision.transforms as transforms

from mlscratch.deep_learning.datasets.processing import image_preprocessing

train_transform = image_preprocessing(type='train')
test_transform = image_preprocessing(type='val')

def build_dataset(data_dir):
  # Download and load the training data
  train_dataset = torchvision.datasets.EMNIST(
      root=data_dir,
      split='balanced',
      train=True,
      download=True,
      transform=train_transform
  )

  # Download and load the test data
  test_dataset = torchvision.datasets.EMNIST(
      root=data_dir,
      split='balanced',
      train=False,
      download=True,
      transform=test_transform
  )
  
  return train_dataset, test_dataset