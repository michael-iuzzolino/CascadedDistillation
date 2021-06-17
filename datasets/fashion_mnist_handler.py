"""CIFAR10/100 loader."""
import copy
import json
import numpy as np
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
from datasets import noise


class FashionMNISTHandler(torchvision.datasets.FashionMNIST):
  """FashionMNIST dataset handler."""

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    img = np.asarray(img)
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target


def get_transforms(dataset_key,
                   mean,
                   std,
                   noise_type=None,
                   noise_transform_all=False):
  """Create dataset transform list."""
  if dataset_key == 'train':
    transforms_list = [
        T.RandomCrop(32, padding=4, padding_mode='reflect'),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]
  else:
    transforms_list = [
        T.ToTensor(),
        T.Normalize(mean, std),
    ]

  if (noise_type is not None
      and (dataset_key == 'train' or noise_transform_all)):
    transforms_list.append(noise.NoiseHandler(noise_type))

  transforms = T.Compose(transforms_list)

  return transforms


def split_dev_set(dataset_src, mean, std, dataset_len,
                  val_split, split_idxs_root, load_previous_splits):
  """Load or create (and save) train/val split from dev set."""
  # Compute number of train / val splits
  n_val_samples = int(dataset_len * val_split)
  n_sample_splits = [dataset_len - n_val_samples, n_val_samples]

  # Split data
  train_set, val_set = torch.utils.data.random_split(dataset_src,
                                                     n_sample_splits)

  train_set.dataset = copy.copy(dataset_src)
  val_set.dataset.transform = get_transforms('test', mean, std)

  # Set indices save/load path
  val_percent = int(val_split * 100)
  if '.json' not in split_idxs_root:
    idx_filepath = os.path.join(
        split_idxs_root, f'{val_percent}-{100-val_percent}_val_split.json')
  else:
    idx_filepath = split_idxs_root

  # Check load indices
  if load_previous_splits and os.path.exists(idx_filepath):
    print(f'Loading previous splits from {idx_filepath}')
    with open(idx_filepath, 'r') as infile:
      loaded_idxs = json.load(infile)

    # Set indices
    train_set.indices = loaded_idxs['train']
    val_set.indices = loaded_idxs['val']

  # Save idxs
  else:
    if not os.path.exists(split_idxs_root):
      os.makedirs(split_idxs_root)
      
    print(f'Saving split idxs to {idx_filepath}...')
    save_idxs = {
        'train': list(train_set.indices),
        'val': list(val_set.indices),
    }

    # Dump to json
    with open(idx_filepath, 'w') as outfile:
      json.dump(save_idxs, outfile)

  # Print
  print(f'{len(train_set):,} train examples loaded.')
  print(f'{len(val_set):,} val examples loaded.')

  return train_set, val_set


def set_dataset_stats():
  """Set dataset stats for normalization given dataset."""
  mean = (0.2519)
  std = (0.3440)
  return mean, std


def build_dataset(root,
                  dataset_key,
                  mean,
                  std,
                  val_split=None,
                  split_idxs_root=None,
                  load_previous_splits=True,
                  noise_type=None,
                  noise_transform_all=False):
  """Build dataset."""
  print(f'Loading {dataset_key} data...')

  # Transforms
  transforms = get_transforms(dataset_key, mean, std,
                              noise_type, noise_transform_all)

  # Build dataset source
  dataset_src = FashionMNISTHandler(root=root,
                                    train=dataset_key == 'train',
                                    transform=transforms,
                                    target_transform=None,
                                    download=True)

  # Get number samples in dataset
  dataset_len = dataset_src.data.shape[0]

  # Split
  if dataset_key == 'train':
    if val_split:
      dataset_src = split_dev_set(dataset_src,
                                  mean,
                                  std,
                                  dataset_len,
                                  val_split,
                                  split_idxs_root,
                                  load_previous_splits)
    else:
      dataset_src = dataset_src, None

  # Stdout out
  print((f'{dataset_len:,} '
         f'{"dev" if dataset_key=="train" else dataset_key} '
         f'examples loaded.'))

  return dataset_src


def create_datasets(root,
                    val_split,
                    load_previous_splits=False,
                    split_idxs_root=None,
                    noise_type=None):
  """Create train, val, test datasets."""

  # Set stats
  mean, std = set_dataset_stats()

  # Build datasets
  train_dataset, val_dataset = build_dataset(
      root,
      dataset_key='train',
      mean=mean,
      std=std,
      val_split=val_split,
      split_idxs_root=split_idxs_root,
      load_previous_splits=load_previous_splits,
      noise_type=noise_type)

  test_dataset = build_dataset(root,
                               dataset_key='test',
                               mean=mean,
                               std=std,
                               noise_type=noise_type,
                               load_previous_splits=load_previous_splits)

  # Package
  dataset_dict = {
      'train': train_dataset,
      'val': val_dataset,
      'test': test_dataset,
  }

  return dataset_dict
