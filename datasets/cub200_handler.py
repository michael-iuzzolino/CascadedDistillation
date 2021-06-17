# https://github.com/fastai/imagenet-fast/blob/master/imagenet_nv/fastai_imagenet.py#L128
import glob
import scipy.io
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import noise


class EnsureShape:
  def __init__(self, n_channels=3):
    self._n_channels = n_channels
    
  def __call__(self, x):
    if x.shape[0] != self._n_channels:
      if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
      elif x.shape[0] > self._n_channels:
        x = x[:self._n_channels]
    return x
  
        
class CUB2011(Dataset):
  def __init__(self, root, dataset_key, transform=None, 
               val_split=0.1, split_idxs_root=None):
    super().__init__()
    self._root = os.path.join(root, 'CUB2011', 'data')
    self._dataset_key = dataset_key
    self._transform = transform
    self._val_split = val_split
    self._split_idxs_root = split_idxs_root
    
    self._setup_data()
  
  def _setup_path_to_idx_lookup(self):
    path = os.path.join(self._root, 'images.txt')
    path_to_idx_lookup = {}
    with open(path, 'r') as infile:
      for line in infile:
        line = line.strip()
        idx, path = line.split()
        path_to_idx_lookup[path] = int(idx)
    return path_to_idx_lookup
  
  def _setup_img_class_labels(self):
    path = os.path.join(self._root, 'image_class_labels.txt')
    img_class_labels = {}
    with open(path, 'r') as infile:
      for line in infile:
        line = line.strip()
        idx, target = line.split(' ')
        img_class_labels[int(idx)] = int(target)
    return img_class_labels
  
  def _setup_is_train_lookup(self):
    path = os.path.join(self._root, 'train_test_split.txt')
    is_train_lookup = {}
    with open(path, 'r') as infile:
      for line in infile:
        line = line.strip()
        idx, is_train = line.split(" ")
        is_train_lookup[int(idx)] = bool(int(is_train))
    return is_train_lookup
  
  def _init_dataframe(self, img_paths, path_to_idx_lookup, 
                      img_class_labels, is_train_lookup):
    # Build DF
    df = pd.DataFrame({"img_path": img_paths})
    for i, img_series in df.iterrows():
      p = os.path.sep.join(img_series.img_path.split(os.path.sep)[-2:])
      idx = path_to_idx_lookup[p]
      target = img_class_labels[idx]
      is_train = is_train_lookup[idx]
      df.at[i, 'idx'] = idx
      df.at[i, 'target'] = target
      df.at[i, 'dataset_key'] = 'dev' if is_train else 'test'

    df['idx'] = [int(ele) for ele in df.idx]
    df['target'] = [int(ele) for ele in df.target]
    assert len(df) == 11788
    assert len(np.unique(df.target)) == 200
    return df
  
  def _setup_data(self):
    path_to_idx_lookup = self._setup_path_to_idx_lookup()
    img_class_labels = self._setup_img_class_labels()
    is_train_lookup = self._setup_is_train_lookup()
    
    imgs_root = os.path.join(self._root, 'images')
    img_paths = glob.glob(f'{imgs_root}/*/*')
    
    df = self._init_dataframe(img_paths, path_to_idx_lookup, 
                              img_class_labels, is_train_lookup)
    
    if self._dataset_key == 'test':
      self._df = df[df.dataset_key=='test']
    else:
      dev_df = df[df.dataset_key=='dev']
      
      if self._split_idxs_root:
        val_percent = int(self._val_split * 100)
        train_percent = 100 - val_percent
        df_basename = f'{train_percent}-{val_percent}_split.csv'
        df_path = os.path.join(self._split_idxs_root, df_basename)
        if os.path.exists(df_path):
          dev_df = pd.read_csv(df_path)
        else:
          dev_df = self._create_dataframe_split(dev_df)
          if not os.path.exists(self._split_idxs_root):
            os.makedirs(self._split_idxs_root)
          dev_df.to_csv(df_path, index=False)
      else:
        dev_df = self._create_dataframe_split(dev_df)
      
      self._df = dev_df[dev_df.dataset_key == self._dataset_key]
      
  def _create_dataframe_split(self, dev_df):
    train_dfs = []
    val_dfs = []
    for class_idx, class_df in dev_df.groupby('target'):
      class_df = class_df.sample(frac=1).reset_index(drop=True)

      n_samples = len(class_df)
      n_val_samples = int(n_samples * self._val_split)

      train_class_df = class_df.iloc[n_val_samples:]
      val_class_df = class_df.iloc[:n_val_samples]

      train_dfs.append(train_class_df)
      val_dfs.append(val_class_df)

    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)

    train_df.dataset_key = 'train'
    val_df.dataset_key = 'val'

    assert len(train_df) + len(val_df) == len(dev_df)
    
    df = pd.concat([train_df, val_df])
    return df
  
  def __len__(self):
    return len(self._df)
  
  def __getitem__(self, idx):
    # Grab df
    df_i = self._df.iloc[idx]
    
    # Load image
    img_path = df_i.img_path
    img = Image.open(img_path)
    if self._transform:
      img = self._transform(img)
    
    # Load target
    target = df_i.target
    
    return img, target


def create_datasets(root, size=64, val_split=0.1, split_idxs_root=None):
  normalize_op = T.Normalize(
      mean=[0.485, 0.456, 0.406], 
      std=[0.229, 0.224, 0.225]
  )
  
  train_transforms = T.Compose([
      T.RandomResizedCrop(size),
      T.RandomHorizontalFlip(),
      T.ColorJitter(.4,.4,.4),
      T.ToTensor(),
      EnsureShape(),
      normalize_op,
  ])

  eval_transforms = T.Compose([
      T.Resize(int(size*1.14)),
      T.CenterCrop(size),
      T.ToTensor(),
      EnsureShape(),
      normalize_op,
  ])

  train_dataset = CUB2011(root, dataset_key='train', transform=train_transforms,
                          val_split=val_split, split_idxs_root=split_idxs_root)
  val_dataset = CUB2011(root, dataset_key='val', transform=eval_transforms,
                        val_split=val_split, split_idxs_root=split_idxs_root)
  test_dataset = CUB2011(root, dataset_key='test', transform=eval_transforms)
  
  # Package
  dataset_dict = {
      'train': train_dataset,
      'val': val_dataset,
      'test': test_dataset,
  }

  return dataset_dict