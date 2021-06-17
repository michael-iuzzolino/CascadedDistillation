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
  
        
class StanfordCars(Dataset):
  def __init__(self, root, dataset_key, transform=None, 
               val_split=0.1, split_idxs_root=None, force_overwrite_split=False):
    super().__init__()
    self._root = os.path.join(root, 'Cars', 'data')
    self._dataset_key = dataset_key
    self._transform = transform
    self._val_split = val_split
    self._split_idxs_root = split_idxs_root
    self._force_overwrite_split = force_overwrite_split
    
    self._setup_data()
  
  def _setup_data(self):
    img_paths = self._setup_img_paths()
    annotations = self._setup_annotations()
    
    df = self._create_dataframe(img_paths, annotations)
    
    if self._dataset_key == 'test':
      self._df = df
    else:
      if self._split_idxs_root:
        val_percent = int(self._val_split * 100)
        train_percent = 100 - val_percent
        df_basename = f'{train_percent}-{val_percent}_split.csv'
        df_path = os.path.join(self._split_idxs_root, df_basename)
        if os.path.exists(df_path) and not self._force_overwrite_split:
          df = pd.read_csv(df_path)
        else:
          df = self._create_split_df(df)
          if not os.path.exists(self._split_idxs_root):
            os.makedirs(self._split_idxs_root)
          df.to_csv(df_path, index=False)
      else:
        df = self._create_split_df(df)

      # Set dataset
      self._df = df[df.dataset_key == self._dataset_key]

  def _create_dataframe(self, img_paths, annotations):
    # Set annotations into dataframe
    df = pd.DataFrame(annotations, index=[0]).T
    df = df.reset_index()
    df.columns = ['img_path', 'target']

    # Set image paths
    df = df.sort_values('img_path')
    df.img_path = img_paths
    return df
  
  def _create_split_df(self, df):
    train_dfs = []
    val_dfs = []
    for class_idx in np.unique(df.target):
      class_df = df[df.target==class_idx]
      class_df = class_df.sample(frac=1).reset_index(drop=True)

      n_samples = len(class_df)
      n_val_samples = int(n_samples * self._val_split)

      train_class_df = class_df.iloc[n_val_samples:]
      val_class_df = class_df.iloc[:n_val_samples]

      train_dfs.append(train_class_df)
      val_dfs.append(val_class_df)

    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    train_df['dataset_key'] = 'train'
    val_df['dataset_key'] = 'val'
    
    assert len(train_df) + len(val_df) == len(df)
    
    df = pd.concat([train_df, val_df])
    return df
  
  def _setup_img_paths(self):
    dataset_key = 'train' if self._dataset_key in ['train', 'val'] else 'test'
    dataset_root = os.path.join(self._root, dataset_key)
    img_paths = glob.glob(f'{dataset_root}/*')
    return img_paths

  def _setup_annotations(self):
    lbl_i = 4
    img_i = 5
    
    # Set devkit root
    devkit_root = os.path.join(self._root, 'devkit')
    
    if self._dataset_key in ['train', 'val']:
      basename = 'cars_train_annos.mat'
    else:
      basename = 'test_annotations.mat'
    
    # Set path and load mat
    annot_path = os.path.join(devkit_root, basename)
    mat = scipy.io.loadmat(annot_path)
    
    # Build annotations
    annotations = {}
    for data in mat['annotations'][0]:
      lbl = data[lbl_i][0][0]
      img_name = data[img_i][0]
      annotations[img_name] = lbl
    
    # Assert correct number of instances for given dataset_key
    if self._dataset_key in ['train', 'val']:
      assert len(annotations) == 8144
    else:
      assert len(annotations) == 8041
      
    # Assert correct number of classes
    assert len(np.unique(list(annotations.values()))) == 196
    
    return annotations

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
    
    # Load target (adjust by 1)
    target = df_i.target - 1
    
    return img, target


def create_datasets(root, size=64, val_split=0.1, 
                    split_idxs_root=None, force_overwrite_split=False):
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

  train_dataset = StanfordCars(root, 
                               dataset_key='train', 
                               transform=train_transforms,
                               val_split=val_split,
                               split_idxs_root=split_idxs_root,
                               force_overwrite_split=force_overwrite_split)
  val_dataset = StanfordCars(root, 
                             dataset_key='val', 
                             transform=eval_transforms,
                             val_split=val_split,
                             split_idxs_root=split_idxs_root,
                             force_overwrite_split=force_overwrite_split)
  test_dataset = StanfordCars(root, 
                              dataset_key='test', 
                              transform=eval_transforms,
                              force_overwrite_split=force_overwrite_split)
  
  # Package
  dataset_dict = {
      'train': train_dataset,
      'val': val_dataset,
      'test': test_dataset,
  }

  return dataset_dict