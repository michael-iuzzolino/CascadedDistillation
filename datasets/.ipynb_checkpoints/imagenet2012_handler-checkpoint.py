# https://github.com/fastai/imagenet-fast/blob/master/imagenet_nv/fastai_imagenet.py#L128
import copy
import json
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import noise
pd.options.mode.chained_assignment = None  # default='warn'


__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class DatasetProcessor:
  def __init__(self, root, test_split=0.1, val_split=0.1):
    self._root = os.path.join(root, 'ImageNet2012')
    self._test_split = test_split
    self._val_split = val_split
    
    self._build_synset_lookup()
    self._set_df_path()
    
    if os.path.exists(self._df_path):
      self.df = pd.read_csv(self._df_path)
    else:
      self._setup_df()
      self.df = self._create_split_df()
      self.df.to_csv(self._df_path, index=False)
      
  def _set_df_path(self):
    df_root = os.path.join(self._root, 'split_dfs')
    if not os.path.exists(df_root):
      os.makedirs(df_root)
    df_basename = f'df__test_split_{test_split}__val_split_{val_split}.csv'
    self._df_path = os.path.join(df_root, df_basename)
    
  def _create_split_df(self):
    unique_synset_idx = np.unique([os.path.basename(os.path.dirname(ele)) 
                                   for ele in self.df.img_path])

    dfs = []
    for i, synset_id in enumerate(unique_synset_idx):
      sys.stdout.write((f'\rProcessing Synset id: {synset_id} '
                        f'[{i+1}/{len(unique_synset_idx)}]'))
      sys.stdout.flush()

      df = self.df[self.df.img_path.str.contains(synset_id)]
      df = df.sample(frac=1).reset_index(drop=True)

      num_test = int(len(df) * self._test_split)
      num_dev = len(df) - num_test

      num_val = int(num_dev * self._val_split)
      num_train = num_dev - num_val

      test_df = df.iloc[:num_test]
      val_df = df.iloc[num_test:num_test+num_val]
      train_df = df.iloc[num_test+num_val:]

      test_df['set'] = 'test'
      val_df['set'] = 'val'
      train_df['set'] = 'train'

      synset_df = pd.concat([train_df, test_df, val_df])

      dfs.append(synset_df)

    dfs = pd.concat(dfs)
    return dfs
    
  def _setup_df(self):
    dataset_root = os.path.join(self._root, 
                                'ILSVRC/Data/CLS-LOC', 
                                'train')
    
    img_paths = glob.glob(f'{dataset_root}/*/*')
    
    self.df = pd.DataFrame({"img_path": img_paths})
    self.df['synset_id'] = [os.path.basename(os.path.dirname(ele))
                            for ele in self.df.img_path]
    self.df['class'] = [self.synset_lookup[ele] for ele in self.df.synset_id]
    
    sorted_keys = np.sort(list(self.synset_lookup.keys()))
    synset_to_labels = {k: i for i, k in enumerate(sorted_keys)}
    self.df['label'] = [synset_to_labels[ele] for ele in self.df.synset_id]
    
  def _build_synset_lookup(self):
    synset_map = os.path.join(root, 'LOC_synset_mapping.txt')
    self.synset_lookup = {}
    with open(synset_map, 'r') as infile:
      for line in infile:
        line = line.strip()
        k = line.split(" ")[0]
        v = " ".join(line.split(" ")[1:])
        self.synset_lookup[k] = v
      
      
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
  
        
class ImageNetHandler(Dataset):
  """Imagenet dataset handler."""
  def __init__(self, root, dataset_key, transform=None, test_split=0.1, val_split=0.1):
    super().__init__()
    self._root = os.path.join(root, 'ImageNet2012')
    self._dataset_key = dataset_key
    self._transform = transform
    self._test_split = test_split
    self._val_split = val_split
    
    self._load_df()
  
  def _load_df(self):
    df_root = os.path.join(self._root, 'split_dfs')
    df_basename = (f'df__test_split_{self._test_split}__'
                   f'val_split_{self._val_split}.csv')
    df_path = os.path.join(df_root, df_basename)
    if not os.path.exists(df_path):
      print(f'{df_path} does not exist.')
      print('Options: ')
      for path in os.listdir(os.path.dirname(df_path)):
        print(path)
      return
    
    df = pd.read_csv(df_path)
    self.df = df[df.set==self._dataset_key]
    
  def __len__(self):
    return len(self.df.img_path)
  
  def __getitem__(self, index):
    df_i = self.df.iloc[index]
    
    # Load iamge
    img = Image.open(df_i.img_path)

    if self._transform is not None:
      img = self._transform(img)

    # Grab target
    target = df_i.label
    cls_label = df_i['class']
    
    return img, target
  
  
# Lighting data augmentation take from here - https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
class Lighting(object):
  """Lighting noise(AlexNet - style PCA - based noise)"""

  def __init__(self, alphastd, eigval, eigvec):
    self.alphastd = alphastd
    self.eigval = eigval
    self.eigvec = eigvec

  def __call__(self, img):
    if self.alphastd == 0:
      return img

    alpha = img.new().resize_(3).normal_(0, self.alphastd)
    rgb = self.eigvec.type_as(img).clone()\
        .mul(alpha.view(1, 3).expand(3, 3))\
        .mul(self.eigval.view(1, 3).expand(3, 3))\
        .sum(1).squeeze()
    return img.add(rgb.view(3, 1, 1).expand_as(img))


def create_datasets(root, size=224, test_split=0.1, val_split=0.1):
  normalize_op = torchvision.transforms.Normalize(
      mean=[0.485, 0.456, 0.406], 
      std=[0.229, 0.224, 0.225]
  )
  train_transforms = T.Compose([
      T.RandomResizedCrop(size),
      T.RandomHorizontalFlip(),
      T.ColorJitter(.4,.4,.4),
      T.ToTensor(),
      EnsureShape(),
      Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
      normalize_op,
  ])

  eval_transforms = T.Compose([
      T.Resize(int(size*1.14)),
      T.CenterCrop(size),
      T.ToTensor(),
      EnsureShape(),
      normalize_op,
  ])

  train_dataset = ImageNetHandler(root, 
                                  dataset_key='train', 
                                  transform=train_transforms, 
                                  test_split=test_split, 
                                  val_split=val_split)
  val_dataset = ImageNetHandler(root, 
                                dataset_key='val', 
                                transform=eval_transforms, 
                                test_split=test_split, 
                                val_split=val_split)
  test_dataset = ImageNetHandler(root, 
                                 dataset_key='test', 
                                 transform=eval_transforms, 
                                 test_split=test_split, 
                                 val_split=val_split)
  
  # Package
  dataset_dict = {
      'train': train_dataset,
      'val': val_dataset,
      'test': test_dataset,
  }

  return dataset_dict