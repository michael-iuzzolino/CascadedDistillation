"""
Source: https://zenodo.org/record/2535967#.Xx0oRIjYoQ8
"""
import os
import sys
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms as T

# CIFAR10 statistics
_MEAN = (0.4917, 0.4824, 0.4469)
_STD = (0.2469, 0.2434, 0.2615)

_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]

_NOISE_TYPES = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 
    'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 
    'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 
    'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur'
]


class CIFAR10_C(Dataset):
  def __init__(self, root, noise_level, target_set_keys=[]):
    self.root = os.path.join(root, 'CIFAR10C', 'data')
    self.noise_level = noise_level
    self.target_set_keys = target_set_keys
    
    self.n_elements_per_set = 50000
    self.n_levels = 5
    # Set class labels
    self.classes = _CLASSES

    self._build_dataset()
    self._build_transforms()

  def _build_transforms(self):
    self.transform = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])

  def _reorder_data(self, labels, sets):
    # Reorder the labels
    reordered_labels = np.reshape(labels, (self.n_levels, -1))
    reordered_labels = reordered_labels.T.flatten()

    # Reorder the sets
    reordered_sets = {}
    for set_key, set_vals in sets.items():
      shape = set_vals.shape
      X = np.reshape(set_vals,
                     (self.n_levels, shape[0] // self.n_levels, *shape[1:]))
      X = np.swapaxes(X, 0, 1).reshape(shape)
      reordered_sets[set_key] = X

    return reordered_labels, reordered_sets

  def _build_dataset(self):
    # Load labels
    self.labels = np.load(glob.glob(f'{self.root}/*labels*')[0])
    # Load corruption sets
    np_sets = [ele 
               for ele in glob.glob(f'{self.root}/*') 
               if 'labels.npy' not in ele]
    if len(self.target_set_keys) > 0:
      np_sets = [
          ele for ele in np_sets
          if os.path.basename(ele).replace('.npy', '') in self.target_set_keys
      ]
    np_sets = np.sort(np_sets)

    self.sets = {
        os.path.basename(set_key).replace('.npy', ''): np.load(set_key)
        for set_key in np_sets
    }

    # Reorder sets
    self.labels, self.sets = self._reorder_data(self.labels, self.sets)

    # Compute number of sets
    self.n_sets = len(self.sets)

    # Compute number of elements
    self.n_elements = sum([ele.shape[0] for ele in self.sets.values()])

    # Set levels
    self.levels = np.tile(list(range(self.n_levels)),
                          len(self.labels) // self.n_levels)

  def __len__(self):
    return self.n_elements

  def __getitem__(self, idx):
    # Get set idx and set-value idx
    set_idx = idx // self.n_elements_per_set
    set_val_idx = idx % self.n_elements_per_set

    # Get set key
    set_key = list(self.sets.keys())[set_idx]

    # Load X, y
    X = self.sets[set_key][set_val_idx]
    y = self.labels[set_val_idx]
    level = self.levels[set_val_idx]

    # Apply transform
    X = self.transform(X)

    return X, y, self.classes[y], level, set_key


def build_dataset(root, noise_level, batch_size=1, target_set_keys=[]):
  dataset_src = CIFAR10_C(root, noise_level, target_set_keys)
  data_loader = DataLoader(dataset_src, batch_size=batch_size, shuffle=False)
  return data_loader