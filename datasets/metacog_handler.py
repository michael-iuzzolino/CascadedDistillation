import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import modules.metacog_utils as metacog_utils

class MetaCogDataset(Dataset):
  def __init__(self, Xs, ys, lbls):
    self.Xs = Xs
    self.ys = ys
    self.lbls = lbls
  
  def __len__(self):
    return len(self.Xs)
  
  def __getitem__(self, idx):
    X = torch.tensor(self.Xs[idx]).float()
    y = torch.tensor(self.ys[idx]).float()
    return X, y
  

def create_datasets(dataset_paths, val_split):
  dataset_src = {}
  for dataset_key, path in dataset_paths.items():
    outrep = torch.load(path)
    logits = outrep['logits']
    predictions = outrep['predictions']
    targets = outrep['target']
    final_logits = logits[-1]

    final_preds = final_logits.argmax(dim=1)
    final_preds = final_preds.unsqueeze(dim=0).repeat(logits.shape[0], 1)

    Xs = logits.permute(1,0,2)
    ys = (predictions==final_preds).int().long().permute(1,0)
    
    print(f"{path} -- {dataset_key}: {Xs.shape}")
    if dataset_key == 'test':
      dataset_src[dataset_key] = {
          "X": Xs.detach().numpy(), 
          "y": np.array(ys), 
          "lbls": np.array(targets)
      }
    else:
      train_idxs, val_idxs = metacog_utils.train_val_split(Xs, val_split)
#       train_idxs, val_idxs = metacog_utils.train_val_split_balanced(Xs, 
#                                                                     ys, 
#                                                                     val_split)

      X_train = Xs[train_idxs].detach().numpy()
      y_train = np.array(ys)[train_idxs]
      X_val = Xs[val_idxs].detach().numpy()
      y_val = np.array(ys)[val_idxs]

      lbl_train = np.array(targets)[train_idxs]
      lbl_val = np.array(targets)[val_idxs]

      dataset_src['train'] = {"X": X_train, "y": y_train, "lbls": lbl_train}
      dataset_src['val'] = {"X": X_val, "y": y_val, "lbls": lbl_val}
  
  return dataset_src

      
def build(dataset_paths, batch_size=256, val_split=0.01):
  # Create datasets
  dataset_src = create_datasets(dataset_paths, val_split)
  
  loaders = {}
  for dataset_key, dataset_vals in dataset_src.items():
    dataset_src = MetaCogDataset(dataset_vals['X'], 
                                 dataset_vals['y'],
                                 dataset_vals['lbls'])
    loader = DataLoader(dataset_src, 
                        batch_size=batch_size, 
                        shuffle=dataset_key=='train')
    loaders[dataset_key] = loader

  # Get train dataset stats
  train_Xs = loaders['train'].dataset.Xs
  train_stats = {"min": train_Xs.min(), "max": train_Xs.max()}

  for dataset_key, loader in loaders.items():
    loader.dataset.train_stats = train_stats
  
  return loaders