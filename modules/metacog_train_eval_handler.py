import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def train_model(net, loaders, figs_root, args):
  train_loader = loaders['train']
  val_loader = loaders['val']
  
  criterion = nn.BCELoss()
  optimizer = optim.Adam(net.parameters(), 
                         lr=args.learning_rate, 
                         weight_decay=args.weight_decay)

  metrics = {dataset_key: {"loss": [], "acc": []} 
             for dataset_key in ['train', 'val']}

  for epoch_i in range(args.n_epochs):
    try:
      net.train()
      train_batch_losses = []
      n_train_correct = 0
      n_train_seen = 0
      for batch_i, (X, y) in enumerate(train_loader):
        X = X.to(args.device)
        y = y.to(args.device)

        out = net(X)

        loss = 0
        for t, out_t in enumerate(out):
          y_t = y[:,t]
          loss += criterion(out_t, y_t)

        # Normalize by number of timesteps
        loss = loss / X.shape[1]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_batch_losses.append(loss.item())
        mean_batch_loss = np.mean(train_batch_losses)

        correct_ts = []
        for t, out_t in enumerate(out):
          y_t = y[:,t]
          pred_t = (out_t > 0.5).int()
          correct_t = (pred_t.flatten()==y_t).int().sum().item()
          correct_ts.append(correct_t)
        n_train_correct += np.mean(correct_ts)
        n_train_seen += X.shape[0]

        train_acc = n_train_correct / n_train_seen * 100

        if epoch_i % 2 == 0 and batch_i % 50 == 0:
          sys.stdout.write((f'\rEpoch {epoch_i+1}/{args.n_epochs} -- '
                            f'Batch {batch_i+1}/{len(train_loader)} -- '
                            f'Loss: {mean_batch_loss:0.4f} -- '
                            f'Acc: {train_acc:0.2f}%'))
          sys.stdout.flush()
    except KeyboardInterrupt:
      print("Ending early.")
      break
      
    epoch_train_acc = n_train_correct / n_train_seen
    metrics['train']['loss'].append((epoch_i, np.mean(train_batch_losses)))
    metrics['train']['acc'].append((epoch_i, epoch_train_acc))
    
    n_val = len(loaders['val'].dataset.Xs)
    if n_val > 0 and epoch_i % args.eval_freq == 0:
      net.eval()
      val_batch_losses = []
      n_val_correct = 0
      n_val_seen = 0

      for batch_i, (X, y) in enumerate(val_loader):
        X = X.to(args.device)
        y = y.to(args.device)

        with torch.no_grad():
          out = net(X)
        val_loss = 0
        for t, out_t in enumerate(out):
          y_t = y[:,t]
          val_loss += criterion(out_t, y_t)
        
        # Normalize time
        val_loss = val_loss / X.shape[1]
        
        val_batch_losses.append(val_loss.item())
        mean_batch_loss = np.mean(val_batch_losses)
        
        correct_ts = []
        for t, out_t in enumerate(out):
          y_t = y[:,t]
          pred_t = (out_t > 0.5).int()
          correct_t = (pred_t.flatten()==y_t).int().sum().item()
          correct_ts.append(correct_t)
        n_val_correct += np.mean(correct_ts)
        n_val_seen += X.shape[0]

        val_acc = n_val_correct / n_val_seen * 100

      epoch_val_acc = n_val_correct / n_val_seen
      print(f"\nEval acc: {epoch_val_acc*100:0.2f}% -- Loss: {mean_batch_loss:0.4f}")
      metrics['val']['loss'].append((epoch_i, np.mean(val_batch_losses)))
      metrics['val']['acc'].append((epoch_i, epoch_val_acc))

  fig, axes = plt.subplots(1, 2, figsize=(12,4))
  for i, metric_key in enumerate(['loss', 'acc']):
    ax_i = axes[i]
    for dataset_key, dataset_vals in metrics.items():
      linestyle = '-' if dataset_key == 'train' else '--'
      x_vals = [ele[0] for ele in dataset_vals[metric_key]]
      y_vals = [ele[1] for ele in dataset_vals[metric_key]]
      ax_i.plot(x_vals, y_vals, label=dataset_key, linestyle=linestyle)
    ax_i.set_xlabel('Epochs')
    ax_i.set_ylabel(f'{metric_key.capitalize()}')
    ax_i.set_title(f'{metric_key.capitalize()} vs. Epochs')
    ax_i.legend()
  save_path = os.path.join(figs_root, f'training_curves.png')
  fig.savefig(save_path)
  plt.close()
  
  return net


def eval_model(net, loader, args):
  net.eval()
  n_test_correct = 0
  n_test_seen = 0

  for batch_i, (X, y) in enumerate(loader):
    X = X.to(args.device)
    y = y.to(args.device)

    with torch.no_grad():
      out = net(X)
    
    correct_ts = []
    for t, out_t in enumerate(out):
      y_t = y[:,t]
      pred_t = (out_t > 0.5).int()
      correct_t = (pred_t.flatten()==y_t).int().sum().item()
      correct_ts.append(correct_t)
    n_test_correct += np.mean(correct_ts)
    n_test_seen += X.shape[0]
    
  test_acc = n_test_correct / n_test_seen * 100
  print(f"test_acc: {test_acc:0.2f}%")
  
  return test_acc