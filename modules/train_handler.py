"""Training function handler for sequential or cascaded."""
import torch
import torch.nn as nn
import sys
import numpy as np
import torch.nn.functional as F
from models import model_utils
from modules import losses
    

class SequentialTrainingScheme:
  """Sequential Training Scheme."""

  def __init__(self, num_classes, flags):
    """Initialize sequential training handler."""
    self.num_classes = num_classes
    self.flags = flags
    self._criterion = losses.categorical_cross_entropy

  def __call__(self, net, loader, epoch_i, optimizer, device):
    # Flag model for training
    net.train()

    batch_losses = []
    batch_accs = []
    for batch_i, (data, targets) in enumerate(loader):
      if self.flags.debug and batch_i > 1:
        break

      # Determine device placement
      data = data.to(device, non_blocking=True)
      targets = targets.to(device, non_blocking=True)
      
      # One-hot-ify targets
      y = torch.eye(self.num_classes)[targets]
      y = y.to(device, non_blocking=True)

      # Zero gradients
      optimizer.zero_grad()

      # Run forward pass
      results = net(data, t=0)

      # Unpack data
      logits = results["logits"]

      # Compute loss
      loss = self._criterion(logits, y)

      # Compute gradients
      loss.backward()

      # Take optimization step
      optimizer.step()

      # Predictions
      softmax_output = F.softmax(logits, dim=1)
      y_pred = torch.argmax(softmax_output, dim=1)

      # Updates batch accs
      n_correct = torch.eq(targets, y_pred).sum()
      acc_i = n_correct / float(targets.shape[0])
      batch_accs.append(acc_i.item())

      # Update batch loss
      batch_losses.append(loss.item())

      sys.stdout.write((f"\rBatch {batch_i+1}/{len(loader)} -- "
                        f"Batch Loss: {np.mean(batch_losses):0.6f} -- "
                        f"Batch Acc: {np.mean(batch_accs)*100:0.2f}%"))
      sys.stdout.flush()

    return batch_losses, batch_accs

  
class DistillationSequentialTrainingScheme:
  """Sequential Training Scheme.
  See: https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/train.py
  """

  def __init__(self, num_classes, flags):
    """Initialize sequential training handler."""
    self.num_classes = num_classes
    self.flags = flags
    self._criterion = losses.categorical_cross_entropy

  def __call__(self, net, loader, epoch_i, optimizer, device, teacher_net):
    # Flag model for training
    net.train()

    batch_losses = []
    batch_accs = []
    for batch_i, (data, targets) in enumerate(loader):
      if self.flags.debug and batch_i > 1:
        break

      # Determine device placement
      data = data.to(device, non_blocking=True)
      targets = targets.to(device, non_blocking=True)
      
      # Get teacher preds
      with torch.no_grad():
        for t in range(teacher_net.timesteps):
          results = teacher_net(data, t)
          # Unpack data
          teacher_logits = results["logits"]

      teacher_targets = F.softmax(teacher_logits, dim=1).argmax(dim=1).long()
      teacher_y = torch.eye(self.num_classes)[teacher_targets].long()
      teacher_y = teacher_y.to(device, non_blocking=True)

      # Zero gradients
      optimizer.zero_grad()

      # Run forward pass
      results = net(data, t=0)

      # Unpack data
      logits = results["logits"]

      # Compute loss
      loss = self._criterion(logits, targets, teacher_y)

      # Compute gradients
      loss.backward()

      # Take optimization step
      optimizer.step()

      # Predictions
      softmax_output = F.softmax(logits, dim=1)
      y_pred = torch.argmax(softmax_output, dim=1)

      # Updates batch accs
      n_correct = torch.eq(targets, y_pred).sum()
      acc_i = n_correct / float(targets.shape[0])
      batch_accs.append(acc_i.item())

      # Update batch loss
      batch_losses.append(loss.item())

      sys.stdout.write((
        f"\rBatch {batch_i+1}/{len(loader)} -- "
        f"Batch Loss: {np.mean(batch_losses):0.6f} -- "
        f"Batch Acc: {np.mean(batch_accs)*100:0.2f}%"
      ))
      sys.stdout.flush()

    return batch_losses, batch_accs

  
class DistillationCascadedTrainingScheme(object):
  """Cascaded training schemes.
  
  See https://github.com/peterliht/knowledge-distillation-pytorch/blob/ef06124d67a98abcb3a5bc9c81f7d0f1f016a7ef/model/net.py#L100
  """
  
  def __init__(self, n_timesteps, num_classes, flags):
    """Initialize cascaded training handler."""
    self.n_timesteps = n_timesteps
    self.num_classes = num_classes
    self.flags = flags
    self._criterion = losses.Distillation_TD_Loss(n_timesteps, flags)
    
  def __call__(
    self, 
    net, 
    loader,
    epoch_i, 
    optimizer, 
    device, 
    teacher_net,
  ):
    # Flag model for training
    net.train()
    
    batch_losses = []
    batch_accs = []
    for batch_i, (data, targets) in enumerate(loader):
      if self.flags.debug and batch_i > 1:
        break
      # Send to device
      data = data.to(device)
      targets = targets.to(device)
      
      # Get teacher preds
      with torch.no_grad():
        for t in range(teacher_net.timesteps):
          results = teacher_net(data, t)
          # Unpack data
          teacher_logits = results["logits"]
      teacher_targets = F.softmax(teacher_logits, dim=1).argmax(dim=1)
      teacher_y = torch.eye(self.num_classes)[teacher_targets]
      teacher_y = teacher_y.to(teacher_targets.device, non_blocking=True)
      
      # Zero out grads
      optimizer.zero_grad()
      
      predicted_logits = []
      predicted_temps = []
      for t in range(self.n_timesteps):
        # Run forward pass
        results = net(data, t)

        # Unpack data
        logit_t = results["logits"]
        predicted_logits.append(logit_t)

        # Check for temp prediction
        if "temp_pred" in results:
          predicted_temps.append(results["temp_pred"])
        else:
          predicted_temps.append(None)

      # One-hot-ify targets and send to output device
      targets = targets.to(logit_t.device, non_blocking=True)
      y = torch.eye(self.num_classes)[targets]
      y = y.to(targets.device, non_blocking=True)
      
      """
      Compute distillation loss within TD loss computation
      """
      loss, target_losses, target_accs = self._criterion(
          predicted_logits=predicted_logits,
          predicted_temps=predicted_temps,
          teacher_y=teacher_y,
          y=y,
          targets=targets,
      )
      
      # Compute gradients
      loss.backward()

      # Take optimization step
      optimizer.step()
      
      # Update batch loss and compute average
      batch_losses.append(target_losses)
      batch_accs.append(target_accs)
      
      # Compute means
      mean_batch_loss = torch.stack(batch_losses).mean().item()
      mean_batch_acc = torch.stack(batch_accs).mean().item() * 100
      
      sys.stdout.write((f"\rTraining Batch {batch_i+1}/{len(loader)} -- "
                        f"Batch Loss: {mean_batch_loss:0.6f} -- "
                        f"Batch Acc: {mean_batch_acc:0.2f}%"))
      sys.stdout.flush()
      
    # Average over the batches per timestep
    batch_losses = torch.stack(batch_losses).detach().numpy()
    batch_accs = torch.stack(batch_accs).detach().numpy()

    return batch_losses, batch_accs

  
class CascadedTrainingScheme(object):
  """Cascaded training schemes."""

  def __init__(self, n_timesteps, num_classes, flags):
    """Initialize cascaded training handler."""
    self.n_timesteps = n_timesteps
    self.num_classes = num_classes
    self.flags = flags
    self._criterion = losses.categorical_cross_entropy

  def __call__(self, net, loader, epoch_i, optimizer, device):
    # Flag model for training
    net.train()
    
    batch_losses = []
    batch_accs = []
    for batch_i, (data, targets) in enumerate(loader):
      if self.flags.debug and batch_i > 1:
        break
      # Send to device
      data = data.to(device)
      targets = targets.to(device)

      # Zero out grads
      optimizer.zero_grad()
      
      predicted_logits = []
      for t in range(self.n_timesteps):
        # Run forward pass
        results = net(data, t)

        # Unpack data
        logit_t = results["logits"]
        predicted_logits.append(logit_t)

      # One-hot-ify targets and send to output device
      targets = targets.to(logit_t.device, non_blocking=True)
      y = torch.eye(self.num_classes)[targets]
      y = y.to(targets.device, non_blocking=True)
      
      loss = 0
      timestep_losses = torch.zeros(self.n_timesteps)
      timestep_accs = torch.zeros(self.n_timesteps)

      for t in range(len(predicted_logits)):
        logit_i = predicted_logits[t]

        # First term
        sum_term = torch.zeros_like(logit_i)
        t_timesteps = list(range(t+1, self.n_timesteps))
        for i, n in enumerate(t_timesteps, 1):
          logit_k = predicted_logits[n].detach().clone()
          softmax_i = F.softmax(logit_k, dim=1)
          sum_term = sum_term + self.flags.lambda_val**(i - 1) * softmax_i

        # Final terms
        term_1 = (1 - self.flags.lambda_val) * sum_term
        term_2 = self.flags.lambda_val**(self.n_timesteps - t - 1) * y
        softmax_j = term_1 + term_2

        # Compute loss
        loss_i = self._criterion(pred_logits=logit_i, y_true_softmax=softmax_j)
        
        # Aggregate loss
        if self.flags.tdl_mode == "EWS":
          loss = loss + loss_i
        else:
          # Ignore first timestep loss (all 0's output)
          if t > 0:
            loss = loss + loss_i

        # Log loss item
        timestep_losses[t] = loss_i.item()

        # Predictions
        softmax_i = F.softmax(logit_i, dim=1)
        y_pred = torch.argmax(softmax_i, dim=1)

        # Updates running accuracy statistics
        n_correct = torch.eq(targets, y_pred).sum()
        acc_i = n_correct / float(targets.shape[0])
        timestep_accs[t] = acc_i

      # Normalize loss
      if self.flags.normalize_loss:
        loss = loss / float(self.n_timesteps)
      
      # Compute gradients
      loss.backward()

      # Take optimization step
      optimizer.step()
      
      # Update batch loss and compute average
      batch_losses.append(timestep_losses)
      batch_accs.append(timestep_accs)
      
      # Compute means
      mean_batch_loss = torch.stack(batch_losses).mean().item()
      mean_batch_acc = torch.stack(batch_accs).mean().item() * 100
      
      sys.stdout.write((
        f"\rTraining Batch {batch_i+1}/{len(loader)} -- "
        f"Batch Loss: {mean_batch_loss:0.6f} -- "
        f"Batch Acc: {mean_batch_acc:0.2f}%"
      ))
      sys.stdout.flush()
      
    # Average over the batches per timestep
    batch_losses = torch.stack(batch_losses).detach().numpy()
    batch_accs = torch.stack(batch_accs).detach().numpy()

    return batch_losses, batch_accs


def get_train_loop(n_timesteps, num_classes, flags):
  """Retrieve sequential or cascaded training function."""
  if flags.distillation:
    if flags.train_mode == "baseline":
      print("Setting training scheme to DistillationSequentialTrainingScheme")
      train_fxn = DistillationSequentialTrainingScheme(num_classes, flags)
    elif flags.train_mode == "cascaded":
      print("Setting training scheme to DistillationCascadedTrainingScheme")
      train_fxn = DistillationCascadedTrainingScheme(
        n_timesteps, num_classes, flags,
      )
    else:
      raise NotImplementedError(f"{flags.train_mode} train mode not implemented")
  else:
    if flags.train_mode == "baseline":
      print("Setting training scheme to SequentialTrainingScheme")
      train_fxn = SequentialTrainingScheme(num_classes, flags)
    elif flags.train_mode == "cascaded":
      print("Setting training scheme to CascadedTrainingScheme")
      train_fxn = CascadedTrainingScheme(
        n_timesteps, num_classes, flags,
      )
    else:
      raise NotImplementedError(f"{flags.train_mode} train mode not implemented")
  return train_fxn
