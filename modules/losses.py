"""Custom losses."""
import torch
import torch.nn as nn


def categorical_cross_entropy(pred_logits, y_true_softmax):
  """Categorical cross entropy."""
  log_softmax_pred = nn.LogSoftmax(dim=1)(pred_logits)
  soft_targets = y_true_softmax.detach().clone()  # Stop gradient
  cce_loss = -(soft_targets * log_softmax_pred).sum(dim=1).mean()
  return cce_loss


class DistillationLossHandler(object):
  def __init__(self, alpha, temp):
    self._alpha = alpha
    self._temp = temp
    self._kl = nn.KLDivLoss()
  
  def __call__(self, outputs, labels, teacher_outputs):
    kl_loss = self._kl(
      nn.functional.log_softmax(outputs / self._temp, dim=1), 
      nn.functional.softmax(teacher_outputs / self._temp, dim=1)
    ) * (self._alpha * self._temp * self._temp)
    target_loss = nn.functional.cross_entropy(outputs, labels) * (1. - self._alpha)
    loss = kl_loss  + target_loss 
    return loss
  
  
class TD_Loss(object):
  def __init__(self, n_timesteps, tau_handler, flags, apply_temp_scaling=False):
    self.n_timesteps = n_timesteps
    self.tau_handler = tau_handler
    self.flags = flags
    self._apply_temp_scaling = apply_temp_scaling
  
  def __call__(self, criterion, predicted_logits, y, targets):
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
        softmax_i = nn.functional.softmax(logit_k, dim=1)
        sum_term = sum_term + self.flags.lambda_val**(i - 1) * softmax_i

      # Final terms
      term_1 = (1 - self.flags.lambda_val) * sum_term
      term_2 = self.flags.lambda_val**(self.n_timesteps - t - 1) * y
      softmax_j = term_1 + term_2
      
      # Temp scale
      if self._apply_temp_scaling:
        logit_i = logit_i / self.flags.distillation_temp
        softmax_j = softmax_j / self.flags.distillation_temp
      
      # Compute loss
      loss_i = criterion(pred_logits=logit_i, y_true_softmax=softmax_j)

      # Tau weighted
      if self.flags.tau_weighted_loss and t < self.n_timesteps - 1:
        tau_i = self.tau_handler(t-1, epoch_i)
        loss_i = tau_i * loss_i

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
      softmax_i = nn.functional.softmax(logit_i, dim=1)
      y_pred = torch.argmax(softmax_i, dim=1)

      # Updates running accuracy statistics
      n_correct = torch.eq(targets, y_pred).sum()
      acc_i = n_correct / float(targets.shape[0])
      timestep_accs[t] = acc_i

    # Normalize loss
    if self.flags.normalize_loss:
      loss = loss / float(self.n_timesteps)
    
    return loss, timestep_losses, timestep_accs
  
