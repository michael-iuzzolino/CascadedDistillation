"""Main training script for Cascaded Nets."""
import argparse
import collections
import glob
import numpy as np
import os
import sys
import torch
import json
from collections import defaultdict, OrderedDict
from datasets.dataset_handler import DataHandler
from models import densenet
from models import resnet
from models import sdn_utils
from modules import eval_handler
from modules import losses
from modules import train_handler
from modules import utils
from torch import optim


def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--random_seed", type=int, default=42,
                      help="random seed")
  
  # Paths
  parser.add_argument("--experiment_root", type=str, 
                      default="experiments",
                      help="Local output dir")
  parser.add_argument("--experiment_name", type=str, 
                      required=True,
                      help="Experiment name")
  

  # Distillation
  parser.add_argument("--distillation", action="store_true", default=False,
                      help="Use distillation")
  parser.add_argument("--distillation_alpha", type=float, default=0.5,
                      help="Distillation alpha: 0.5 default")
  parser.add_argument("--teacher_dir", type=str, 
                      default="",
                      help="Teacher network root")
  parser.add_argument("--trainable_temp", action="store_true", default=False,
                      help="Trainable temperature scaling")
  parser.add_argument("--init_temp", type=float, default=1.0,
                      help="Init temp value: 1.0 default")
                      
  parser.add_argument("--temp_fc_lr", type=float, default=0.00001,
                      help="Temp param learning rate: 0.00001 default")
  
  # Dataset
  parser.add_argument("--dataset_root", type=str, required=True,
                      help="Dataset root")
  parser.add_argument("--dataset_name", type=str, required=True,
                      help="Dataset name: CIFAR10, CIFAR100, TinyImageNet")
  parser.add_argument("--split_idxs_root", type=str, default="split_idxs",
                      help="Split idxs root")
  parser.add_argument("--val_split", type=float, default=0.1,
                      help="Validation set split: 0.1 default")
  parser.add_argument("--test_split", type=float, default=0.1,
                      help="Test set split: 0.1 default")
  parser.add_argument("--augmentation_noise_type", type=str, 
                      default="occlusion",
                      help="Augmentation noise type: occlusion")
  parser.add_argument("--batch_size", type=int, default=128,
                      help="batch_size")
  parser.add_argument("--num_workers", type=int, default=2,
                      help="num_workers")
  parser.add_argument("--drop_last", action="store_true", default=False,
                      help="Drop last batch remainder")
  
  # Model
  parser.add_argument("--model_key", type=str, default="resnet18",
                      help="Model: resnet18, resnet34, ..., densenet_cifar")
  parser.add_argument("--train_mode", type=str, 
                      default="baseline",
                      help="Train mode: baseline, ic_only, sdn")
  parser.add_argument("--bn_time_affine", action="store_true", default=False,
                      help="Use temporal affine transforms in BatchNorm")
  parser.add_argument("--bn_time_stats", action="store_true", default=False,
                      help="Use temporal stats in BatchNorm")
  parser.add_argument("--tdl_mode", type=str, 
                      default="OSD",
                      help="TDL mode: OSD, EWS, noise")
  parser.add_argument("--tdl_alpha", type=float, default=0.0,
                      help="TDL alpha for EWS temporal kernel")
  parser.add_argument("--noise_var", type=float, default=0.0,
                      help="Noise variance on noise temporal kernel")
  parser.add_argument("--lambda_val", type=float, default=1.0,
                      help="TD lambda value")
  parser.add_argument("--cascaded", action="store_true", default=False,
                      help="Cascaded net")
  parser.add_argument("--cascaded_scheme", type=str, default="parallel",
                      help="cascaded_scheme: serial, parallel")
  parser.add_argument("--target_IC_inference_costs", nargs="+", type=float, 
                      default=[0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90],
                      help="target_IC_inference_costs")
  parser.add_argument("--use_pretrained_weights", action="store_true", default=False,
                      help="Use pretrained weights")
  parser.add_argument("--use_all_ICs", action="store_true", default=False,
                      help="Use all internal classifiers")
  parser.add_argument("--multiple_fcs", action="store_true", default=False,
                      help="One FC per timestep")
  
  
  # Optimizer
  parser.add_argument("--learning_rate", type=float, default=0.1,
                      help="learning rate")
  parser.add_argument("--momentum", type=float, default=0.9,
                      help="momentum")
  parser.add_argument("--weight_decay", type=float, default=0.0005,
                      help="weight_decay")
  parser.add_argument("--nesterov", action="store_true", default=False,
                      help="Nesterov for SGD")
  parser.add_argument("--normalize_loss", action="store_true", default=False,
                      help="Normalize temporal loss")
  
  # LR scheduler
  parser.add_argument("--lr_milestones", nargs="+", type=float, 
                      default=[60, 120, 150],
                      help="lr_milestones")
  parser.add_argument("--lr_schedule_gamma", type=float, default=0.2,
                      help="lr_schedule_gamma")
  
  # Other
  parser.add_argument("--use_cpu", action="store_true", default=False,
                      help="Use cpu")
  parser.add_argument("--device", type=int, default=0,
                      help="GPU device num")
  parser.add_argument("--n_epochs", type=int, default=150,
                      help="Number of epochs to train")
  parser.add_argument("--eval_freq", type=int, default=10,
                      help="eval_freq")
  parser.add_argument("--save_freq", type=int, default=5,
                      help="save_freq")
  parser.add_argument("--keep_logits", action="store_true", default=False,
                      help="Keep logits")
  parser.add_argument("--debug", action="store_true", default=False,
                      help="Debug mode")
  
  args = parser.parse_args()
  
  # Ensure teacher_dir set if distillation mode enabled
  if args.distillation:
    if args.teacher_dir == "":
      distill_model_key = args.model_key
      if "_small" in distill_model_key:
        distill_model_key = distill_model_key.replace("_small", "")
      dirname = f"{distill_model_key}_{args.dataset_name}"
      args.teacher_dir = os.path.join("teacher_ckpts", dirname)
      print(f"Setting teacher dir to {args.teacher_dir}")
      exit()
      if not os.path.exists(args.teacher_dir):
        raise AssertionError(f"Teacher ckpt does not exist @ {args.teacher_dir}")
    else:
      print(f"Teacher dir @ {args.teacher_dir}")
  
  # Set debug condition
  if args.debug:
    args.n_epochs = 5
    
  # Flag check
  if args.tdl_mode == "EWS":
    assert args.tdl_alpha is not None, "tdl_alpha not set"
  elif args.tdl_mode == "noise":
    assert args.noise_var is not None, "noise_var not set"
  
  if args.train_mode == "cascaded":
    args.cascaded = True
  return args


def get_baseline_ckpt_path(save_root, args):
  baseline_ckpt_roots = glob.glob(
    f"{os.path.dirname(save_root)}/std*seed_{args.random_seed}*"
  )
  assert len(baseline_ckpt_roots), f"No baseline for seed {args.random_seed}"
  baseline_ckpt_root = baseline_ckpt_roots[0]
  final_ckpt_path = np.sort(glob.glob(f"{baseline_ckpt_root}/ckpts/*.pt"))[-1]
  return final_ckpt_path


def setup_output_dir(args, save_args_to_root=True):
  if args.train_mode in ["baseline", "cascaded"]:
    out_basename = (f"td({args.lambda_val}),{args.cascaded_scheme}" 
                    if args.cascaded else 
                    "std")
  else:
    out_basename = f"{args.train_mode},td({args.lambda_val})"
  out_basename += f",lr_{args.learning_rate}"
  out_basename += f",wd_{args.weight_decay}"
  out_basename += f",seed_{args.random_seed}"
  
  if args.distillation:
    out_basename += f",distillation,alpha_{args.distillation_alpha}"
    teacher_dir = os.path.basename(args.teacher_dir)
    out_basename += f"++teacher__{teacher_dir}"

    if args.trainable_temp:
      out_basename += ",trainable_temp"
    else:
      out_basename += ",fixed_temp"
  
  if args.train_mode in ["sdn", "cascaded"] and args.use_pretrained_weights:
    out_basename += f",pretrained_weights"
  
  if args.train_mode in ["cascaded"] and args.multiple_fcs:
    out_basename += f",multiple_fcs"
  
  save_root = os.path.join(args.experiment_root,
                           args.experiment_name,
                           "experiments",
                           out_basename)
  if not os.path.exists(save_root):
    os.makedirs(save_root)
  print(f"Saving experiment to {save_root}")
  
  # Save args
  if save_args_to_root:
    utils.save_args(args, save_root)

  return save_root


def setup_dataset(args):
  # Data Handler
  data_dict = {
      "dataset_name": args.dataset_name,
      "data_root": args.dataset_root,
      "val_split": args.val_split,
      "test_split": args.test_split,
      "split_idxs_root": args.split_idxs_root,
      "noise_type": args.augmentation_noise_type,
      "load_previous_splits": True,
      "imagenet_params": {
        "target_classes": ["terrier"],
        "max_classes": 10,
      }
  }
  data_handler = DataHandler(**data_dict)

  # Set Loaders
  loaders = {
      "train": data_handler.build_loader("train", args),
      "val": data_handler.build_loader("val", args),
      "test": data_handler.build_loader("test", args),
  }
  print("Data handler loaded.")
  
  return data_handler, loaders


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

def setup_teacher_model(teacher_dir, data_handler, device, args):
  teacher_args_path = os.path.join(teacher_dir, "args.json")
  with open(teacher_args_path, "r") as infile:
    teacher_args = dotdict(json.load(infile))
  
  # Model
  model_dict = {
      "seed": args.random_seed,
      "num_classes": data_handler.num_classes,
      "pretrained": False,
      "train_mode": teacher_args.train_mode,
      "cascaded": teacher_args.cascaded,
      "cascaded_scheme": teacher_args.cascaded_scheme,
      "multiple_fcs": teacher_args.multiple_fcs,
      "lambda_val": teacher_args.lambda_val,
      "tdl_alpha": teacher_args.tdl_alpha,
      "tdl_mode": teacher_args.tdl_mode,
      "noise_var": teacher_args.noise_var,
      "bn_opts": {
          "temporal_affine": teacher_args.bn_time_affine,
          "temporal_stats": teacher_args.bn_time_stats,
      },
      "imagenet": teacher_args.dataset_name == "ImageNet2012",
      "imagenet_pretrained": False, # args.dataset_name == "ImageNet2012",
      "n_channels": 1 if teacher_args.dataset_name == "FashionMNIST" else 3
  }
  
  # Model init op
  if teacher_args.model_key.startswith("resnet"):
    model_init_op = resnet
  elif teacher_args.model_key.startswith("densenet"):
    model_init_op = densenet

  # Initialize net
  print("Instantiating teacher model...")
  teacher_net = model_init_op.__dict__[teacher_args.model_key](**model_dict)
  teacher_net = teacher_net.to(device)
  
  # Load ckpt
  ckpt_dir = os.path.join(teacher_dir, "ckpts")
  ckpt_path = np.sort(glob.glob(f"{ckpt_dir}/*epoch*.pt"))[-1]
  print(f"Loading teacher ckpt {ckpt_path}...")
  model_state_dict = torch.load(ckpt_path, map_location=device)["model"]
  
  # Fix state dict
  fixed_dict = {}
  for key, val in model_state_dict.items():
    if key == "fc.weight" or key == "fc.bias":
      key = f"fc.{key}"
    fixed_dict[key] = val
  teacher_net.load_state_dict(fixed_dict)
  
  print("Teacher Model instantiated.")
  teacher_net = teacher_net.eval()
  return teacher_net


def setup_model(data_handler, device, save_root, args):
  # Model
  model_dict = {
      "seed": args.random_seed,
      "num_classes": data_handler.num_classes,
      "pretrained": False,
      "train_mode": args.train_mode,
      "cascaded": args.cascaded,
      "cascaded_scheme": args.cascaded_scheme,
      "multiple_fcs": args.multiple_fcs,
      "lambda_val": args.lambda_val,
      "tdl_alpha": args.tdl_alpha,
      "tdl_mode": args.tdl_mode,
      "noise_var": args.noise_var,
      "trainable_temp": args.trainable_temp,
      "init_temp": args.init_temp,
      "bn_opts": {
          "temporal_affine": args.bn_time_affine,
          "temporal_stats": args.bn_time_stats,
      },
      "imagenet": args.dataset_name == "ImageNet2012",
      "imagenet_pretrained": False, # args.dataset_name == "ImageNet2012",
      "n_channels": 1 if args.dataset_name == "FashionMNIST" else 3
  }

  # Model init op
  if args.model_key.startswith("resnet"):
    model_init_op = resnet
  elif args.model_key.startswith("densenet"):
    model_init_op = densenet

  # Initialize net
  print("Instantiating model...")
  net = model_init_op.__dict__[args.model_key](**model_dict).to(device)
  number_parameters = utils.count_parameters(net)
  print(f"# model parameters: {number_parameters:,}")
  model_dict["number_parameters"] = number_parameters
  
  print("Model instantiated.")
  
  # Compute inference costs if ic_only / SDN
  if args.train_mode in ["ic_only", "sdn"]:
    all_flops, normed_flops = sdn_utils.compute_inference_costs(
      data_handler, 
      model_dict, 
      args
    )
    net.set_target_inference_costs(
      normed_flops, 
      args.target_IC_inference_costs,
      use_all=args.use_all_ICs
    )
    
    if args.use_all_ICs:
      args.target_IC_inference_costs = normed_flops

  # Save model config
  model_dict["model_key"] = args.model_key
  utils.save_model_config(model_dict, save_root, args.debug)
  
  return net


def fix_dict(model_state_dict, args):
  if args.train_mode == "sdn":
    # Fix dictionary
    fixed_dict = OrderedDict()
    for k, v in model_state_dict.items():
      if k == "fc.weight":
        fixed_dict["fc.fc.weight"] = v
      elif k == "fc.bias":
        fixed_dict["fc.fc.bias"] = v
      else:
        fixed_dict[k] = v
  elif args.train_mode == "cascaded":
    fixed_dict = OrderedDict()
    for k, v in model_state_dict.items():
      if args.cascaded and "running_" in k and len(v.size()):
        continue
      fixed_dict[k] = v
  return fixed_dict


def condition_model(save_root, args):
  # Check mode and load optimizer
  if (args.train_mode == "ic_only" 
      or (args.train_mode in ["sdn", "cascaded"] 
          and args.use_pretrained_weights)
     ):
    baseline_ckpt_path = get_baseline_ckpt_path(save_root, args)
    assert os.path.exists(baseline_ckpt_path), (
        f"Path does not exist: {baseline_ckpt_path}")
    print(f"Loading baseline for ic_only from {baseline_ckpt_path}")
    checkpoint = torch.load(baseline_ckpt_path, map_location="cpu")
    model_state_dict = checkpoint["model"]
    
    # Fix model dict
    fixed_dict = fix_dict(model_state_dict, args)
    
    # Load dict
    net.load_state_dict(fixed_dict, strict=False)
    
    # Set handler params
    if args.train_mode == "ic_only":
      net.freeze_backbone()
      n_epochs = 25
      lr_schedule_milestones = [5, 10, 15, 20]
      weight_decay = 0.0
      lr = 0.001
    elif args.train_mode == "cascaded":
      n_epochs = args.n_epochs
      lr_schedule_milestones = args.lr_milestones
      weight_decay = args.weight_decay
      lr = args.learning_rate
    else:
      n_epochs = 50
      lr_schedule_milestones = [15, 30, 45]
      weight_decay = 0.0
      lr = 0.001
    
    optimizer_dict = {
        "lr": lr,
        "weight_decay": weight_decay, 
    }

    optimizer_init_op = optim.Adam
    lr_schedule_gamma = 0.1
    args.n_epochs = n_epochs
  else:
    optimizer_dict = {
        "lr": args.learning_rate, 
        "momentum": 0.9, 
        "weight_decay": args.weight_decay, 
        "nesterov": True,
    }
    optimizer_init_op = optim.SGD
    lr_schedule_milestones = args.lr_milestones
    lr_schedule_gamma = args.lr_schedule_gamma
  
  returns = {
      "optimizer_dict": optimizer_dict,
      "optimizer_init_op": optimizer_init_op,
      "lr_schedule_milestones": lr_schedule_milestones,
      "lr_schedule_gamma": lr_schedule_gamma,
  }
  return returns


def main(args):
  # Make reproducible
  utils.make_reproducible(args.random_seed)

  # Set Device
  device = torch.device(
    args.device
    if torch.cuda.is_available() and not args.use_cpu
    else "cpu"
  )

  # Setup output directory
  save_root = setup_output_dir(args)

  # Setup dataset loaders
  data_handler, loaders = setup_dataset(args)
  
  # Setup model
  net = setup_model(data_handler, device, save_root, args)
  
  if args.distillation:
    teacher_net = setup_teacher_model(
      args.teacher_dir, 
      data_handler, 
      device, 
      args
    )
  
  # Condition model and get handler opts
  opts = condition_model(save_root, args)
  
  # Init optimizer
  if args.trainable_temp:
    base_params = [v for k, v in net.named_parameters() if not k.startswith("temp_fc")]
    temp_params = [v for k, v in net.named_parameters() if k.startswith("temp_fc")]
    optimizer = opts["optimizer_init_op"]([
      {"params": base_params, **opts["optimizer_dict"]},
      {"params": temp_params, "lr": args.temp_fc_lr},
    ])
  else:
    optimizer = opts["optimizer_init_op"](
      net.parameters(), **opts["optimizer_dict"]
    )

  # Scheduler
  lr_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=opts["lr_schedule_milestones"],
    gamma=opts["lr_schedule_gamma"]
  )

  # train and eval functions
  print("Setting up train and eval functions...")
  train_fxn = train_handler.get_train_loop(
    net.timesteps,
    data_handler.num_classes,
    args,
  )
  
  eval_fxn = eval_handler.get_eval_loop(
    net.timesteps,
    data_handler.num_classes,
    cascaded=args.cascaded,
    flags=args,
    keep_logits=False,
    keep_embeddings=False,
  )
  print("Complete.")

  # Metrics container
  metrics = {
      "train": collections.defaultdict(list),
      "val": collections.defaultdict(list),
      "test": collections.defaultdict(float),
  }
    
  # Main training loop
  try:
    print("Training network...")
    for epoch_i in range(args.n_epochs):
      print(f"\nEpoch {epoch_i+1}/{args.n_epochs}")
      # Train net
      train_params = {
          "net": net, 
          "loader": loaders["train"], 
          "epoch_i": epoch_i, 
          "optimizer": optimizer, 
          "device": device,
      }
      if args.distillation:
        train_params["teacher_net"] = teacher_net
      train_loss, train_acc = train_fxn(**train_params)

      # Log train metrics
      metrics["train"]["loss"].append((epoch_i, train_loss))
      metrics["train"]["acc"].append((epoch_i, train_acc))

      # Update lr scheduler
      lr_scheduler.step()

      # Compute train loss / acc
      if args.train_mode == "baseline":
        train_loss_val = np.mean(train_loss, axis=0)
        train_acc_val = np.mean(train_acc, axis=0) * 100
      else:
        train_loss_val = np.mean(train_loss, axis=0)[-1]
        train_acc_val = np.mean(train_acc, axis=0)[-1] * 100

      stdout_str = (f"\nAvg. Train Loss: {train_loss_val:0.6f} -- "
                    f"Avg. Train Acc: {train_acc_val:0.2f}%")

      if epoch_i % args.eval_freq == 0:
        # Evaluate net
        eval_params = {
            "net": net,
            "loader": loaders["val"],
            "epoch_i": epoch_i,
            "device": device
        }
        if args.distillation:
          eval_params["teacher_net"] = teacher_net
        val_loss, val_acc, _ = eval_fxn(**eval_params)
        stdout_str += (f" -- Avg. Eval Acc: {np.mean(val_acc)*100:0.2f}%")
        # Log eval metrics
        metrics["val"]["loss"].append((epoch_i, val_loss))
        metrics["val"]["acc"].append((epoch_i, val_acc))

      # Stdout
      print(stdout_str)

      # Ckpt model
      if epoch_i % args.save_freq == 0:
        utils.save_model(net, optimizer, save_root, epoch_i, args.debug)

      # Update metrics
      utils.save_metrics(metrics, save_root, args.debug)
  except KeyboardInterrupt:
    print(f"\nUser exited training early @ epoch {epoch_i}.")
  
  # Final validation eval
  eval_params = {
      "net": net,
      "loader": loaders["val"],
      "epoch_i": epoch_i,
      "device": device
  }
  if args.distillation:
    eval_params["teacher_net"] = teacher_net
  val_loss, val_acc, _ = eval_fxn(**eval_params)

  # Log eval metrics
  metrics["val"]["loss"].append((epoch_i, val_loss))
  metrics["val"]["acc"].append((epoch_i, val_acc))

  # Evaluate test set
  print("Evaluating test set.")
  eval_params["loader"] = loaders["test"]
  test_loss, test_acc, _ = eval_fxn(**eval_params)
  
  metrics["test"]["loss"] = test_loss
  metrics["test"]["acc"] = test_acc

  if args.train_mode == "baseline":
    final_mean_test_acc = np.mean(test_acc)
  else:
    try:
      final_mean_test_acc = test_acc.mean(axis=0)[-1]
    except:
      final_mean_test_acc = test_acc.mean(axis=0)
  print(f"Test Acc: {final_mean_test_acc*100:0.2f}%")

  # Save model and metrics
  print("Saving...")
  utils.save_model(net, optimizer, save_root, epoch_i, args.debug)
  utils.save_metrics(metrics, save_root, args.debug)
  print("Fin.")
  
if __name__ == "__main__":
  args = setup_args()
  main(args)
