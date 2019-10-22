# -*- coding: utf-8 -*-
"""Single CPU/GPU training."""

import os
import sys
import argparse
import importlib
import inspect
from math import ceil
from datetime import datetime

sys.path.append("models")
sys.path.append("utils")

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import dataloader
import visualization


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="CMAPSS/FD001", help="Dataset [default: CMAPSS/FD001]")
parser.add_argument("--model", default="frequentist_dense3", help="Model name [default: frequentist_dense3]")
parser.add_argument("--normalization", default="min-max", help="Normalization (min-max | z-score) [default: min-max]")
parser.add_argument("--log_dir", default="log/CMAPSS/FD001/min-max/frequentist_dense3", help="Log dir [default: log/CMAPSS/FD001/min-max/frequentist_dense3]")
parser.add_argument("--max_epoch", type=int, default=250, help="Epoch to run [default: 250]")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size during training/evaluation [default: 512]")
parser.add_argument("--optimizer", default="adam", help="Optimizer (adam | momentum) [default: adam]")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate [default: 0.001]")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (used only for momentum optimizer) [default: 0.9]")
parser.add_argument("--step_size", type=int, default=200, help="Step size for learning rate decay [default: 200]")
parser.add_argument("--gamma", type=float, default=0.1, help="Learning rate decay factor [default: 0.1]")
parser.add_argument("--num_mc", type=int, default=1, help="Number of Monte Carlo samples [default: 1]")
parser.add_argument("--metric", default="rmse", help="Metric according to which the best model is saved (rmse | mae | score) [default: rmse]")
parser.add_argument("--max_rul", type=int, default=10000, help="Label rectification threshold [default: 10000]")
parser.add_argument("--quantity", type=float, default=1.0, help="Ratio of training data to use [default: 1.0]")
parser.add_argument("--visualize_step", type=int, default=50, help="Visualize figures every 'visualize_step' epochs [default: 50]")
parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
FLAGS = parser.parse_args()

model_type = FLAGS.model.split("_")[0]
DATASET = FLAGS.dataset
NORMALIZATION = FLAGS.normalization
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
OPTIMIZER = FLAGS.optimizer
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
STEP_SIZE = FLAGS.step_size
GAMMA = FLAGS.gamma
NUM_MC = FLAGS.num_mc
METRIC = FLAGS.metric
MAX_RUL = FLAGS.max_rul
QUANTITY = FLAGS.quantity
VISUALIZE_STEP = FLAGS.visualize_step
RESUME = FLAGS.resume

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type not in ["frequentist", "bayesian"]:
    print("Model file name must start with either 'bayesian' or 'frequentist', got '" + model_type + "'.")
    sys.exit(0)

if NORMALIZATION not in ["min-max", "z-score"]:
    print("'normalization' must be either 'min-max' or 'z-score', got '" + NORMALIZATION + "'.")
    sys.exit(0)

if BATCH_SIZE < 1:
    print("'batch_size' must be a value greater or equal to 1, got %.2f" % BATCH_SIZE + ".")
    sys.exit(0)

if MAX_EPOCH < 1:
    print("'max_epoch' must be a value greater or equal to 1, got %.2f" % MAX_EPOCH + ".")
    sys.exit(0)

if OPTIMIZER not in ["adam", "momentum"]:
    print("'optimizer' must be either 'adam' or 'momentum', got '" + OPTIMIZER + "'.")
    sys.exit(0)

if BASE_LEARNING_RATE <= 0:
    print("'learning_rate' must be a value greater than 0, got %.2f" % BASE_LEARNING_RATE + ".")
    sys.exit(0)

if MOMENTUM < 0:
    print("'momentum' must be a value greater or equal to 0, got %.2f" % MOMENTUM + ".")
    sys.exit(0)

if STEP_SIZE < 1:
    print("'step_size' must be a value greater or equal to 1, got %d" % STEP_SIZE + ".")
    sys.exit(0)

if GAMMA <= 0 or GAMMA > 1.0:
    print("'gamma' must be a value within (0, 1], got %.2f" % GAMMA + ".")
    sys.exit(0)

if model_type == "frequentist" and NUM_MC != 1:
    print("'num_mc' must be equal to 1 for 'frequentist' models, got %d" % NUM_MC + ".")
    sys.exit(0)

if model_type == "bayesian" and  NUM_MC < 1:
    print("'num_mc' must be greater or equal to 1 for 'bayesian' models, got %d" % NUM_MC + ".")
    sys.exit(0)

if METRIC not in ["rmse", "mae", "score"]:
    print("'metric' must be either 'rmse' or 'mae' or 'score', got '" + METRIC + "'.")
    sys.exit(0)

if MAX_RUL < 0:
    print("'max_rul' must be a value greater or equal to 0, got %.2f" % MAX_RUL + ".")
    sys.exit(0)

if QUANTITY <= 0 or QUANTITY > 1.0:
    print("'quantity' must be a value within (0, 1], got %.2f" % QUANTITY + ".")
    sys.exit(0)

if VISUALIZE_STEP < 1:
    print("'visualize_step' must be a value greater or equal to 1, got %.2f" % VISUALIZE_STEP + ".")
    sys.exit(0)

module = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join("models", FLAGS.model + ".py")
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system("cp %s %s" % (MODEL_FILE, LOG_DIR)) # backup of model definition
os.system("cp train.py %s" % LOG_DIR) # backup of training procedure
LOG_FOUT = open(os.path.join(LOG_DIR, "log_train.txt"), "w") # open log file
LOG_FOUT.write(str(FLAGS) + "\n")

# TensorBoard writer
WRITER = SummaryWriter(LOG_DIR)


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)


log_string("pid: %s" % str(os.getpid()))
log_string("use_cuda: %s" % str(torch.cuda.is_available()))

# dataset
TRAIN_DATASET = dataloader.Dataloader(root="./datasets", dataset=DATASET, split="train", normalization=NORMALIZATION, batch_size=BATCH_SIZE, max_rul=MAX_RUL, quantity=QUANTITY)

VALIDATION_DATASET = dataloader.Dataloader(root="./datasets", dataset=DATASET, split="validation", normalization=NORMALIZATION, batch_size=BATCH_SIZE)

log_string("Dataset: " + DATASET)

input_size = (TRAIN_DATASET.num_channels, TRAIN_DATASET.window, TRAIN_DATASET.num_features)

# validation set non-empty
VALIDATION = len(VALIDATION_DATASET) > 0
   
# get class name
clss = [m[0] for m in inspect.getmembers(module, inspect.isclass) if m[1].__module__ == FLAGS.model]
assert len(clss) == 1
cls = clss[0]

# init model
MODEL = getattr(module, cls)(input_size).to(DEVICE)
        
# init optimizer
if OPTIMIZER == "momentum":
    optimizer = optim.SGD(MODEL.parameters(), lr=BASE_LEARNING_RATE, momentum=MOMENTUM)
elif OPTIMIZER == "adam":
    optimizer = optim.Adam(MODEL.parameters(), lr=BASE_LEARNING_RATE)

# learning rate decay schedule
scheduler = optim.lr_scheduler.StepLR(optimizer, STEP_SIZE, gamma=GAMMA)

if RESUME:
    # load checkpoint
    log_string("Resuming " + str(MODEL.__class__).split(".")[-1].split("'")[0] + " from checkpoint...")
    load_path = os.path.join(LOG_DIR, "checkpoint.pth.tar")
    checkpoint = torch.load(load_path, map_location=DEVICE)
    start_epoch = checkpoint["epoch"] + 1
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
else:
    log_string("Building " + str(MODEL.__class__).split(".")[-1].split("'")[0] + "...")
    start_epoch = 0

log_string("Done.")

EPOCH_CNT = start_epoch


def train():
    """Train the model."""
    time_start = datetime.now()
    log_string("**** start time: " + str(time_start) + " ****")
  
    global EPOCH_CNT

    # log model architecture
    MODEL.summary(log_string)

    for epoch in range(start_epoch, MAX_EPOCH, 1):
        log_string("**** EPOCH %03d ****" % epoch)
        sys.stdout.flush()
        
        train_loss = train_one_epoch()

        WRITER.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
        WRITER.add_scalar("train/loss", train_loss, epoch)

        # weights
        if epoch % VISUALIZE_STEP == 0 or epoch == MAX_EPOCH - 1:
            names, qmeans, qstds = MODEL.get_weight_statistics()
            fig = visualization.plot_weight_distr(names, qmeans, qstds)
            WRITER.add_figure("weights", fig, global_step=epoch)

        # learning rate decay step
        scheduler.step()

        if VALIDATION:
            # validation set non-empty: run validation routine
            with torch.no_grad():
                eval_loss, eval_rmse, eval_mae, eval_score, epistemic = eval_one_epoch()
            WRITER.add_scalar("eval/loss", eval_loss, epoch)
            WRITER.add_scalar("eval/rmse", eval_rmse, epoch)
            WRITER.add_scalar("eval/mae", eval_mae, epoch)
            WRITER.add_scalar("eval/score", eval_score, epoch)
            WRITER.add_scalar("eval/epistemic", epistemic, epoch)

        EPOCH_CNT += 1

        delta = datetime.now() - time_start
        log_string("elapsed time: " + str(delta))

    # save last model to disk
    save_path = os.path.join(LOG_DIR, "checkpoint.pth.tar")
    checkpoint = {
        "epoch":                EPOCH_CNT - 1,
        "model_state_dict":     MODEL.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, save_path)
    log_string("checkpoint saved in file: %s" % save_path)
    
    log_string("**** end time: " + str(datetime.now()) + " ****")


def train_one_epoch():
    """Train one epoch.

    Returns
    -------
    float
        Training loss.
    """
    time_start = datetime.now()

    MODEL.train()
    
    loss = 0.
    loss_10_batches = 0.
    batch_idx = 0
    m = ceil(len(TRAIN_DATASET) / BATCH_SIZE)
    beta = 1 / m if model_type == "bayesian" else 0

    log_string("---- EPOCH %03d TRAINING ----" % EPOCH_CNT)
    log_string(str(time_start))

    log_string("learning rate: " + str(optimizer.param_groups[0]["lr"]))

    while TRAIN_DATASET.has_next_batch():
        # (BATCH_SIZE, TRAIN_DATASET.num_channels, TRAIN_DATASET.window, TRAIN_DATASET.num_features), (BATCH_SIZE)
        sample_batch, label_batch = TRAIN_DATASET.next_batch()
        bsize = sample_batch.shape[0]
        sample_batch = torch.tensor(sample_batch).float().to(DEVICE)
        label_batch = torch.tensor(label_batch).float().to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # compute prediction
        pred_batch = MODEL(sample_batch)

        # compute loss
        loss_batch = MODEL.loss(pred_batch, label_batch, beta)

        # backpropagation
        loss_batch.backward()

        # optimization step
        optimizer.step()

        loss_10_batches += loss_batch / bsize
        loss += loss_batch / bsize
        if (batch_idx + 1) % 10 == 0:
            loss_10_batches /= 10
            log_string(" ---- batch: %03d ----" % (batch_idx + 1))
            log_string("mean loss: %.2f" % loss_10_batches.item())
            loss_10_batches = 0.
        batch_idx += 1

    loss /= float(batch_idx)
    loss = loss.item()
    log_string("train mean loss: %.2f" % loss)

    TRAIN_DATASET.reset()

    delta = datetime.now() - time_start
    log_string("epoch train time: " + str(delta))
    return loss


def eval_one_epoch():
    """Evaluate one epoch.

    Returns
    -------
    (float, float, float, float, float)
        Evaluation loss, RMSE, MAE, score, epistemic.
    """
    time_start = datetime.now()
    
    MODEL.eval()

    loss = 0.
    sse = 0.
    sae = 0.
    score = 0.
    all_sample = torch.FloatTensor().to(DEVICE)
    all_label = torch.FloatTensor().to(DEVICE)
    all_pred_mc = torch.FloatTensor().to(DEVICE)
    all_epistemic = torch.FloatTensor().to(DEVICE)
    batch_idx = 0
    m = ceil(len(VALIDATION_DATASET) / BATCH_SIZE)
    beta = 1 / m if model_type == "bayesian" else 0
       
    log_string("---- EPOCH %03d EVALUATION ----" % EPOCH_CNT)
    log_string(str(time_start))
    
    while VALIDATION_DATASET.has_next_batch():
        # (BATCH_SIZE, VALIDATION_DATASET.num_channels, VALIDATION_DATASET.window, VALIDATION_DATASET.num_features), (BATCH_SIZE)
        sample_batch, label_batch = VALIDATION_DATASET.next_batch()
        bsize = sample_batch.shape[0]
        sample_batch = torch.tensor(sample_batch).float().to(DEVICE)
        label_batch = torch.tensor(label_batch).float().to(DEVICE)
    
        # compute prediction
        pred_batch_mc = torch.stack([MODEL(sample_batch) for _ in range(NUM_MC)])

        epistemic_var = torch.var(pred_batch_mc, dim=0, unbiased=False)
        pred_batch = torch.mean(pred_batch_mc, dim=0)

        # compute loss (NUM_MC)
        loss_batch = MODEL.loss(pred_batch, label_batch, beta)
        
        # batch sse (NUM_MC)
        sse_batch = torch.sum((pred_batch - label_batch) ** 2)

        # batch sae (NUM_MC)
        sae_batch = torch.sum(torch.abs(pred_batch - label_batch))

        # batch score (NUM_MC)
        delta = pred_batch - label_batch
        mask = delta < 0
        delta[mask] /= -13
        delta[~mask] /= 10
        score_batch = torch.sum(torch.exp(delta) - 1)
        
        loss += loss_batch / bsize
        sse += sse_batch
        sae += sae_batch
        score += score_batch

        all_sample = sample_batch if len(all_sample) == 0 else torch.cat([all_sample, sample_batch], 0)
        all_label = label_batch if len(all_label) == 0 else torch.cat([all_label, label_batch], 0)
        all_pred_mc = pred_batch_mc if len(all_pred_mc) == 0 else torch.cat([all_pred_mc, pred_batch_mc], 1)
        all_epistemic = epistemic_var if len(all_epistemic) == 0 else torch.cat([all_epistemic, epistemic_var], 0)

        batch_idx += 1

    loss /= float(batch_idx)
    mse = sse / float(len(VALIDATION_DATASET))
    rmse = torch.sqrt(mse)
    mae = sae / float(len(VALIDATION_DATASET))
    predictive_mean = torch.mean(all_pred_mc, dim=0)
    predictive_var = all_epistemic
    predictive_std = torch.sqrt(predictive_var)
   
    loss = loss.item()
    rmse = rmse.item()
    mae = mae.item()
    score = score.item()
    epistemic = torch.mean(all_epistemic).item()

    log_string("eval mean loss: %.2f" % loss)
    log_string("eval rmse: %.2f" % rmse)
    log_string("eval mae: %.2f" % mae)
    log_string("eval score: %.2f" % score)
    log_string("epistemic: %.2f" % epistemic)

    # predictive distribution
    if EPOCH_CNT % VISUALIZE_STEP == 0 or EPOCH_CNT == MAX_EPOCH - 1:
        names, qmeans, qstds = MODEL.get_weight_statistics()
        fig = visualization.plot_predictive_distr(all_pred_mc[:, 0].detach().cpu().numpy(),
                                                  predictive_mean[0].detach().cpu().numpy(),
                                                  predictive_std[0].detach().cpu().numpy())
        WRITER.add_figure("predictive_distribution_rul{:03d}".format(int(all_label[0].detach().cpu().numpy())), fig, global_step=EPOCH_CNT)

    VALIDATION_DATASET.reset()

    delta = datetime.now() - time_start
    log_string("eval time: " + str(delta))
    return loss, rmse, mae, score, epistemic


if __name__ == "__main__":
    """Train."""
    train()
    LOG_FOUT.close()
