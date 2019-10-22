# -*- coding: utf-8 -*-
"""Single CPU/GPU evaluation."""

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

import dataloader


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="CMAPSS/FD001", help="Dataset [default: CMAPSS/FD001]")
parser.add_argument("--model", default="frequentist_dense3", help="Model name [default: frequentist_dense3]")
parser.add_argument("--normalization", default="min-max", help="Normalization (min-max | z-score) [default: min-max]")
parser.add_argument("--model_path", default="log/CMAPSS/FD001/min-max/frequentist_dense3/checkpoint.pth.tar", help="Model checkpoint file path [default: log/CMAPSS/FD001/min-max/frequentist_dense3/checkpoint.pth.tar]")
parser.add_argument("--dump_dir", default="dump/CMAPSS/FD001/min-max/frequentist_dense3", help="Dump dir [default: dump/CMAPSS/FD001/min-max/frequentist_dense3]")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size during evaluation [default: 512]")
parser.add_argument("--num_mc", type=int, default=1, help="Number of Monte Carlo samples [default: 1]")
FLAGS = parser.parse_args()

model_type = FLAGS.model.split("_")[0]
DATASET = FLAGS.dataset
NORMALIZATION = FLAGS.normalization
MODEL_PATH = FLAGS.model_path
BATCH_SIZE = FLAGS.batch_size
NUM_MC = FLAGS.num_mc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type not in ["frequentist", "bayesian"]:
    print("Model file name must start with either 'bayesian' or 'frequentist', got '" + model_type + "'.")
    sys.exit(0)

if NORMALIZATION not in ["min-max", "z-score"]:
    print("'normalization' must be either 'min-max' or 'z-score', got '" + NORMALIZATION + "'.")
    sys.exit(0)

if not os.path.exists(MODEL_PATH):
    print(MODEL_PATH + " does not exist.")
    sys.exit(0)

if BATCH_SIZE < 1:
    print("'batch_size' must be a value greater or equal to 1, got %.2f" % BATCH_SIZE + ".")
    sys.exit(0)

if model_type == "frequentist" and NUM_MC != 1:
    print("'num_mc' must be equal to 1 for 'frequentist' models, got %d" % NUM_MC + ".")
    sys.exit(0)

if model_type == "bayesian" and  NUM_MC < 1:
    print("'num_mc' must be greater or equal to 1 for 'bayesian' models, got %d" % NUM_MC + ".")
    sys.exit(0)

module = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join("models", FLAGS.model + ".py")
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.makedirs(DUMP_DIR)
os.system("cp evaluate.py %s" % DUMP_DIR) # backup of evaluation procedure
LOG_FOUT = open(os.path.join(DUMP_DIR, "log_evaluate.txt"), "w")
LOG_FOUT.write(str(FLAGS) + "\n")


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)


log_string("pid: %s" % str(os.getpid()))
log_string("use_cuda: %s" % str(torch.cuda.is_available()))

# dataset
DATA_PATH = os.path.join("datasets", DATASET)

TEST_DATASET = dataloader.Dataloader(root="./datasets", dataset=DATASET, split="test", normalization=NORMALIZATION, batch_size=BATCH_SIZE)

log_string("Dataset: " + DATASET)

input_size = (TEST_DATASET.num_channels, TEST_DATASET.window, TEST_DATASET.num_features)

# get class name
clss = [m[0] for m in inspect.getmembers(module, inspect.isclass) if m[1].__module__ == FLAGS.model]
assert len(clss) == 1
cls = clss[0]

# init model
MODEL = getattr(module, cls)(input_size).to(DEVICE)
  
# load checkpoint
log_string("Restoring " + str(MODEL.__class__).split(".")[-1].split("'")[0] + "...")
load_path = os.path.join(MODEL_PATH)
checkpoint = torch.load(load_path, map_location=DEVICE)
MODEL.load_state_dict(checkpoint["model_state_dict"])
log_string("Done.")


def evaluate():
    """Evaluate the model."""
    time_start = datetime.now()
    log_string("**** start time: " + str(time_start) + " ****")
  
    # log model architecture
    MODEL.summary(log_string)

    with torch.no_grad():
        eval_loss, eval_rmse, eval_mae, eval_score, epistemic = eval_one_epoch()

    log_string("**** end time: " + str(datetime.now()) + " ****")


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
    m = ceil(len(TEST_DATASET) / BATCH_SIZE)
    beta = 1 / m if model_type == "bayesian" else 0
       
    log_string(str(time_start))
    log_string("ground truth | pred +/- std:")

    while TEST_DATASET.has_next_batch():
        # (BATCH_SIZE, TEST_DATASET.num_channels, TEST_DATASET.window, TEST_DATASET.num_features), (BATCH_SIZE)
        sample_batch, label_batch = TEST_DATASET.next_batch()
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
    mse = sse / float(len(TEST_DATASET))
    rmse = torch.sqrt(mse)
    mae = sae / float(len(TEST_DATASET))
    predictive_mean = torch.mean(all_pred_mc, dim=0)
    predictive_var = all_epistemic
    predictive_std = torch.sqrt(predictive_var)
   
    loss = loss.item()
    rmse = rmse.item()
    mae = mae.item()
    score = score.item()
    epistemic = torch.mean(all_epistemic).item()

    for i in range(predictive_mean.shape[0]):
        log_string("%.2f | %.2f +/- %.2f" % (all_label[i].item(), predictive_mean[i].item(), predictive_std[i].item()))

    log_string("eval mean loss: %.2f" % loss)
    log_string("eval rmse: %.2f" % rmse)
    log_string("eval mae: %.2f" % mae)
    log_string("eval score: %.2f" % score)
    log_string("epistemic: %.2f" % epistemic)
    log_string("epoch: %d" % checkpoint["epoch"])

    log_string("ground truth std: %.2f" % torch.std(all_label, unbiased=False).item())
    log_string("pred std: %.2f" % torch.std(predictive_mean, unbiased=False).item())
    
    TEST_DATASET.reset()

    delta = datetime.now() - time_start
    log_string("eval time: " + str(delta))
    return loss, rmse, mae, score, epistemic

        
if __name__ == "__main__":
    """Evaluate."""
    evaluate()
    LOG_FOUT.close()
