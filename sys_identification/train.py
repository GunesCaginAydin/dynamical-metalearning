"""
TO IMPLEMENT: 

progress bar, estimated time
postprocessor
optimization of algorithms
scheduler.step()
checks

utils - process ???
"""

import time
import torch
import wandb
import sys
from datasets import *
from utils import arguments
from losses import getloss
from tqdm import tqdm
from rich.progress import track

torch.set_float32_matmul_precision("medium")
torch.use_deterministic_algorithms(False)
torch.cuda.empty_cache()

args = arguments().parse_arguments()

if args.log_wandb:
    wandb.init(
    group='Ds_1',
    project="sysid-Franka",
    name = f"batch{args.batch_size}_embd{args.n_embd}_lay{args.n_layer}_heads{args.n_head}_{args.loss_function}",
    config=vars(args))

datasets = dataset(args=args)
print(datasets)

torch.set_float32_matmul_precision("high")

datasets.resolve_datasets()
datasets.initialize_model()

datasetdict = datasets.getgeneration()
datadict = datasets.getmetadata()
datasets.getdataset()
datasets.getmodel()

datasets.settime(time.perf_counter())
cum_validation_loss = 0
modelargs, model, optimizer, scheduler = datasets.getmodel()

for data in datadict['traindatalist']:
    datasets.load(data)
    datasets.configure_dataset()

    training_dataset, validation_dataset, test_dataset = datasets.getdataset()

    model.train()
    
    for iter_num, (ubatch,ybatch) in tqdm(enumerate(training_dataset, start=datasets.iter)):
        ubatch, ybatch = ubatch.cuda(non_blocking=True), ybatch.cuda(non_blocking=True)
        ubatch, umean, ustd = datasets.normalizestd(ubatch)
        ybatch, ymean, ystd = datasets.normalizestd(ybatch)

        uctx,unew = datasets.seperate_context(ubatch)
        yctx,ynew = datasets.seperate_context(ybatch)

        optimizer.zero_grad()
        ysim = model(yctx, uctx, unew)
        training_loss = getloss(args, yact=ynew, ysim=ysim)
        datasets.setlosslist(training_loss=training_loss.item())
        training_loss.backward()
        optimizer.step()

        if ((iter_num % args.validate_at) == 0) and iter_num>0:

            model.eval()
            with torch.inference_mode():

                for eval_iter, (ubatchv, ybatchv) in enumerate(validation_dataset):

                    ubatchv, ybatchv = ubatchv.cuda(non_blocking=True), ybatchv.cuda(non_blocking=True)
                    ubatchv, umeanv, ustdv = datasets.normalizestd(ubatchv)
                    ybatchv, ymeanv, ystdv = datasets.normalizestd(ybatchv)

                    uctxv,unewv = datasets.seperate_context(ubatchv)
                    yctxv,ynewv = datasets.seperate_context(ybatchv)
                    
                    ysimv = model(yctxv, uctxv, unewv)

                    ysimv = datasets.denormalizestd(ysimv, ymeanv, ystdv)
                    ynewv = datasets.denormalizestd(ynewv, ymeanv, ystdv)

                    validation_loss = getloss(args, yact=ynewv, ysim=ysimv)
                    cum_validation_loss += validation_loss.item()
                    datasets.setlosslist(validation_loss=validation_loss.item())

                validation_loss_interval = cum_validation_loss/eval_iter
                print(f"\n{iter_num=} {validation_loss_interval=:.4f}\n")

                if validation_loss_interval < datasets.best_validation_loss[1]:
                    datasets.setlosslist(best_validation_loss=validation_loss_interval)
                    datasets.setcheckpoint(iter=iter_num, curtime=time.perf_counter())   

            model.train()

        scheduler.step()
        print(f"\n{iter_num=} {training_loss.item()=:.4f} {scheduler.get_last_lr()=}")

    datasets.setmodel(modelargs, model, optimizer, scheduler)

    datasets.setcheckpoint(iter_num, time.perf_counter())
    torch.save(datasets.checkpoint, f'{datasets.modelpath}/{datasets.modelname}')
    datasets.reset()

torch.cuda.empty_cache()
datasets.gettime(time.perf_counter())

datasets.writer()
datasets.logger()

if args.log_wandb:
    wandb.finish()

