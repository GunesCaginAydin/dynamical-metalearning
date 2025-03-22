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
import tqdm
import wandb
from utils import *
from losses import getloss

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

datasets.resolve_datasets()
datasets.initialize_model()
datasets.settime(time.perf_counter())

for data in datasets.datadict['datalist']:

    datasets.load(data)
    datasetdict = datasets.getgeneration()

    datasets.configure_dataset(data)
    training_dataset, validation_dataset, validation_loss_list, training_loss_list, best_validation_loss = datasets.getdataset()
    modelargs, model, optimizer = datasets.getmodel()

    for iter_num, (ybatch, ubatch) in tqdm.tqdm(enumerate(training_dataset, start=datasets.iter)):
           
        ubatch, umean, ustd = datasets.normalize(ubatch)
        ybatch, ymean, ystd = datasets.normalize(ybatch)
        
        uctx,unew = datasets.seperate_context(ubatch)
        yctx,ynew = datasets.seperate_context(ybatch)

        optimizer.zero_grad()
        ysim = model(yctx, uctx, unew)
        training_loss = getloss(data, args, yact=ynew, ysim=ysim)
        training_loss_list.append(training_loss.item())
        training_loss.backward()
        optimizer.step()
        
        if iter_num % args.log_iteration == 0:
            datasets.configure_learning_rate()
            print(f"\n{datasets.iter_num=} {datasets.training_loss=:.4f} {datasets.validation_loss=:.4f} {datasets.learning_rate=}\n")

            if args.log_wandb:
                wandb.log({"loss": training_loss,
                           "loss_val": validation_loss})
                
        if (iter_num % args.evaluation_iteration == 0) and iter_num > 0:
            model.eval()
            with torch.inference_mode():
                validation_loss = getloss(data, validation_dataset, args)
            model.train()
            validation_loss_list.append(validation_loss)
            print(f"\n{iter_num=} {validation_loss=:.4f}\n")

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                datasets.setcheckpoint()

        if iter_num == args.max_iters-1:
            break

    datasets.setdataset(training_dataset, validation_dataset, training_loss_list, validation_loss_list, best_validation_loss)
    datasets.setmodel(modelargs, model, optimizer)

    datasets.setcheckpoint()
    torch.save(datasets.checkpoint, datasets.modelpath)
    datasets.reset()

torch.cuda.empty_cache()

datasets.gettime(time.perf_counter())

datasets.writer()
datasets.logger()

if args.log_wandb:
    wandb.finish()

# implement writer and logger
# run cProfile to optimize training