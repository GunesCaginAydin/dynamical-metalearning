import time
import torch
import wandb
import sys

from datasets import *
from utils import arguments
from losses import defaultloss
from tqdm import tqdm
from rich.progress import track
from einops import reduce
from torch import nn

args = arguments().parse_arguments()

datasets = dataset(args=args) # initialize dataset object for training
print(datasets)

datasets.resolve_datasets() # define training/testing data/folder

datasets.initialize_model() # create model/initialize for finetuning

datasetdict = datasets.getgeneration() # data
print(datasetdict)

datadict = datasets.getmetadata() # metadata
print(datadict)

torch.cuda.empty_cache()
datasets.settime(time.perf_counter()) # start timer

modelargs, model, optimizer, scheduler = datasets.getmodel() # model
ema, ema_model = datasets.getema()

for data in datadict['trainlist']:
    print(data)
    datasets.load(data) # load current data from trainlist
    datasets.configure_datasets() # configure data for training

    training_dataset, validation_dataset, test_dataset  = datasets.getdataset() # dataset

    model.train()
    for iter_num, batches in tqdm(enumerate(training_dataset, start=datasets.iter)):
        ubatch, ybatch = batches[0].cuda(non_blocking=True), batches[1].cuda(non_blocking=True)
        if args.controller: # if controller implemented import other variables
            tbatch, kpbatch, kvbatch, kibatch = batches[2].cuda(non_blocking=True), batches[3].cuda(non_blocking=True), batches[4].cuda(non_blocking=True), batches[5].cuda(non_blocking=True)
            ytarget, taux = datasets.normalize(x=tbatch)
            targetctx, targetnew = datasets.seperate_context(ytarget)
            kpbatch, kpaux = datasets.normalize(x=kpbatch,ndim=1)
            kvbatch, kvaux = datasets.normalize(x=kvbatch,ndim=1)
            kibatch, kiaux = datasets.normalize(x=kibatch,ndim=1)
            controller_gains = torch.cat((kpbatch,kvbatch,kibatch),dim=1)

        uraw = copy.deepcopy(ubatch)
        yraw = copy.deepcopy(ybatch)

        ubatch, uaux = datasets.normalize(x=ubatch)
        ybatch, yaux = datasets.normalize(x=ybatch)

        uctx,unew = datasets.seperate_context(ubatch)
        yctx,ynew = datasets.seperate_context(ybatch)
    
        optimizer.zero_grad()
 
        if args.subcommand=='transformer': # embedding u + y_ctx -> y_new
            if args.trftype==1: # u | y sim 
                ysim = model(yctx, uctx, unew)
            elif args.trftype==2: # track | y sim
                ysim = model(yctx, targetctx, targetnew)
            training_loss = defaultloss(args).getloss(yact=ynew, ysim=ysim)

        elif args.subcommand=='diffuser': # inpainting -> y | 1000 no_cond
            if args.diftype==1:
                xd = torch.cat((ubatch,ybatch),dim=2) 
                est, noise, xstart = model.get_loss_params(x=xd, cond=yctx, force=ubatch)
            elif args.diftype==2:
                xd = torch.cat((ytarget,ybatch),dim=2) 
                est, noise, xstart = model.get_loss_params(x=xd, cond=yctx, force=ytarget)
            weights = model.loss_weights
            if args.predict_epsilon:
                training_loss = defaultloss(args=args, weights=weights).getloss(ysim=est, yact=noise).mean()
            else:
                training_loss = defaultloss(args=args, weights=weights).getloss(ysim=est, yact=xstart).mean()
            
        elif args.subcommand=='rechorUnet' or args.subcommand=='rechorTrf':
            if args.rechortype==1: # uy | 200 condition -> y | 1000, local_cond or global_cond
                obs = torch.cat(
                    (ubatch, ybatch),
                     dim=2
                    )
                action = ybatch
            elif args.rechortype==2: # u | 1000 condition -> y | 800, local_cond or global_cond
                obs = ubatch
                action = ybatch
            elif args.rechortype==3: # u | 1000 , y | 200 condition -> y | 800, local_cond or global_cond
                obs = ybatch
                action = ybatch
            elif args.rechortype==4: # inpainting -> y | 1000, no_cond
                obs = ubatch
                action = ybatch
            elif args.rechortype==5: # target conditioning -> y | 1000
                obs = ytarget
                action = ybatch
            elif args.rechortype==6: # controller Kc conditioning -> y | 1000
                B,G = controller_gains.shape
                obs = controller_gains if args.controlgain_horizon==1 else controller_gains.unsqueeze(dim=1).repeat(1,args.controlgain_horizon,1)
                action = ybatch
            batch = {
                'action':action,
                'obs':obs
            }
            pred, target, mask = model.compute_loss(batch)
            training_loss = defaultloss(args).getloss(ysim=pred, yact=target, reduction='none')
            training_loss = training_loss * mask.type(training_loss.dtype)
            training_loss = reduce(training_loss, 'b ... -> b (...)', 'mean')
            training_loss = training_loss.mean()

        datasets.setlosslist(training_loss=training_loss.item())
        training_loss.backward()
        optimizer.step()
        scheduler.step()

        if args.include_ema:
            ema.step(model)

        if ((iter_num % args.validate_at) == 0) and iter_num>0:
            cum_validation_loss = 0
            model.eval()
            with torch.inference_mode():
                for eval_iter, batchesv in enumerate(validation_dataset):
                    ubatchv, ybatchv = batchesv[0].cuda(non_blocking=True), batchesv[1].cuda(non_blocking=True)
                    if args.controller:
                        tbatchv, kpbatchv, kvbatchv, kibatchv = batchesv[2].cuda(non_blocking=True), batchesv[3].cuda(non_blocking=True), batchesv[4].cuda(non_blocking=True), batchesv[5].cuda(non_blocking=True)
                        ytargetv, tauxv = datasets.normalize(x=tbatchv)
                        targetctxv, targetnewv = datasets.seperate_context(ytargetv)
                        kpbatchv, kpauxv = datasets.normalize(x=kpbatchv,ndim=1)
                        kvbatchv, kvauxv = datasets.normalize(x=kvbatchv,ndim=1)
                        kibatchv, kiauxv = datasets.normalize(x=kibatchv,ndim=1)
                        controller_gainsv = torch.cat((kpbatchv,kvbatchv,kibatchv),dim=1)
                    urawv = copy.deepcopy(ubatchv)
                    yrawv = copy.deepcopy(ybatchv)

                    ubatchv, uauxv = datasets.normalize(x=ubatchv)
                    ybatchv, yauxv = datasets.normalize(x=ybatchv)

                    uctxv,unewv = datasets.seperate_context(ubatchv)
                    yctxv,ynewv = datasets.seperate_context(ybatchv)

                    if args.subcommand=='transformer': # embedding u + y_ctx -> y_new 
                        if args.trftype==1: # u | y sim 
                            ysimv = model(yctxv, uctxv, unewv)
                        elif args.trftype==2: # track | y sim
                            ysimv = model(yctxv, targetctxv, targetnewv)
                        ysimv = datasets.denormalize(nx=ysimv, aux=yauxv)
                        ynewv = datasets.denormalize(nx=ynewv, aux=yauxv)
                        validation_loss = defaultloss(args).getloss(yact=ynewv, ysim=ysimv)

                    elif args.subcommand=='diffuser': # inpainting -> y | 1000 no_cond
                        if args.diftype==1:
                            xdv = torch.cat((ubatchv,ybatchv),dim=2) 
                            estv, noisev, xstartv = model.get_loss_params(x=xdv, cond=yctxv, force=ubatchv)
                        elif args.diftype==2:
                            xdv = torch.cat((ytargetv,ybatchv),dim=2)  
                            estv, noisev, xstartv = model.get_loss_params(x=xdv, cond=yctxv, force=ytargetv)
                        if args.predict_epsilon:
                            validation_loss = defaultloss(args).getloss(ysim=estv, yact=noisev)
                        else:
                            validation_loss = defaultloss(args).getloss(ysim=estv, yact=xstartv)
                    
                    elif args.subcommand=='rechorUnet' or args.subcommand=='rechorTrf':
                        if args.rechortype==1: # uy | 200 condition -> y | 1000, local_cond or global_cond
                            obsv = torch.cat(
                                (ubatchv, ybatchv),
                                dim=2
                                )
                            actionv = ybatchv
                        elif args.rechortype==2: # u | 1000 condition -> y | 800, local_cond or global_cond
                            obsv = ubatchv
                            actionv = ybatchv
                        elif args.rechortype==3: # u | 1000 , y | 200 condition -> y | 800, local_cond or global_cond
                            obsv = ybatchv
                            actionv = ybatchv
                        elif args.rechortype==4: # inpainting -> y | 1000, no_cond
                            obsv = ubatchv
                            actionv = ybatchv
                        elif args.rechortype==5: # target conditioning -> y | 1000
                            obsv = ytargetv
                            actionv = ybatchv
                        elif args.rechortype==6: # controller Kc conditioning -> y | 1000
                            Bv,Gv = controller_gainsv.shape
                            obsv = controller_gainsv if args.controlgain_horizon==1 else controller_gainsv.unsqueeze(dim=1).repeat(1,args.controlgain_horizon,1)
                            actionv = ybatchv

                        batchv = {
                            'action':actionv,
                            'obs':obsv
                        }
                        predv, targetv, maskv = model.compute_loss(batchv) if not args.include_ema else ema_model.compute_loss(batchv)
                        validation_loss = defaultloss(args).getloss(ysim=predv, yact=targetv, reduction='none')
                        validation_loss = validation_loss * maskv.type(validation_loss.dtype)
                        validation_loss = reduce(validation_loss, 'b ... -> b (...)', 'mean')
                        validation_loss = validation_loss.mean()

                    cum_validation_loss += validation_loss.item()
                    datasets.setlosslist(validation_loss=validation_loss.item())

                validation_loss_interval = cum_validation_loss/eval_iter

                if validation_loss_interval < datasets.best_validation_loss[1]:
                    datasets.setlosslist(best_validation_loss=validation_loss_interval)
                    datasets.setcheckpoint(iter=iter_num, curtime=time.perf_counter())     
            model.train()
            print(f"\n{iter_num=} {training_loss.item()=:.4f} {scheduler.get_last_lr()=} {validation_loss_interval=:.4f}")
        datasets.setlrlist(scheduler.get_last_lr())
            
    datasets.setmodel(modelargs, model, optimizer, scheduler)

    datasets.setcheckpoint(iter_num, time.perf_counter())
    torch.save(datasets.checkpoint, f'{datasets.modelpath}/{datasets.modelname}')
    datasets.reset()

torch.cuda.empty_cache()
datasets.gettime(time.perf_counter())