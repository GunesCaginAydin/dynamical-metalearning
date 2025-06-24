from pathlib import Path
import sys
sys.path.insert(0, "/home/gunes/isaacgym/tfm/RoboMorph/data_generation") # should be changed ---
import genutil
import time
import torch
import numpy as np
import os   
from datasets import *
from utils import *
import tqdm

torch.set_float32_matmul_precision("high")
torch.use_deterministic_algorithms(False)
torch.cuda.empty_cache()

args = arguments(description="FrankaSysId",params=[]).parse_arguments()

savefig=True

pre = preprocess(args=args, modelfolder='trial') # initialize preprocessing object for inference
pre.resolve_datasets(eval=True, modelfolder='trial') # condition datasets for inference

post = postprocess(args=args,data=pre.getcurrrentprocess()) # initialize postprocessing object for analysis

for i, nmodel in enumerate(pre.modellist):
    initparams = {}
    pre.initialize_model(modelname=nmodel, **initparams) # obtain model as "nmodel"
    pre.settime(time.perf_counter()) # start timer
    modelargs, model, optimizer, scheduler = pre.getmodel() # model data
    pre.reset(outdim=modelargs['n_y'], **initparams) # condition inference as "initparams" -> context, timestep, fewshot changes
    params = sum(p.numel() for p in model.parameters())
    datasize = int(pre.datadict['tenvs'])
    post.configure_models(nmodel) # condition model
    
    for data in pre.testlist:
        pre.load(data=data,
                 itr=args.total_sim_iterations,
                 eval=True)
        pre.configure_datasets(eval=True) # condition datasets

        training_dataset, validation_dataset, test_dataset = pre.getdataset() # testing dataset

        score = pre.check_distribution(train=pre.trainlist, test=data) # testing score

        model.eval()
        for iter, single in tqdm.tqdm(enumerate(test_dataset, start=0)):
            usingle, ysingle = single[0].cuda(non_blocking=True), single[1].cuda(non_blocking=True)
            if modelargs['n_y']==16:
                ysingle = torch.cat((ysingle[:,:,:3],genutil.decide_orientation(ysingle[:,:,3:7],dim='6D'),ysingle[:,:,7:]),dim=2)
            if args.controller: # if controller implemented import additional variables
                tsingle, kpsingle, kvsingle, kisingle = single[2].cuda(non_blocking=True), single[3].cuda(non_blocking=True), single[4].cuda(non_blocking=True), single[5].cuda(non_blocking=True)
                ytarget, tauxv = pre.normalize(x=tsingle)
                targetctx, targetnew = pre.seperate_context(ytarget)
                kpsingle, kpauxv = pre.normalize(x=kpsingle,ndim=1)
                kvsingle, kvauxv = pre.normalize(x=kvsingle,ndim=1)
                kisingle, kiauxv = pre.normalize(x=kisingle,ndim=1)
                controller_gains = torch.cat((kpsingle,kvsingle,kisingle),dim=1)
                controller_gains, kaux = pre.normalize(x=controller_gains, ndim=1)
            usingleraw = copy.deepcopy(usingle)
            ysingleraw = copy.deepcopy(ysingle)
            
            usingle, uaux = pre.normalize(x=usingle, ndim=1)
            ysingle, yaux = pre.normalize(x=ysingle, ndim=1)

            uctx,unew = pre.seperate_context(usingle, ctx=modelargs['seq_len_ctx'])
            yctx,ynew = pre.seperate_context(ysingle, ctx=modelargs['seq_len_ctx'])
            
            ytrue = ynew[:,:modelargs['seq_len_new'],:]
            
            with torch.inference_mode():
                if args.subcommand=='transformer' or str.split(nmodel,'_')[0]=='transformer': # embedding u + y_ctx -> y_new 
                    
                    if iter==1000:
                        break

                    if not('trftype' in modelargs.keys()) or modelargs['trftype']==1: # u | y sim 
                         ysim = model(yctx, uctx, unew)[:,:modelargs['seq_len_new'],:]
                    elif modelargs['trftype']==2: # track | y sim
                        ysim = model(yctx, targetctx, targetnew)[:,:modelargs['seq_len_new'],:]
        
                    ysim = pre.denormalize(nx=ysim, aux=yaux)

                elif args.subcommand=='diffuser' or str.split(nmodel,'_')[0]=='diffuser': # inpainting -> y | 1000 no_cond

                    if iter==100:
                        break
                    
                    if not('diftype' in modelargs.keys()) or modelargs['diftype']==1:
                        if not args.warmstart: # default inpainting
                            xs, _, _ = model.conditional_sample_from_noisy_distiribution(cond=yctx, force=usingle, horizon=args.total_sim_iterations,
                                                                                            verbose=False)
                    elif modelargs['diftype']==2:
                        if not args.warmstart: # default inpainting
                            xs, _, _ = model.conditional_sample_from_noisy_distiribution(cond=yctx, force=ytarget, horizon=args.total_sim_iterations,
                                                                                            verbose=False)                        
                    
                    if args.warmstart: # inpainting warmstart inference startup
                        tstart = torch.int(10, (1,), device=args.graphics_device_id).long()
                        xsample = torch.cat((usingle,ysingle),dim=0)
                        xnoised = model.get_noisy_distribution(xstart=xsample, t=tstart)
                        xs, _, _ = model.conditional_sample_from_noisy_distiribution(cond=yctx, force=usingle, horizon=args.total_sim_iterations,
                                                                                    verbose=False, xwarmstart=xnoised, twarmstart=tstart)

                    if args.reward_function_training: # reward function training for goal oriented reinforcement learning - NOT IMPLEMENTED
                        xs, _, _ = model.conditional_sample_from_noisy_distiribution(cond=yctx, force=usingle, horizon=args.total_sim_iterations,
                                                                    verbose=False, xwarmstart=xnoised, twarmstart=tstart)        

                    us = xs[:,:,:args.in_dimension]
                    ys = xs[:,modelargs['seq_len_ctx']:,args.in_dimension:]
                    usim = pre.denormalize(nx=us, aux=uaux)
                    ysim = pre.denormalize(nx=ys, aux=yaux)
                
                elif args.subcommand=='rechorUnet' or str.split(nmodel,'_')[0]=='rechorUnet' or args.subcommand=='rechorTrf' or str.split(nmodel,'_')[0]=='rechorTrf':

                    if iter==10:
                        break

                    if modelargs['rechortype']==1: # uy | 200 condition -> y | 1000, local_cond or global_cond
                        obs = torch.cat((
                            usingle, ysingle),
                            dim=2
                        )
                        action = ysingle
                    elif modelargs['rechortype']==2: # u | 1000 condition -> y | 800, local_cond or global_cond
                        obs = usingle
                        action = ysingle
                    elif modelargs['rechortype']==3: # u | 1000 , y | 200 condition -> y | 800, local_cond or global_cond
                        obs = ysingle
                        action = ysingle
                    elif modelargs['rechortype']==4: # inpainting -> y | 1000, no_cond
                        obs = usingle 
                        action = ysingle
                    elif modelargs['rechortype']==5: # target conditioning -> y | 1000
                        obs = ytarget
                        action = ysingle
                    elif modelargs['rechortype']==6: # controller Kc conditioning -> y | 1000
                        B,G = controller_gains.shape
                        obs = controller_gains if modelargs['controlgain_horizon']==1 else controller_gains.unsqueeze(dim=1).repeat(1,modelargs['controlgain_horizon'],1)
                        action = ysingle

                    batch = {
                        'obs':obs,
                        'action':action
                    }

                    preddict = model.predict_action(batch)
                    ysim = preddict['action_pred'][:,modelargs['seq_len_ctx']:,:]
                    ysim = pre.denormalize(nx=ysim, aux=yaux)
                    
            yctx, ynew = pre.seperate_context(ysingleraw, ctx=modelargs['seq_len_ctx'])
            ytrue = ynew[:,:modelargs['seq_len_new'],:] # EXTRACT UNTIL ITER
            yerr = ytrue - ysim

            pre.cast2original(yctx=yctx,
                            ytrue=ytrue,
                            ysim=ysim,
                            yerr=yerr)
        model.train()

        pre.detachtensors()
        _rmse = pre.test(method='rmse')
        _nrmse = pre.test(method='nrmse')
        _fit = pre.test(method='fitidx')    
        _r2 = pre.test(method='r2')
        _aic = pre.test(method='aic', modelsize=params, datasize=datasize)
        _fpe = pre.test(method='fpe', modelsize=params, datasize=datasize)

        tm = pre.cast2dict(rmse=_rmse,
                        nrmse=_nrmse, 
                        fit=_fit, 
                        r2=_r2,
                        aic=_aic,
                        fpe=_fpe,
                        modelname=nmodel,
                        testname=data)
    
        post.configure_tests(pre.ytrue, pre.ysim, pre.err, pre.yctx, tm, score, data, nmodel)
        
        

        post.plotsim2sim(of='best',
                        dim=args.out_dimension-10,
                        title='sim2sim Prediction of the Best Environment',
                        save=savefig)
        post.plothorizon(of='best',
                        iter=modelargs['seq_len_ctx'] + modelargs['seq_len_new'],
                        ctx=modelargs['seq_len_ctx'],
                        dim=args.out_dimension-10,
                        title='Horizon Prediction of the Best Environment',
                        save=savefig)
        post.plotpredictionerror(of='best',
                        dim=args.out_dimension-10,
                        title='Absolute Error of the Best Environment',
                        save=savefig)
        
        post.plotsim2sim(of='median',
                        dim=args.out_dimension-10,
                        title='sim2sim Prediction of the Median Environment',
                        save=savefig)
        post.plothorizon(of='median',
                        iter=modelargs['seq_len_ctx'] + modelargs['seq_len_new'],
                        ctx=modelargs['seq_len_ctx'],
                        dim=args.out_dimension-10,
                        title='Horizon Prediction of the Median Environment',
                        save=savefig)
        post.plotpredictionerror(of='median',
                        dim=args.out_dimension-10,
                        title='Absolute Error of the Median Environment',
                        save=savefig)

        post.plotsim2sim(of='worst',
                        dim=args.out_dimension-10,
                        title='sim2sim Prediction of the Worst Environment',
                        save=savefig)
        post.plothorizon(of='worst',
                        iter=modelargs['seq_len_ctx'] + modelargs['seq_len_new'],
                        ctx=modelargs['seq_len_ctx'],
                        dim=args.out_dimension-10,
                        title='Horizon Prediction of the Worst Environment',
                        save=savefig)
        post.plotpredictionerror(of='worst',
                        dim=args.out_dimension-10,
                        title='Absolute Error of the Worst Environment',
                        save=savefig)
        
        post.plotmetrics_overjoints(save=savefig)
        post.plotmetrics_overenvironments(save=savefig)
        post.plotmetrics_overtime(save=savefig)
        
        post.plotvariation_overjoints(save=savefig)
        post.plotvariation_overenvironments(save=savefig)

        post.plotmetric(metric='aic',
                        envmean=False,
                        jointmean=True,
                        save=savefig)
        
        post.plotmetric(metric='fpe',
                        envmean=False,
                        jointmean=True,
                        save=savefig)
        
        post.plotmetric(metric='aic',
                        envmean=True,
                        jointmean=False,
                        save=savefig)
        
        post.plotmetric(metric='fpe',
                        envmean=True,
                        jointmean=False,
                        save=savefig)
                
        post.plotmetric(metric=None,
                        envmean=True,
                        jointmean=True,
                        save=savefig)
        
        post.logdata(specificto='test')

        pre.reset()
        
    post.plotvariation_tests(labels=['rng_VS','rng_FS','rng_FC'],
                             include_legend=False,
                             save=savefig,
                             metric='r2')
    post.logdata(specificto='model')

torch.cuda.empty_cache()

post.plotvariation_models(labels=['rng_trf','rng_dif','rg_trf','rg_dif'],
                          legendentries=['rng_VS','rng_FS','rng_FC'],
                          save=savefig,
                          metric='r2',
                          omitscores=True,
                          limitaxes=True)
post.logdata(specificto='all')

post.plotlosses(labels=['rng_trf','rng_dif','rg_trf','rg_dif'],
                save=savefig)
