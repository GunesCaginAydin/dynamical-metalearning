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
pre = preprocess(args=args, modelfolder='trial') # initialize preprocessing object for inference
pre.resolve_datasets(eval=True, modelfolder='trial') # condition datasets for inference
post = postprocess(args=args,data=pre.getcurrrentprocess()) # initialize postprocessing object for analysis
print(pre.modellist)
print(pre.testlist)
savefig=True

ctxs = [320]
#ctxs = [80,160,320,640,1280,2560]
seqs = [80]
#seqs = [20,40,80,160,320,640]
#iters = 400*np.ones_like(ctxs)
iters = [400]
tws = 5
for (ctx,seq,iter) in zip(ctxs,seqs,iters):
    args.total_sim_iterations=iter

    for i, nmodel in enumerate(pre.modellist):
        #initparams = {'ctxlen':ctx,
        #            'seqlen':seq}
        initparams = {}
        pre.initialize_model(modelname=nmodel, **initparams)
        pre.settime(time.perf_counter())
        modelargs, model, optimizer, scheduler = pre.getmodel()
        pre.reset(outdim=modelargs['n_y'], **initparams)
        params = sum(p.numel() for p in model.parameters())
        datasize = int(pre.datadict['tenvs'])
        post.configure_models(nmodel)

        for data in pre.testlist:
            pre.load(data=data,
                    itr=args.total_sim_iterations,
                    eval=True,
                    real=False)
            pre.configure_datasets(eval=True)

            training_dataset, validation_dataset, test_dataset = pre.getdataset()

            #score = pre.check_distribution(train=pre.trainlist, test=data)
            score = 0
            model.eval()
            for iter, single in tqdm.tqdm(enumerate(test_dataset, start=0)):
                usingle, ysingle = single[0].cuda(non_blocking=True), single[1].cuda(non_blocking=True)
                ysingle = ysingle[:,:,7:]
                if modelargs['n_y']==16:
                    ysingle = torch.cat((ysingle[:,:,:3],genutil.decide_orientation(ysingle[:,:,3:7],dim='6D'),ysingle[:,:,7:]),dim=2)
                if args.controller:
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

                usinglenormraw = copy.deepcopy(usingle)
                ysinglenormraw = copy.deepcopy(ysingle)

                #usingle = add_noise(usingle,25)
                #ysingle = add_noise(ysingle,25)

                uctx,unew = pre.seperate_context(usingle, ctx=modelargs['seq_len_ctx'])
                yctx,ynew = pre.seperate_context(ysingle, ctx=modelargs['seq_len_ctx'])

                ytrue = ynew[:,:modelargs['seq_len_new'],:]

                ct = time.perf_counter()
                with torch.inference_mode():
                    if args.subcommand=='transformer' or str.split(nmodel,'_')[0]=='transformer': # embedding u + y_ctx -> y_new

                        if iter==100:
                            break

                        if not('trftype' in modelargs.keys()) or modelargs['trftype']==1: # u | y sim
                            ysim = model(yctx, uctx, unew)[:,:modelargs['seq_len_new'],:]
                        elif modelargs['trftype']==2: # track | y sim
                            ysim = model(yctx, targetctx, targetnew)[:,:modelargs['seq_len_new'],:]

                        ysim = pre.denormalize(nx=ysim, aux=yaux)

                    elif args.subcommand=='diffuser' or str.split(nmodel,'_')[0]=='diffuser': # inpainting -> y | 1000 no_cond

                        if iter==100:
                            break

                        if (not('diftype' in modelargs.keys()) or modelargs['diftype']==1) and not args.warmstart:
                            xs, _, snapshot = model.conditional_sample_from_noisy_distiribution(cond=yctx, force=usingle, horizon=args.total_sim_iterations,
                                                                                                verbose=False, return_chain=False)
                        elif (modelargs['diftype']==2) and not args.warmstart:
                            xs, _, snapshot = model.conditional_sample_from_noisy_distiribution(cond=yctx, force=ytarget, horizon=args.total_sim_iterations,
                                                                                                verbose=False, return_chain=False)

                        if args.warmstart: # inpainting warmstart inference startup
                            if iter==0:
                                if not('diftype' in modelargs.keys()) or modelargs['diftype']==1:
                                    xs, _, snapshot = model.conditional_sample_from_noisy_distiribution(cond=yctx, force=usingle, horizon=args.total_sim_iterations,
                                                                                                verbose=False, return_chain=False)
                                elif modelargs['diftype']==2:
                                    xs, _, snapshot = model.conditional_sample_from_noisy_distiribution(cond=yctx, force=ytarget, horizon=args.total_sim_iterations,
                                                                                                verbose=False, return_chain=False)
                            else:
                                tstart = torch.tensor((tws,),device='cuda',dtype=torch.long)
                                xnoised = model.get_noisy_distribution(xstart=xs, t=tstart)[0] # add inference conditions???
                                xs, _, snapshot = model.conditional_sample_from_noisy_distiribution(cond=yctx, force=usingle, horizon=args.total_sim_iterations,
                                                                                            verbose=False, xwarmstart=xnoised, twarmstart=tws,return_chain=False)

                        if args.reward_function_training: # reward function training for goal oriented reinforcement learning - NOT IMPLEMENTED
                            xs, _, snapshot = model.conditional_sample_from_noisy_distiribution(cond=yctx, force=usingle, horizon=args.total_sim_iterations,
                                                                        verbose=False, xwarmstart=xnoised, twarmstart=tws,return_chain=False)

                        us = xs[:,:,:args.in_dimension]
                        ys = xs[:,modelargs['seq_len_ctx']:,args.in_dimension:]
                        usim = pre.denormalize(nx=us, aux=uaux)
                        ysim = pre.denormalize(nx=ys, aux=yaux)
                        if take_snapshots:
                            snapshotsinp.append(snapshot.squeeze().detach().to('cpu').numpy())

                    elif args.subcommand=='rechorUnet' or str.split(nmodel,'_')[0]=='rechorUnet' or args.subcommand=='rechorTrf' or str.split(nmodel,'_')[0]=='rechorTrf':

                        if iter==100:
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

                        if not args.warmstart:
                            preddict,snapshot = model.predict_action(batch,return_chain=False)
                        elif args.warmstart:
                            if iter==0:
                                preddict,snapshot = model.predict_action(batch,return_chain=False)
                            else:
                                model.num_inference_steps = tws
                                pasttraj = preddict['action_pred']
                                preddict,snapshot = model.predict_action(batch,trajectory=pasttraj,return_chain=False)

                        ysim = preddict['action_pred'][:,modelargs['seq_len_ctx']:,:]
                        ysim = pre.denormalize(nx=ysim, aux=yaux)

                        if args.subcommand=='rechorUnet' or str.split(nmodel,'_')[0]=='rechorUnet' and take_snapshots:
                            snapshotscnn.append(snapshot.squeeze().detach().to('cpu').numpy())
                        elif args.subcommand=='rechorTrf' or str.split(nmodel,'_')[0]=='rechorTrf' and take_snapshots:
                            snapshotstrf.append(snapshot.squeeze().detach().to('cpu').numpy())

                et = time.perf_counter()
                dt = et - ct
                pre.addtime(dt)

                model.train()
                ysinglerawclone = torch.clone(ysingleraw)
                ysimclone = torch.clone(ysim)
                for k in range(yctx.shape[2]):
                    #if k<3:
                    #    ysinglerawclone[:,:,k] = ysinglerawclone[:,:,k]*1000
                    #    ysimclone[:,:,k] = ysimclone[:,:,k]*1000
                    #elif k<7:
                    #    continue
                    #else:
                    ysinglerawclone[:,:,k] = ysinglerawclone[:,:,k]*180/torch.pi
                    ysimclone[:,:,k] = ysimclone[:,:,k]*180/torch.pi

                yctx, ynew = pre.seperate_context(ysinglerawclone, ctx=modelargs['seq_len_ctx'])
                ytrue = ynew[:,:modelargs['seq_len_new'],:] # EXTRACT UNTIL ITER
                yerr = ytrue - ysimclone

                pre.cast2original(yctx=yctx,
                                ytrue=ytrue,
                                ysim=ysimclone,
                                yerr=yerr)

                if iter==1000:
                    break

            pre.detachtensors()
            _rmse = pre.test(method='rmse') # time start = ctxs[-1]
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

            post.configure_tests(pre.ytrue, pre.ysim, pre.err, pre.yctx, tm, score, data, nmodel, pre.dtl)

            pre.reset()
        post.logdata(specificto='model')
    post.savedata()