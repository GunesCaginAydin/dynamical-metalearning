"""
TO IMPLEMENT: 

optimization of algorithms
checks

utils - process ???
"""
from pathlib import Path
import time
import torch
import numpy as np
from transformer_sim import Config, TSTransformer
import os   
from datasets import dataset
from utils import *
import sys
import tqdm

torch.set_float32_matmul_precision("medium")
torch.use_deterministic_algorithms(False)

sys.argv = ['','--data-name=MG1','transformer','-ctx=20']
args = arguments(description="FrankaSysId",params=[]).parse_arguments()

pre = preprocess(args=args)

torch.cuda.empty_cache()

pre.resolve_datasets(eval=True)
modelname = pre.modellist[0]
data = pre.testdatalist[0]
pre.initialize_model(modelname=modelname)
pre.settime(time.perf_counter())

pre.load(data=data,
         eval=True)
pre.configure_dataset(eval=True)  

training_dataset, validation_dataset, test_dataset = pre.getdataset()
modelargs, model, optimizer, scheduler = pre.getmodel()
score = pre.check_distribution(pre.traindatalist,data)

for model in pre.modellist:

    pre.initialize_model(modelname=model,
                            eval=True)
    pre.configure_dataset(eval=True)
    post.plotlosses()

    for data in pre.testdatalist: 

        pre.load(data)
        training_dataset, validation_dataset, validation_loss_list, training_loss_list, best_validation_loss = pre.getdataset()
        modelargs, model, optimizer = pre.getmodel()
        score = pre.check_distribution()
        
        model.eval()

        with torch.inference_mode():
            for iter, (ysingle, usingle) in enumerate(training_dataset):

                usingle, umean, ustd = pre.normalize(usingle)
                ysingle, ymean, ystd = pre.normalize(ysingle)
                
                uctx,unew = pre.seperate_context(usingle)
                yctx,ynew = pre.seperate_context(ysingle)

                ytrue = ynew
                ysim = model(yctx, uctx, unew)
                yerr = ytrue - ysim
                pre.cast2original(ytrue=ytrue,
                                  ysim=ysim,
                                  yerr=yerr)
        
        model.train()

        _rmse = pre.test(method='rmse')
        _nrmse = pre.test(method='nrmse')
        _fit = pre.test(method='fit')    
        _r2 = pre.test(method='r2')

        tm = pre.cast2dict(rmse=_rmse,
                      nrmse=_nrmse, 
                      fit=_fit, 
                      r2=_r2,
                      modelname=model,
                      testname=data)
        
        #post.plotsim2sim(tm)
        #post.plotsim2real(tm)

        post.plotmetrics_test(tm)

        post.plothorizon(tm)

        post.plotvariation_test(tm)
        
    post.plotmetrics_model()
post.plotmetrics_total()

torch.cuda.empty_cache()

dt = pre.gettime(time.perf_counter())

post.tabulate()
post.logdata()
post.summarize()


