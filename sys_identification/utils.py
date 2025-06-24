import matplotlib.gridspec
import torch
import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from tabulate import tabulate, SEPARATING_LINE
from torch.utils.tensorboard import SummaryWriter
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy import interpolate
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from pathlib import Path
import datetime
import argparse
import os

try:
    from metrics import *
    from datasets import dataset
except:
    from Sys_Identification.metrics import *
    from Sys_Identification.datasets import dataset

class arguments():
    """
    Custom parser, a merged script of older training programs, model hyper-parameters,
    training parameters and testing parameters are obtained through the parser - refer
    to README.md.

    Parameters
    ---
        description (string): id of parser - DEPRECATED
        params (list): list of extra parameters - DEPRECATED
    """
    def __init__(self,
                 description="FrankaSysId",
                 params=[]):
        self.description = description
        self.params = params
        self.parser = argparse.ArgumentParser(description=description)
        self.subparser = self.parser.add_subparsers(help="model type", dest="subcommand")

    def __str__(self):
        return f'Parser Object instantiated'

    def parse_arguments(self):
        """
        Some arguments COULD BE DEPRECATED
        """
        self.parser.add_argument('-init','--init-type', type=str, default='scratch', choices=['scratch',
                                                                                              'resume',
                                                                                              'finetune'],
                            help='init from (scratch|resume|pretrained|test)')
        self.parser.add_argument('-dn','--data-name', type=str, default='MG1',
                            help='name of the training/finetuning dataset')
        self.parser.add_argument('-tn','--test-name', type=str, default='T1',
                            help='name of the testing dataset')
        self.parser.add_argument('-mn','--save-name', type=str, default=None,
                            help='name to save the created model by')
        self.parser.add_argument('-ns','--process-name',type=str,default=None,
                            help='name to postprocess the created model by')
        self.parser.add_argument('-dc','--disable_cuda', action='store_true',
                            help="tensor process device")   
        self.parser.add_argument('-in','--in-dimension', type=int, default=7,
                            help="umber of joints of interest of the dataset")   
        self.parser.add_argument('-out','--out-dimension', type=int, default=14,
                            help="number of coordinates of interest of the dataset")
        self.parser.add_argument('-tsi','--total-sim-iterations', type=int, default=1000,
                            help="number of total simulation iterations of the dataset")                     
        
        self.parser.add_argument('-trb','--training-batch-size',type=int,default=8,
                            help='batch size for training data')
        self.parser.add_argument('-vlb','--validation-batch-size',type=int,default=8,
                            help='batch size for validation data')
        self.parser.add_argument("-lf",'--loss-function', type=str, default='MSE', choices=["MAE",
                                                                                         "MSE",
                                                                                         "Huber",
                                                                                         "RMSE",
                                                                                         "logcosh"],
                            help="loss function: 'MAE'-'MSE'-'Huber'-'RMSE'-'logcosh'")
        
        self.parser.add_argument("-evitr",'--validate-at', type=int, default=100,
                            help='evaluation iteraration')
        self.parser.add_argument("-il",'--log-at', type=int, default=1000,
                            help='log to wandb at iteration num')

        self.parser.add_argument("-lr",'--learning-rate', type=float, default=6e-4,
                            help='learning rate magnitude')
        self.parser.add_argument("-cons",'--constant-decay', action='store_true',
                            help='constant learning rate')        
        self.parser.add_argument("-exp",'--exponential-decay', action='store_true',
                            help='exponential decay learning rate') 
        self.parser.add_argument("-stp",'--step-decay', action='store_true',
                            help='step decay learning rate') 
        self.parser.add_argument("-cos",'--cosine-annealing', action='store_true',
                            help='cosine annealed learning rate')
        self.parser.add_argument("-cwu",'--cosine-annealing-with-warmup', action='store_true',
                            help='cosine annealed learning rate with warmup iterations')
        self.parser.add_argument("-std",'--std-normalizer', action='store_true',
                            help='Gaussian normalizer')
        self.parser.add_argument("-lin",'--lin-normalizer', action='store_true',
                            help='linear (limits) normalizer')
        self.parser.add_argument("-cdf",'--cdf-normalizer', action='store_true',
                            help='probability density function normalizer')                 
        self.parser.add_argument("-wd",'--weight-decay', type=float, default=0.0,
                            help='weight decay')
        self.parser.add_argument("-b1",'--beta1', type=float, default=.9,
                            help="loss function ext 1: 'MAE'-'MSE'-'Huber'")
        self.parser.add_argument("-b2",'--beta2', type=float, default=.95,
                            help="loss function ext 2: 'MAE'-'MSE'-'Huber'")

        self.parser.add_argument('-s','--seed', default=False,
                            help='seed for random number generation')

        self.parser.add_argument('-ctx', '--context', type=int, default=20,
                            help='percent of provided context')
        self.parser.add_argument('-ema','--include-ema', default=False, action='store_true',
                            help='use ema for evaluating models')
        self.parser.add_argument('-ctrl','--controller', action='store_true',
                            help='implement controller training/inference for gain mapping')
        self.parser.add_argument('-cgh','--controlgain_horizon', type=int,
                            help='implement controller horizon for gain inference')
        
        self.parser.add_argument('-ws', '--warmstart', action='store_true', default=False,
                             help='if true warmstart diffusion sampling process to speed up inference') 
        self.parser.add_argument('-rft', '--reward_function_training', action='store_true', default=False,
                             help='if true sample from the diffusion model through classifier guidance') 

        parsert = self.subparser.add_parser('transformer', help='transformer doc')
        parsert.add_argument('-ctx', '--context', type=int, default=20,
                             help='percent of provided context')
        parsert.add_argument('-nl','--n-layer', type=int, default=12,
                            help='number of transformer layers')
        parsert.add_argument('-nh','--n-head', type=int, default=8,
                            help='number of transformer heads')
        parsert.add_argument('-ne','--n-embd', type=int, default=192,
                            help='embedding dimension')
        parsert.add_argument('-d','--dropout', type=float, default=0.0,
                            help='dropout magnitude')
        parsert.add_argument('-nb','--bias', action='store_false', default=True,
                            help='disable nn biases')
        parsert.add_argument('-ttrf','--trftype', type=int, default=1,
                            help='type of simulation transformer pipeline')
        

        parserd = self.subparser.add_parser("diffuser", help="diffusion doc")
        parserd.add_argument('-ctx', '--context', type=int, default=20,
                             help='percent of provided context')
        parserd.add_argument('-ts', '--timesteps', type=int, default=1000,
                             help='number of denoising timesteps')
        parserd.add_argument('-nh', '--hidden', type=int, default=64,
                             help='number of hidden layers')
        parserd.add_argument('-nc', '--multiplier', type=int, default=3,
                             help='number of Unet dimension multipliers, pooling num')
        parserd.add_argument('-att', '--attention', action='store_true',
                             help='if true add attention block to Unet') 
        parserd.add_argument('-ucond', '--input_condition', action='store_true', default=False,
                             help='if true predict noise in denoising process')   
        parserd.add_argument('-ycond', '--output_condition', action='store_true', default=False,
                             help='if true predict noise in denoising process')   
        parserd.add_argument('-eps', '--predict_epsilon', action='store_true', default=False,
                             help='if true predict noise in denoising process')
        parserd.add_argument('-lw', '--loss-weight', type=float, default=3.0,
                             help='loss weighting of ycond in inpainting')
        parserd.add_argument('-ws', '--warmstart', action='store_true', default=False,
                             help='if true warmstart diffusion sampling process to speed up inference') 
        parserd.add_argument('-rft', '--reward_function_training', action='store_true', default=False,
                             help='if true sample from the diffusion model through classifier guidance') 
        parserd.add_argument('-tdif','--diftype', type=int, default=1,
                            help='type of inpainting diffusion pipeline')
    

        parserrhd = self.subparser.add_parser('rechorUnet', help='receding horizon Unet doc')
        parserrhd.add_argument('-ctx', '--context', type=int, default=20,
                             help='percent of provided context')
        parserrhd.add_argument('-ts', '--timesteps', type=int, default=1000,
                             help='number of denoising timesteps')
        parserrhd.add_argument('-pt', '--prediction_type', type=str, default='epsilon',
                             help='if true predict noise in denoising process')
        parserrhd.add_argument('-nh', '--hidden', type=int, default=256,
                             help='number of hidden layers')
        parserrhd.add_argument('-nc', '--multiplier', type=int, default=3,
                             help='number of Unet dimension multipliers, pooling num')
        parserrhd.add_argument('-gc','--global-obscond', action='store_true',
                            help='global observation conditioning')
        parserrhd.add_argument('-lc','--local-obscond', action='store_true',
                            help='local observation conditioning')
        parserrhd.add_argument('-trch','--rechortype', type=int, default=2,
                            help='type of receding horizon pipeline')
        

        parserrht = self.subparser.add_parser('rechorTrf', help='receding horizon Transformer doc')
        parserrht.add_argument('-ctx', '--context', type=int, default=20,
                             help='percent of provided context')
        parserrht.add_argument('-ts', '--timesteps', type=int, default=100,
                             help='number of denoising timesteps')
        parserrht.add_argument('-pt', '--prediction_type', type=str, default='epsilon',
                             help='if true predict noise in denoising process')
        parserrht.add_argument('-nl','--layers', type=int, default=12,
                            help='number of transformer layers')
        parserrht.add_argument('-nh','--heads', type=int, default=8,
                            help='number of transformer heads')
        parserrht.add_argument('-ne','--embeddings', type=int, default=192,
                            help='embedding dimension')
        parserrht.add_argument('-csl','--causality', action='store_true',
                            help='causal attention masking')
        parserrht.add_argument('-tc','--timecond', action='store_true',
                            help='time conditioning')
        parserrht.add_argument('-oc','--obscond', action='store_true',
                            help='observation conditioning')
        parserrht.add_argument('-trch','--rechortype', type=int, default=2,
                            help='type of receding horizon pipeline')
        

        parserm = self.subparser.add_parser("meta", help="metalearner doc")

        
        args = self.parser.parse_args()
        return args

class preprocess(dataset):
    """
    Preprocessing class used to manipulate created models and datasets. Current use is only
    in testing. Preprocess module applies the available metrics and casts the acquired values
    into formats readily interpretable by the postprocess module. Currently available
    metrics are:

        rmse: root mean squared error | dataset 
        nrmse: normalized root mean squared error | dataset
        r2: coefficient of determination | dataset
        fitidx: fit index | dataset 
        aic: akaike information criterion | dataset & model
        fpe: final prediction error | dataset & model

    Using the above-metrics, it is possible to accumulate simulation and benchmark values
    of all instances of a model evaluated on a specific test, as well as accuracy and variation
    data of a model evaluated of a model evaluated on a specific test.

    Processing is from the folders:
        /logs
        /plots
        /models

    Parameters
    ---
        args (dict): arguments passed during testing
    """
    def __init__(self, 
                args,
                logpath='sys_identification/logs',
                figpath='sys_identification/plots',
                sumpath='sys_identification/summary',
                modelloc='sys_identification/models',
                modelfolder=False):
        super().__init__(args)

        self.yctx = torch.empty(size=(0,self.sqlctx,self.args.out_dimension),
                            device=self.device)
        self.ytrue = torch.empty(size=(0,self.sqlpar,self.args.out_dimension),
                            device=self.device)
        self.ysim = torch.empty(size=(0,self.sqlpar,self.args.out_dimension),
                            device=self.device)
        self.err = torch.empty(size=(0,self.sqlpar,self.args.out_dimension),
                            device=self.device)

        self.metricdict = {}
        self.currentdict = {}
        self.accusation = {}
        self.dataname = args.data_name

        self.modelname = ''
        self.testname = ''

        self.modellist = []
        self.testlist = []
        self.trainlist = []

        parentpath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.logpath = os.path.join(parentpath,logpath)
        self.figpath = os.path.join(parentpath,figpath)
        self.sumpath = os.path.join(parentpath,sumpath)
        self.modelloc= os.path.join(parentpath,modelloc)
        self.modelfolder = args.data_name if not modelfolder else modelfolder
        self.modelloc = os.path.join(self.modelloc,self.modelfolder)

    def __str__(self):
        print("Preprocess Object Instantiated")

    
    def settime(self, 
                time_):
        return super().settime(time_)
    
    def gettime(self):
        return super().gettime()
    

    def getmetadata(self):
        return super().getmetadata()
    
    def setmetadata(self, 
                    datadict_):
        return super().setmetadata(datadict_)
    

    def getdataset(self):
        return super().getdataset()
    
    def setdataset(self, 
                   training_dataset_=None, 
                   validation_dataset_=None, 
                   test_dataset_=None):
        return super().setdataset(training_dataset_, validation_dataset_, test_dataset_)
    

    def getmodel(self):
        return super().getmodel()
    
    def setmodel(self, 
                 modelargs_, 
                 model_, 
                 optimizer_,
                 scheduler_):
        return super().setmodel(modelargs_, 
                                model_, 
                                optimizer_,
                                scheduler_
                                )
    

    def getmetrics(self):
        return self.metricdict

    def setmetrics(self, test_):
        self.metricdict = test_


    def getsim2sim(self):
        return self.accusation

    def setsim2sim(self,
                    control_,
                    position_,
                    diff_=None,
                    mass_=None):
        self.accusation = {
            "control" : control_,
            "position" : position_,
            "diff" : diff_,
            "mass" : mass_
        }


    def getsim2real(self):
        return self.accusation
    
    def setsim2real(self,
                    control_,
                    position_,
                    diff_=None,
                    mass_=None,
                    params_=None):
        self.accusation = {
            "control" : control_,
            "position" : position_,
            "diff" : diff_,
            "mass" : mass_,
            "params" : params_
        }

    
    def getcurrrentprocess(self):
        return {
            "logpath" : self.logpath,
            "figpath" : self.figpath,
            "sumpath" : self.sumpath,
            "modelloc" : self.modelloc,
            "metricdict" : self.metricdict,
            "dataname" : self.dataname,
            "modellist" : self.modellist,
            "testlist" : self.testlist
        }

    def setcurrentprocess(self, 
                          logpath_,
                          figpath_,
                          sumpath_,
                          metricdict_,
                          dataname_,
                          modellist_,
                          testlist_):
        self.logpath = logpath_
        self.figpath = figpath_
        self.sumpath = sumpath_
        self.metricdict = metricdict_
        self.dataname = dataname_
        self.modellist = modellist_
        self.testlist = testlist_


    def reset(self, horizon=None, seqlen=None, ctxlen=None, outdim=None):
        """
        Resets the state of input/output pair and corresponding accuracy evaluation buffers,
        should be called at the end of every successful test on a model. Resetting with 
        arguments allow inferencetime changes of model parameters.

        Parameters
        ---
        horizon (int) : horizon length 
        seqlen (int) : non-context length
        ctxlen (int) : context length
        outdim (int) : output trajectory dimension
        """
        if outdim is not None:
            self.args.out_dimension = outdim
        
        if ctxlen is not None and seqlen is not None:
            self.sqlctx = ctxlen
            self.sqlpar = seqlen

        self.yctx = torch.empty(size=(0,self.sqlctx,self.args.out_dimension),
                            device=self.device)
        self.ytrue = torch.empty(size=(0,self.sqlpar,self.args.out_dimension),
                            device=self.device)
        self.ysim = torch.empty(size=(0,self.sqlpar,self.args.out_dimension),
                            device=self.device)
        self.err = torch.empty(size=(0,self.sqlpar,self.args.out_dimension),
                            device=self.device)

        self.metricdict = {}
        self.currentdict = {}
        self.accusation = {}
        

    def resolve_datasets(self, **kwargs):
        return super().resolve_datasets(**kwargs)
    
    def initialize_model(self, **kwargs):
        return super().initialize_model(**kwargs)
    
    def configure_datasets(self, **kwargs):
        return super().configure_datasets(**kwargs)
    
    def check_distribution(self, train, test):
        """
        Checks the distribution of a training dataset that a given model is trained on with
        the distribution of a test dataset, the return value is a distribution score which
        assesses the closeness of each dataset. An example namescape of a training/testing
        dataset pair is:

            train: 0_16_1000_015_1_1_1_1_10_10_10_10_10_10_10_10_0_0_0_MS___train

            test: 0_16_1000_010_0_0_0_0_25_25_25_25_25_25_25_25_0_0_0_MS___train
                            | | | | |  |  |  |  |  |  |  |  |    |||
            score: ~ 4.7 -> out-of-distribution
        
        Scores above 5 are in-distribution whereas below are out-of-distribution. Differing
        classes of control action always constitute to out-of-distribution testing regardless
        of individual similarities of randomization.

        Parameters
        ---
        train : string
            all available training datasets, extracted using datasets module
        test : string
            specific test dataset that the distribution check is for

        Returns
        ---
        score : float
            single number between [0-10] reflecting the closeness of distribution, higher is closer
        """
        traintype = [str.split(x, '_')[-3] for x in train]
        testtype =  str.split(test, '_')[-3]
        comparetype = np.any(np.array([x==y for tr in traintype for x,y in zip(tr,testtype)]))

        if comparetype:
            trainstr1 = [str.split(x, '_')[4:8] for x in train]
            teststr1 = str.split(test, '_')[4:8]
            compare1 = np.array([x==y for tr in trainstr1 for x,y in zip(tr,teststr1)])

            trainstr2 = [str.split(x, '_')[8:-7] for x in train]
            teststr2 = str.split(test, '_')[8:-7]
            compare2 = np.array([int(x)-int(y) for tr in trainstr2 for x,y in zip(tr,teststr2)])

            trainstrf = [str.split(x, '_')[3] for x in train]
            teststrf = str.split(test, '_')[3]
            comparef = np.array([float(x)/100-float(y)/100 for tr in trainstrf for x,y in zip(tr,teststrf)])

            trainstr3 = [str.split(x, '_')[-6:-3] for x in train]
            teststr3 = str.split(test, '_')[-6:-3]
            compare3 = np.array([x==y for tr in trainstr3 for x,y in zip(tr,teststr3)])

            score1 = (compare1==True).sum()
            score2 = (abs(compare2)/100).sum()
            scoref = (abs(comparef)*100).sum()
            score3 = (compare3==True).sum()
            #lcompare = np.size(compare1,axis=0) + np.size(compare3,axis=0) + np.size(compare2,axis=0) + np.size(comparef,axis=0)
            lcompare = score1 + score3

            score = (score1-score2-scoref+score3)/lcompare*10
            score = score if score>=0 else 0.0
            scoreinterp = 'out of distribution' if score<=5 else 'in distribution'

            print(f'Train/Test Dataset Match Score: {score}\nCurrent dataset is {scoreinterp}')
        else:
            score = 0.0
            print(f'Train/Test Datasets are of Different Input Types')
        return score
    
    def cast2original(self, yctx, ytrue, ysim, yerr=None):
        """
        Casts single position pairs into the original deconstructed dataset for data accumulation.

        Parameters
        ---
        yctx : np.array(size=(1,iternum,coordinatenum))
            context of a simulation
        ytrue : np.array(size=(1,iternum,coordinatenum))
            true trajectory
        ysim : np.array(size=(1,iternum,coordinatenum))
            prediction of a simulation
        yerr : np.array(size=(1,iternum,coordinatenum))
            error between prediction and true trajectory
        """
        self.yctx = torch.cat((self.yctx,yctx),dim=0)

        self.ytrue = torch.cat((self.ytrue,ytrue),dim=0)

        self.ysim = torch.cat((self.ysim,ysim),dim=0)

        if yerr is not None:
            self.err = torch.cat((self.err,yerr),dim=0)

    def cast2dict(self, modelname, testname, 
                        rmse={}, nrmse={}, r2={}, fit={}, aic={}, fpe={}):
        """
        Casts metric and model information of a single test into a suitable data array.

        Parameters
        ---
        modelname : string
            name of the model for future reference
        testname : string
            name of the test for future reference
        metrics: dict
            {
            valuec : temporal mean of coordinates
            valuee : temporal mean of environments
            cmean : mean over environments - coordinate
            cvar : variation over environments - coordinate
            emean : mean over coordinates - environment
            evar : variation over coordinates - environment 
            }

        Returns
        ---
        metricdict : dict
            {
            "r": rmse,
            "nr" : nrmse,
            "r2" : r2,
            "f" : fit,
            "aic" : aic,
            "fpe" : fpe     
            }
            
        """
        self.currentdict = {
            "r": rmse,
            "nr" : nrmse,
            "r2" : r2,
            "f" : fit,
            "aic" : aic,
            "fpe" : fpe
        }
        self.metricdict[modelname] = {
            testname : self.currentdict
        }

        return self.currentdict

    def parse_metrics(self, method=None):
        """
        Assigns metrics according to a user selected method, current available metrics are:

            rmse: root mean squared error | dataset 

            nrmse: normalized root mean squared error | dataset

            r2: coefficient of determination | dataset

            fitidx: fit index | dataset 

            aic: akaike information criterion | dataset & model

            fpe: final prediction error | dataset & model

        Parameters
        ---
        method: string
            metric name

        Returns
        ---
        metric: function
            choice of metric function from method
        """
        if method=='rmse':
            metric = rmse 
        elif method=='nrmse':
            metric = nrmse 
        elif method=='r2':
            metric = r_squared 
        elif method=='fitidx':
            metric = fit_index
        elif method=='aic':
            metric = naic
        elif method=='fpe':
            metric = fpe
        return metric
    
    def detachtensors(self):
        """
        Conditions tensors for postprocessing.

            torch.Tensor -> numpy.Array
        """
        self.yctx = self.yctx.to("cpu").detach().numpy()
        self.ytrue = self.ytrue.to("cpu").detach().numpy()
        self.ysim = self.ysim.to("cpu").detach().numpy() 
        self.err = self.err.to("cpu").detach().numpy()
    
    def test(self, method=None, **params):
        """
        Tests a dataset according to a chosen metric as well as evaluates the
        std variation and mean of coordinate and environment projections.

        Parameters
        ---
        method=None: string
            available metric methods
        **params: int
            **keyword arguments for datasize and modelsize for aic&fpe

        Returns
        ---
        testdict: dict
            {            
            valuec: temporal mean of coordinates
            valuee: temporal mean of environments
            cmean: mean over environments - coordinate
            cvar: variation over environments - coordinate
            emean: mean over coordinates - environment
            evar: variation over coordinates - environment
            }
        """
        testmetric = self.parse_metrics(method)

        ry = testmetric(self.ytrue, self.ysim, time_axis=1, **params)

        rync = np.array([np.nan_to_num(col, np.mean(col)) for col in ry.T]).T
        ry_mc = np.nanmean(rync,axis=1) 
        ry_stdc = np.nanstd(rync,axis=1)

        ryne = np.array([np.nan_to_num(row, np.mean(row)) for row in ry])
        ry_me = np.nanmean(ryne,axis=0)
        ry_stde = np.nanstd(ryne,axis=0)

        return {
            "valuec" : rync,
            "valuee" : ryne,
            "cmean" : ry_mc,
            "cvar" : ry_stdc,
            "emean" : ry_me,
            "evar" : ry_stde
        }

    def load(self, *args, **kwargs):
        return super().load(*args, **kwargs)
    
    def normalize(self, *args, **kwargs):
        return super().normalize(*args, **kwargs)
    
    def denormalize(self, *args, **kwargs):
        return super().denormalize(*args, **kwargs)
    
    def normalizeststd(self, *args, **kwargs):
        return super().normalizestd(*args, **kwargs)
    
    def denormalizestd(self, *args, **kwargs):
        return super().denormalizestd(*args, **kwargs)

    def normalizelin(self, *args, **kwargs):
        return super().normalizelin(*args, **kwargs)
    
    def denormalizelin(self, *args, **kwargs):
        return super().denormalizelin(*args, **kwargs)
    
    def normalizecdf(self, *args, **kwargs):
        return super().normalizecdf(*args, **kwargs)
    
    def denormalizecdf(self, *args, **kwargs):
        return super().denormalizecdf(*args, **kwargs)
    
    def seperate_context(self, *args, **kwargs):
        return super().seperate_context(*args, **kwargs)

class plotcfg():

    FONT = {'family' : 'DejaVu Sans',
                        'weight' : 'normal',
                        'size'   : 25}

    LABELCOORDS4D = ["$x$","$y$","$z$" ,"$X$","$Y$","$Z$" ,"$W$",
                        "q0","q1","q2","q3","q4","q5","q6"]
    LABELCOORDS6D = ["$x$","$y$","$z$" ,"$d11$","$d12$","$d13$","$d22$","$d23$" ,"$d33$",
                        "q0","q1","q2","q3","q4","q5","q6"]

    LABELPRED4D = ["$\hat x$","$\hat y$","$\hat z$","$\hat X$","$ \hat Y$","$\hat Z$" ,"$ \hat W$"
                        ,"$ \hat q0$","$ \hat q1$","$ \hat q2$","$ \hat q3$","$ \hat q4$","$ \hat q5$","$ \hat q6$"]
    LABELPRED6D = ["$\hat x$","$\hat y$","$\hat z$","$\hat d11$","$ \hat d12$","$\hat d13$" ,"$ \hat d22$"
                        ,"$\hat d23$" ,"$ \hat d33$","$ \hat q0$","$ \hat q1$","$ \hat q2$","$ \hat q3$"
                        ,"$ \hat q4$","$ \hat q5$","$ \hat q6$"]

    LABELERROR4D = ["$\epsilon_{x}$","$\epsilon_{y}$","$\epsilon_{z}$","$\epsilon_{X}$","$ \epsilon_{Y}$",
                        "$\epsilon_{Z}$" ,"$ \epsilon_{W}$",
                        "$ \epsilon_{q0}$","$ \epsilon_{q1}$","$ \epsilon_{q2}$","$ \epsilon_{q3}$",
                        "$ \epsilon_{q4}$","$ \epsilon_{q5}$","$ \epsilon_{q6}$"] 
    LABELERROR6D = ["$\epsilon_{x}$","$\epsilon_{y}$","$\epsilon_{z}$","$\epsilon_{d11}$","$ \epsilon_{d12}$",
                        "$\epsilon_{d13}$" ,"$ \epsilon_{d22}$","$\epsilon_{d23}$" ,"$ \epsilon_{d33}$",
                        "$ \epsilon_{q0}$","$ \epsilon_{q1}$","$ \epsilon_{q2}$","$ \epsilon_{q3}$",
                        "$ \epsilon_{q4}$","$ \epsilon_{q5}$","$ \epsilon_{q6}$"] 

    LABELID =['RMSE',
              'R²',
              'NRMSE',
              'Fit Index',
              'AIC',
              'FPE']  

    BBOX = dict(facecolor = 'yellow', alpha = 1)

    OFFSET = 72

    ARROWPROPS = dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=90,rad=10")

    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#bca801', '#d8d9a4', '#6e3800', '#ed6300', '#ff5d00', '#e25f3c', '#776cdb', '#65bacd', '#2e4e2d', '#666547',
               '#fb2e01', '#6fcb9f', '#ffe28a', '#fffeb3', '#96ceb4', '#ffeead', '#ff6f69', '#ffcc5c', '#88d8b0']

class postprocess(dataset, plotcfg):
    """
    Postprocessor object to log and plot the acquired data. Works in a script with the metric
    dictionary if preprocessor is already instantiated.

    For a single model and single test:
        * plotlosses
        * plotsim2sim
        * plotsim2real
        * plothorizon
        * plotpredictionerror
        ---
        * plotmetrics_overtime
        * plotmetrics_overjoints
        * plotmetrics_overenvironments
        ---
        * plotvariation_overjoints
        * plotvariation_overenvironments

    For a single model and a list of tests:
        * plotvariation_tests

    For a list of models and a list of tests:
        * plotvariation_models
    
    Test/Model Agnostic:
        * plotmetric

    FineTuning: DEPRECATED
        * plotfinetune_horizon
        * plotfinetune_horizon_comparison
        ---
        * plotfinetune_metrics_overjoints
        * plotfinetune_metrics_overjoints_comparison
        ---
        * plotfinetune_variation_overjoints
        * plotfinetune_variation_overjoints_comparison

    ZeroShot, OneShot, FewShot: DEPRECATED
        * zeroshot_horizon
        * zeroshot_metrics_overjoints
        * zeroshot_variation_overjoints
        ---
        * oneshot_horizon
        * oneshot_metrics_overjoints
        * oneshot_variation_overjoints
        * oneshot_horizon_comparison
        ---
        * fewshot_horizon
        * fewshot_metrics_overjoints
        * fewshot_variation_overjoints
        * fewshot_horizon_comparison

    Also included: helper methods such as spike removal, data smoothening, and median
    plotting.

    Processing locations are taken from preprocessor class.

    Parameters
    ---
        dataname (string): data(s) on which the model(s) is/are trained
        savename (string): postprocessname to save under the postprocess location
    """
    def __init__(self,
                 args,
                 data
                 ):
        
        self.dataname = data['dataname']
        self.testname = ''
        self.modelname = ''
        self.testlist = data['testlist']
        self.modellist = data['modellist']

        self.metricdict = data['metricdict'] # data of each instance of evaluation of each model

        self.modeldict = {} # data of each model from .pt
        loc = data['modelloc']
        for model in self.modellist:
            self.modeldict[model] = torch.load(f=f'{loc}/{model}', weights_only=False, map_location='cpu')

        self.logpath = data['logpath']
        self.figpath = data['figpath']
        self.sumpath = data['sumpath']

        d = datetime.datetime.now()
        self.savename = args.process_name if args.process_name is not None else d.strftime("%d%m%Y_%Hh%Mm%Ss")

        plt.rc('font', **self.FONT)
        plt.rc('axes', prop_cycle=matplotlib.cycler(color=self.COLORS))

    def __str__(self):
        return 'Posprocess Object Instantiated'

    def logdata(self, *args, **kwargs):
        """
        Data logger into txt file, verbose summary.
        """
        logpath = os.path.join(self.logpath,self.dataname,self.savename)
        os.makedirs(logpath, exist_ok=True)
        logto = os.path.join(logpath,'testresults.txt')
        
        with open(logto, 'a+') as f:
            f.write(self.tabulate(*args, **kwargs))

    def tabulate(self, modelname=None, testname=None, specificto='test', *args, **kwargs):
        """
        Tabulates testing metrics.
            
            Parameters
            ---
            specificto (string): create tables for all, test or model metrics
        """
        model = self.modelname if modelname is None else modelname
        test = self.testname if testname is None else testname
        if specificto=='test':
            r2 = self.metricdict[model][test]['metrics']['r2']['emean']
            r = self.metricdict[model][test]['metrics']['r']['emean']
            f = self.metricdict[model][test]['metrics']['f']['emean']
            nr = self.metricdict[model][test]['metrics']['nr']['emean']
            aic = self.metricdict[model][test]['metrics']['aic']['emean']
            fpe = self.metricdict[model][test]['metrics']['fpe']['emean']
        
            data = np.array([r2,r,f,nr,aic,fpe]).T
            headers = ['r2','r','f','nr','daic','fpe']
            str1 = f'--- Model Name: {model} ---\n'
            str2 = f'--- Test Name: {test} ---\n'
            str3 = '\n\n\n\n\n'
            table = tabulate(data, headers, *args, tablefmt='fancy_grid', floatfmt='.4', **kwargs)
            print(str1)
            print(str2)
            print(table)
            print(str3)

            table = str1 + str2 + table + str3
        
        elif specificto=='model':
            r2t = []
            headers = []
            for test in self.testlist:
                r2 = self.metricdict[model][test]['metrics']['r2']['emean']
                r2t.append(r2)
                headers.append(test)
            data = np.array(r2t).T
            str1 = f'--- Model Name: {model} ---\n'
            str2 = 'All Tests Are Evaluated Using r2\n'
            str3 = '\n\n\n\n\n'
            table = tabulate(data, headers, *args, tablefmt='fancy_grid', floatfmt='.4', **kwargs)
            print(str1)
            print(str2)
            print(table)
            print(str3)

            table = str1 + str2 + table + str3

        elif specificto=='all':
            r2m = []
            r2t = []
            headers = []
            for model in self.modellist:
                for test in self.testlist:
                    r2 = self.metricdict[model][test]['metrics']['r2']['emean']
                    r2t.append(r2)
                r2m.append(np.array(r2t).mean(axis=0))
                r2t = []
                headers.append(model)

            data = np.array(r2m).T
            str1 = 'All Models Are Evaluated Using r2\n'
            str2 = '\n\n\n\n\n'
            table = tabulate(data, headers, tablefmt='fancy_grid', floatfmt='.4')
            print(str1)
            print(table)
            print(str2)

            table = str1 + table + str2
        
        return table

    def configure_models(self, modelname):
        """
        Logs metric data from preprocessed test datasets into the metric dict that stores 
        accuracy information for all available tests for all available models. Should be 
        called after the successful testing of a model.

        Parameters
        ---
            modelname (string): name of the model
        """
        self.metricdict.update(
            {
                modelname : {

                }
            }
        )
        
        self.modelname = modelname

        modelpath = os.path.join(self.figpath,self.dataname,self.savename,self.modelname)
        os.makedirs(modelpath, exist_ok=True)
    
    def configure_tests(self, ytrue, ysim, yerr, yctx, testmetrics, testscore, testname, modelname):
        """
        Logs metric data from preprocessed test datasets into the metric dict that stores 
        accuracy information for all available tests for all available models. Should be
        called after each successful test on a model.

        Parameters
        ---
            ytrue (torch.Tensor): true accumulated value of a test on a model
            ysim (torch.Tensor): simulation accumulated value of a test on a model
            yerr (torch.Tensor): error between ytrue and ysim
            yctx (torch.Tensor): context window of ytrue
            testmetrics (dict): all metrics on a test on a model
            testscore (float): test score of the specific test
            testname (string): name of the test
            modelname (string): name of the model
        """
        self.metricdict[modelname].update(
        {
            testname : {
                'metrics' : testmetrics,
                'score' : testscore,
                'ytrue' : ytrue,
                'ysim' : ysim,
                'yerr' : yerr,
                'yctx' : yctx
            }
        }
        )

        self.testname = testname

        testpath = os.path.join(self.figpath,self.dataname,self.savename,self.modelname,self.testname)
        os.makedirs(testpath, exist_ok=True)

    def spike_removal(y, 
                  width_threshold, 
                  prominence_threshold=None, 
                  moving_average_window=10, 
                  width_param_rel=0.8, 
                  interp_type='linear'):
        """
        Detects and replaces spikes in the input spectrum with interpolated values. Algorithm first 
        published by N. Coca-Lopez in Analytica Chimica Acta. https://doi.org/10.1016/j.aca.2024.342312

        Parameters
        ---
            y (numpy.ndarray): Input spectrum intensity.
            width_threshold (float): Threshold for peak width.
            prominence_threshold (float): Threshold for peak prominence.
            moving_average_window (int): Number of points in moving average window.
            width_param_rel (float): Relative height parameter for peak width.
            tipo: type of interpolation (linear, quadratic, cubic)
        
        Returns
        ---
            yout (numpy.ndarray): 
                Signal with spikes replaced by interpolated values.
        """
        peaks, _ = find_peaks(y, prominence=prominence_threshold)
        
        spikes = np.zeros(len(y))
        
        widths = peak_widths(y, peaks)[0]
        
        widths_ext_a = peak_widths(y, peaks, rel_height=width_param_rel)[2]
        widths_ext_b = peak_widths(y, peaks, rel_height=width_param_rel)[3]

        for a, width, ext_a, ext_b in zip(range(len(widths)), widths, widths_ext_a, widths_ext_b):
            if width < width_threshold:
                spikes[int(ext_a) - 1: int(ext_b) + 2] = 1 
                
        y_out = y.copy()
        
        for i, spike in enumerate(spikes):
            if spike != 0: 
                window = np.arange(i - moving_average_window, i + moving_average_window + 1) 
                window_exclude_spikes = window[spikes[window] == 0]
                interpolator = interpolate.interp1d(window_exclude_spikes, y[window_exclude_spikes], kind=interp_type) 
                y_out[i] = interpolator(i)
                
        return y_out

    def plotlosses(self, modelnames=None, save=False, labels=None):
        """
        Plots the training and validation losses as recorded in the checkpoint save of a chosen model. If 
        more than 1 model is available, losses are plotted for each provided model.

        Parameters
        ---
            modelnames (string): modelnames to draw loss data from
            save (bool): if True, save the model under figpath (/plots/{data_name})
        """
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(20,20))
        for i, model in enumerate(self.modellist if modelnames is None else modelnames):
            losslabel = model if labels is None else labels[i]

            ax[0].semilogy(self.modeldict[model]['trloss'], linewidth=0.5)
            ax[0].semilogy(savgol_filter(self.modeldict[model]['trloss'], 50, 5),linewidth=3, label=losslabel)
            ax[0].set_title(f'Training Loss Over Iterations')
            ax[0].grid(True)
            leg = ax[0].legend(loc='best',fancybox=True, 
                             framealpha=1, shadow=True, borderpad=1, fontsize='small', title='model type')
            lframe = leg.get_frame()
            lframe.set_facecolor('#b4aeae')
            lframe.set_edgecolor('black')
            lframe.set_alpha(1)
            for line in leg.get_lines():
                line.set_linewidth(4.0)
            ax[0].set_xlabel('Iterations')
            ax[0].set_ylabel(f'Loss')

            ax[1].semilogy(self.modeldict[model]['valloss'], linewidth=0.5)
            ax[1].semilogy(savgol_filter(self.modeldict[model]['valloss'], 50, 5),linewidth=3, label=losslabel)
            ax[1].set_title(f'Validation Loss Over Iterations')
            ax[1].grid(True)
            leg = ax[1].legend(loc='best',fancybox=True, 
                             framealpha=1, shadow=True, borderpad=1, fontsize='small', title='model type')
            lframe = leg.get_frame()
            lframe.set_facecolor('#b4aeae')
            lframe.set_edgecolor('black')
            lframe.set_alpha(1)
            for line in leg.get_lines():
                line.set_linewidth(4.0)
            ax[1].set_xlabel('Iterations')
            ax[1].set_ylabel(f'Loss')
        lossname = 'loss_single.png' if len(self.modellist)==1 else 'loss_multi.png'
        fig.suptitle(f'Loss Variation Over Iterations', y = 1.0, weight='bold')
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/{lossname}',
                        bbox_inches='tight')
        else:
            plt.show()

    def plotsim2sim(self, modelname=None, testname=None, of='best', dim=4, save=False, 
                    title='Ground Truth vs Model Prediction for a Single Simulation'):
        """
        Plots the true generated pose, estimated pose and relative simulation error on the original
        time frame defined by the true generated pose.

        Parameters
        ---
            modelname (string): name of the current model
            testname (string): name of the current string
            of (string): best-worst-median-random plots of sim2sim
            dim (int): orientation dimension [3,4,6]
            save (bool): if True, save the model under figpath (/plots/{data_name})
            title (string): title variability
        """
        model = self.modelname if modelname is None else modelname
        test = self.testname if testname is None else testname
        if of=='best':
            idx = np.argmax(self.metricdict[model][test]['metrics']['r2']['cmean'])
        elif of=='worst':
            idx = np.argmin(self.metricdict[model][test]['metrics']['r2']['cmean'])
        elif of=='median':
            idx = np.argsort(self.metricdict[model][test]['metrics']['r2']['cmean'])[len(self.metricdict[model][test]['metrics']['r2']['cmean'])//2]
        elif of=='random':
            idx = np.random.random_integers(low = 0, high=len(self.metricdict[model][test]['metrics']['r2']['cmean']))
        else:
            raise ValueError

        true = self.metricdict[model][test]['ytrue'][idx,:,:]
        sim = self.metricdict[model][test]['ysim'][idx,:,:]
        err = self.metricdict[model][test]['yerr'][idx,:,:]

        if dim==4:
            fig, ax = plt.subplots(7, 2, figsize=(20, 20))

            k = 0
            for j in range(2):
                for i in range(7):
                    ax[i,j].plot(true[:,k] ,'k', 
                        label=self.LABELCOORDS4D[k],linewidth=3)
                    ax[i,j].plot(sim[:,k], 'b', 
                        label=self.LABELPRED4D[k],linewidth=1)
                    ax[i,j].plot(err[:,k], 'r', 
                        label=self.LABELERROR4D[k],linewidth=1)
                    ax[i,j].grid(True)
                    ax[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax[i,j].set_xlabel('Iterations')
                    k=k+1

        elif dim==6:
            fig = plt.figure(figsize = (20,20))
            gs = matplotlib.gridspec.GridSpec(9,2, figure=fig)

            k = 0
            for i1 in range(9):
                ax1 = fig.add_subplot(gs[i1,0])
                ax1.plot(true[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=3)
                ax1.plot(sim[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=1)
                ax1.plot(err[:,k], 'r', 
                    label=self.LABELERROR6D[k],linewidth=1)
                ax1.grid(True)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax1.set_xlabel('Iterations')
                k=k+1
            
            for i2 in range(7):
                ax2 = fig.add_subplot(gs[i2,1])
                ax2.plot(true[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=3)
                ax2.plot(sim[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=1)
                ax2.plot(err[:,k], 'r', 
                    label=self.LABELERROR6D[k],linewidth=1)
                ax2.grid(True)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax2.set_xlabel('Iterations')
                k=k+1

        score = np.round(self.metricdict[model][test]['score'],decimals=2)

        fig.suptitle(title + 
                     f'\n\nTest Name: {test}\n\n Model Name: {model}'
                     f'\n\nTest Score: {score}/10.0', y=1.01, weight='bold')
        fig.tight_layout()
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/{model}/{test}/sim2sim_{of}.png',
                        bbox_inches='tight')
        else:
            plt.show()

    def plotsim2real(self, modelname=None, testname=None, of='best', dim=4, save=False, 
                    title='Acquisation vs Model Prediction for a Single Simulation'):
        """
        Plots the true acquired and conditioned pose, estimated pose and relative simulation error 
        on the original time frame defined by the true acquired pose. Real data has to be preprocessed
        before using plotsim2real method.

        Parameters
        ---
            modelname (string): name of the current model
            testname (string): name of the current string
            of (string): best-worst-median-random plots of sim2sim
            dim (int): orientation dimension [3,4,6]
            save (bool): if True, save the model under figpath (/plots/{data_name})
            title (string): title variability
        """
        model = self.modelname if modelname is None else modelname
        test = self.testname if testname is None else testname
        if of=='best':
            idx = np.argmax(self.metricdict[model][test]['metrics']['r2']['cmean'])
        elif of=='worst':
            idx = np.argmin(self.metricdict[model][test]['metrics']['r2']['cmean'])
        elif of=='median':
            idx = np.argsort(self.metricdict[model][test]['metrics']['r2']['cmean'])[len(self.metricdict[model][test]['metrics']['r2']['cmean'])//2]
        elif of=='random':
            idx = np.random.random_integers(low = 0, high=len(self.metricdict[model][test]['metrics']['r2']['cmean']))
        else:
            raise ValueError
        
        real = self.metricdict[model][test]['ytrue'][idx,:,:]
        sim = self.metricdict[model][test]['ysim'][idx,:,:]
        err = self.metricdict[model][test]['yerr'][idx,:,:]

        if dim==4:
            fig, ax = plt.subplots(7, 2, figsize=(20, 20))

            k = 0
            for j in range(2):
                for i in range(7):
                    ax[i,j].plot(real[:,k] ,'k', 
                        label=self.LABELCOORDS4D[k],linewidth=3)
                    ax[i,j].plot(sim[:,k], 'b', 
                        label=self.LABELPRED4D[k],linewidth=1)
                    ax[i,j].plot(err[:,k], 'r', 
                        label=self.LABELERROR4D[k],linewidth=1)
                    ax[i,j].grid(True)
                    ax[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax[i,j].set_xlabel('Iterations')
                    k=k+1

        elif dim==6:
            fig = plt.figure(figsize = (20,20))
            gs = matplotlib.gridspec.GridSpec(9,2, figure=fig)

            k = 0
            for i1 in range(9):
                ax1 = fig.add_subplot(gs[i1,0])
                ax1.plot(real[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=3)
                ax1.plot(sim[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=1)
                ax1.plot(err[:,k], 'r', 
                    label=self.LABELERROR6D[k],linewidth=1)
                ax1.grid(True)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax1.set_xlabel('Iterations')
                k=k+1
            
            for i2 in range(7):
                ax2 = fig.add_subplot(gs[i2,1])
                ax2.plot(real[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=3)
                ax2.plot(sim[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=1)
                ax2.plot(err[:,k], 'r', 
                    label=self.LABELERROR6D[k],linewidth=1)
                ax2.grid(True)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax2.set_xlabel('Iterations')
                k=k+1

        score = np.round(self.metricdict[model][test]['score'],decimals=2)

        fig.suptitle(title + 
                     f'\n\nTest Name: {test}\n\n Model Name: {model}'
                     f'\n\nTest Score: {score}/10.0', y=1.01, weight='bold')
        fig.tight_layout()
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/{model}/{test}/sim2real_{of}.png',
                        bbox_inches='tight')
        else:
            plt.show()

    def plothorizon(self, modelname=None, testname=None, of='best', dim=4, iter=1000, ctx=200, save=False, 
                    title='Horizon Prediction for a Single Simulation'):
        """
        Plots the horizon prediction for a given simulation determined by the user.

        Parameters
        ---
            modelname (string): name of the current model
            testname (string): name of the current string
            dim (np.array): orientation dimension [3,4,6]
            of (string): best-worst-median-random plots of sim2sim
            iter (int): number of total iterations
            ctx (int): number of context iterations
            save (bool): if True, save the model under figpath (/plots/{data_name})
            title (string): title variability
        """
        model = self.modelname if modelname is None else modelname
        test = self.testname if testname is None else testname
        if of=='best':
            idx = np.argmax(self.metricdict[model][test]['metrics']['r2']['cmean'])
        elif of=='worst':
            idx = np.argmin(self.metricdict[model][test]['metrics']['r2']['cmean'])
        elif of=='median':
            idx = np.argsort(self.metricdict[model][test]['metrics']['r2']['cmean'])[len(self.metricdict[self.modelname][self.testname]['metrics']['r2']['cmean'])//2]
        elif of=='random':
            idx = np.random.random_integers(low = 0, high=len(self.metricdict[model][test]['metrics']['r2']['cmean']))
        else:
            raise ValueError
        
        yctx = self.metricdict[model][test]['yctx'][idx,:,:]
        sim = self.metricdict[model][test]['ysim'][idx,:,:]
        err = self.metricdict[model][test]['yerr'][idx,:,:]
        true = self.metricdict[model][test]['ytrue'][idx,:,:]
        total = np.concatenate([yctx,true])

        t_context = np.arange(1, ctx)
        t_prediction = np.arange(1, iter-ctx+1) + ctx
        t_total = np.arange(0, iter)

        if dim==4:
            fig, ax = plt.subplots(7, 2, figsize=(20, 20))

            k = 0
            for j in range(2):
                for i in range(7):
                    ax[i,j].axvline(x = t_prediction[-1], 
                        color = 'k', linestyle='--')
                    ax[i,j].axvline(x = t_context[-1], 
                        color = 'k', linestyle='--')
                    ax[i,j].axvspan(t_context[0], t_context[-1],
                        facecolor='lime', alpha=0.2)
                    ax[i,j].plot(t_total,total[:,k] ,'k', 
                        label=self.LABELCOORDS4D[k],linewidth=3)
                    ax[i,j].plot(t_prediction,sim[:,k], 'b', 
                        label=self.LABELPRED4D[k],linewidth=1)
                    ax[i,j].plot(t_prediction,err[:,k], 'r', 
                        label=self.LABELERROR4D[k],linewidth=1)
                    ax[i,j].grid(True)
                    ax[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax[i,j].set_xlabel('Iterations')
                    k=k+1

        elif dim==6:
            fig = plt.figure(figsize = (20,20))
            gs = matplotlib.gridspec.GridSpec(9,2, figure=fig)

            k = 0
            for i1 in range(9):
                ax1 = fig.add_subplot(gs[i1,0])
                ax1.axvline(x = t_prediction[-1], 
                    color = 'k', linestyle='--')
                ax1.axvline(x = t_context[-1], 
                    color = 'k', linestyle='--')
                ax1.axvspan(t_context[0], t_context[-1],
                        facecolor='lime', alpha=0.2)

                ax1.plot(t_total,total[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=3)
                ax1.plot(t_prediction,sim[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=1)
                ax1.plot(t_prediction,err[:,k], 'r', 
                    label=self.LABELERROR6D[k],linewidth=1)
                ax1.grid(True)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax1.set_xlabel('Iterations')
                k=k+1

            for i2 in range(7):
                ax2 = fig.add_subplot(gs[i2,1])
                ax2.axvline(x = t_prediction[-1], 
                    color = 'k', linestyle='--')
                ax2.axvline(x = t_context[-1], 
                    color = 'k', linestyle='--')
                ax2.axvspan(t_context[0], t_context[-1],
                        facecolor='lime', alpha=0.2)

                ax2.plot(t_total,total[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=3)
                ax2.plot(t_prediction,sim[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=1)
                ax2.plot(t_prediction,err[:,k], 'r', 
                    label=self.LABELERROR6D[k],linewidth=1)
                ax2.grid(True)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax2.set_xlabel('Iterations')
                k=k+1

        score = np.round(self.metricdict[model][test]['score'],decimals=2)

        fig.suptitle(title + 
                     f'\n\nTest Name: {test}\n\n Model Name: {model}'
                     f'\n\nTest Score: {score}/10.0', y=1.01, weight='bold')
        fig.tight_layout()
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/{model}/{test}/horizon_{of}.png',
                        bbox_inches='tight')
        else:
            plt.show()

    def plotpredictionerror(self, modelname=None, testname=None, of='best', dim=4, save=False, 
                            title='Prediction Error for a Single Simulation'):
        """
        Plots the horizon prediction for a given simulation determined by the user.

        Parameters
        ---
            of (string): best-worst-median-random plots of sim2sim
            dim (int): orientation dimension [3,4,6]
            save (bool): if True, save the model under figpath (/plots/{data_name})
            title (string): title variability
        """
        model = self.modelname if modelname is None else modelname
        test = self.testname if testname is None else testname
        if of=='best':
            idx = np.argmax(self.metricdict[model][test]['metrics']['r2']['cmean'])
        elif of=='worst':
            idx = np.argmin(self.metricdict[model][test]['metrics']['r2']['cmean'])
        elif of=='median':
            idx = np.argsort(self.metricdict[model][test]['metrics']['r2']['cmean'])[len(self.metricdict[self.modelname][self.testname]['metrics']['r2']['cmean'])//2]
        elif of=='random':
            idx = np.random.random_integers(low = 0, high=len(self.metricdict[model][test]['metrics']['r2']['cmean']))
        else:
            raise ValueError
        
        sim = self.metricdict[model][test]['ysim'][idx,:,:]
        err = self.metricdict[model][test]['yerr'][idx,:,:]
        true = self.metricdict[model][test]['ytrue'][idx,:,:]

        if dim==4:
            fig, ax = plt.subplots(7, 2, figsize=(20, 20))

            k = 0
            for j in range(2):
                for i in range(7):
                    errc = err[:,k]
                    ax[i,j].plot(errc, 'r', 
                        label=self.LABELERROR4D[k],linewidth=3)
                    ax[i,j].grid(True)
                    ax[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax[i,j].set_xlabel('Iterations')
                    k=k+1

        elif dim==6:
            fig = plt.figure(figsize = (20,20))
            gs = matplotlib.gridspec.GridSpec(9,2, figure=fig)

            k = 0
            for i1 in range(9):
                ax1 = fig.add_subplot(gs[i1,0])
                errc = err[:,k]
                ax1.plot(errc, 'r', 
                    label=self.LABELERROR6D[k],linewidth=3)
                ax1.grid(True)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax1.set_xlabel('Iterations')
                k=k+1
            
            for i2 in range(7):
                ax2 = fig.add_subplot(gs[i2,1])
                errc = err[:,k]
                ax2.plot(errc, 'r', 
                    label=self.LABELERROR6D[k],linewidth=3)
                ax2.grid(True)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax2.set_xlabel('Iterations')
                k=k+1

        score = np.round(self.metricdict[model][test]['score'],decimals=2)

        fig.suptitle(title + 
                     f'\n\nTest Name: {test}\n\n Model Name: {model}'
                     f'\n\nTest Score: {score}/10.0', y=1.01, weight='bold')
        fig.tight_layout()
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/{model}/{test}/percent_error_{of}.png',
                        bbox_inches='tight')
        else:
            plt.show()

    def plotmetrics_overtime(self, modelname=None, testname=None, save=False,
                            title = f'Cumulative Accuracy Projected on Iteration Steps'):
        """
        Plots the variation of metrics over a single test on a single model over the entire iteration
        regime.

        Parameters
        ---
            modelname (string): name of the current model
            testname (string): name of the current string
            save (bool): if True, save the model under figpath (/plots/{data_name})
            title (string): title variability
        """
        model = self.modelname if modelname is None else modelname
        test = self.testname if testname is None else testname
        true = self.metricdict[model][test]['ytrue']
        sim = self.metricdict[model][test]['ysim']
        r2 = r_squared(true, sim, time_axis=0)
        r2m = np.mean(r2, axis=1)

        r = rmse(true, sim, time_axis=0)
        rm = np.mean(r, axis=1)

        f = fit_index(true, sim, time_axis=0)
        fm = np.mean(f, axis=1)

        nr = nrmse(true, sim, time_axis=0)
        nrm = np.mean(nr, axis=1)       

        fig,[[ax1, ax2],[ax3, ax4]]  = plt.subplots(2, 2,figsize=(20, 20))

        ax1.plot(r2m,
                        linewidth=2)
        ax1.axhline(np.mean(r2m),c='r',
                        linewidth=2.0,linestyle='solid') 
        ax1.scatter(np.nanargmin(r2m),
                        np.nanmin(r2m),
                        marker='o', color='red', s=120)
        ax1.annotate(f'worst iter',
                     (np.nanargmin(r2m),np.nanmin(r2m)),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax1.scatter(np.nanargmax(r2m),
                        np.nanmax(r2m),
                        marker='o', color='red', s=120)
        ax1.annotate(f'best iter',
                     (np.nanargmax(r2m),np.nanmax(r2m)),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax1.axhspan(np.nanpercentile(r2m,25), 
                        np.nanpercentile(r2m,75), 
                        facecolor='gold', alpha=0.2)
        ax1.grid(visible=True, which='both')
        ax1.set_title('$R^{2}$ Value')
        ax1.set(xlabel='Iteration')
        

        ax2.plot(rm,
                        linewidth=2.0)
        ax2.axhline(np.mean(rm),c='r',
                        linewidth=2.0,linestyle='solid') 
        ax2.scatter(np.nanargmin(rm),
                        np.nanmin(rm),
                        marker='o', color='red', s=120)
        ax2.annotate(f'best iter',
                     (np.nanargmin(rm),np.nanmin(rm)),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax2.scatter(np.nanargmax(rm),
                        np.nanmax(rm),
                        marker='o', color='red', s=120)
        ax2.annotate(f'worst iter',
                     (np.nanargmax(rm),np.nanmax(rm)),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax2.axhspan(np.nanpercentile(rm,25), 
                        np.nanpercentile(rm,75), 
                        facecolor='gold', alpha=0.2)
        ax2.grid(visible=True, which='both')
        ax2.set_title('$RMSE$ Value')
        ax2.set(xlabel='Iteration')


        ax3.plot(fm,
                        linewidth=2.0)
        ax3.axhline(np.mean(fm),c='r',
                        linewidth=2.0,linestyle='solid') 
        ax3.scatter(np.nanargmin(fm),
                        np.nanmin(fm),
                        marker='o', color='red', s=120)
        ax3.annotate(f'worst iter',
                     (np.nanargmin(fm),np.nanmin(fm)),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax3.scatter(np.nanargmax(fm),
                        np.nanmax(fm),
                        marker='o', color='red', s=120)
        ax3.annotate(f'best iter',
                     (np.nanargmax(fm),np.nanmax(fm)),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax3.axhspan(np.nanpercentile(fm,25), 
                        np.nanpercentile(fm,75), 
                        facecolor='gold', alpha=0.2)
        ax3.grid(visible=True, which='both')
        ax3.set_title('$fit$ Value')
        ax3.set(xlabel='Iteration')


        ax4.plot(nrm,
                        linewidth=2.0)
        ax4.axhline(np.mean(nrm),c='r',
                        linewidth=2.0,linestyle='solid') 
        ax4.scatter(np.nanargmin(nrm),
                        np.nanmin(nrm),
                        marker='o', color='red', s=120)
        ax4.annotate(f'best iter',
                     (np.nanargmin(nrm),np.nanmin(nrm)),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax4.scatter(np.nanargmax(nrm),
                        np.nanmax(nrm),
                        marker='o', color='red', s=120)
        ax4.annotate(f'worst iter',
                     (np.nanargmax(nrm),np.nanmax(nrm)),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax4.axhspan(np.nanpercentile(nrm,25), 
                        np.nanpercentile(nrm,75), 
                        facecolor='gold', alpha=0.2)
        ax4.grid(visible=True, which='both')
        ax4.set_title('$NRMSE$ Value')
        ax4.set(xlabel='Iteration')

        score = np.round(self.metricdict[model][test]['score'],decimals=2)

        fig.tight_layout()
        fig.suptitle(title + 
                     f'\n\nTest Name: {test}\n\n Model Name: {model}'
                     f'\n\nTest Score: {score}/10.0', y=1.15, weight='bold')
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/{model}/{test}/metrics_overtime.png',
                        bbox_inches='tight')
        else:
            plt.show()


    def plotmetrics_overenvironments(self, modelname=None, testname=None, save=False, 
                                    title='Cumulative Accuracy Metrics Mean over Coordinates Projected on All Environments'):
        """
        Plots the variation of metrics over a single test on a single model over the available 
        coordinates.

        Parameters
        ---
            modelname (string): name of the current model
            testname (string): name of the current string
            save (bool): if True, save the model under figpath (/plots/{data_name})
            title (string): title variability
        """
        model = self.modelname if modelname is None else modelname
        test = self.testname if testname is None else testname
        metrics = self.metricdict[model][test]['metrics']
        fig,[[ax1, ax2],[ax3, ax4]]  = plt.subplots(2, 2,figsize=(20, 20))
        
        ax1.plot(metrics['r2']['cmean'],
                        linewidth=2)
        ax1.axhline(np.mean(metrics['r2']['cmean']),c='r',
                        linewidth=2.0,linestyle='solid') 
        ax1.scatter(np.nanargmin(metrics['r2']['cmean']),
                        np.nanmin(metrics['r2']['cmean']),
                        marker='o', color='red', s=120)
        ax1.annotate(f'worst env',
                     (np.nanargmin(metrics['r2']['cmean']),np.nanmin(metrics['r2']['cmean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax1.scatter(np.nanargmax(metrics['r2']['cmean']),
                        np.nanmax(metrics['r2']['cmean']),
                        marker='o', color='red', s=120)
        ax1.annotate(f'best env',
                     (np.nanargmax(metrics['r2']['cmean']),np.nanmax(metrics['r2']['cmean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax1.axhspan(np.nanpercentile(metrics['r2']['cmean'],25), 
                        np.nanpercentile(metrics['r2']['cmean'],75), 
                        facecolor='gold', alpha=0.2)
        ax1.grid(visible=True, which='both')
        ax1.set_title('$R^{2}$ Value')
        ax1.set(xlabel='Environment')
        

        ax2.plot(metrics['r']['cmean'],
                        linewidth=2.0)
        ax2.axhline(np.mean(metrics['r']['cmean']),c='r',
                        linewidth=2.0,linestyle='solid') 
        ax2.scatter(np.nanargmin(metrics['r']['cmean']),
                        np.nanmin(metrics['r']['cmean']),
                        marker='o', color='red', s=120)
        ax2.annotate(f'best env',
                     (np.nanargmin(metrics['r']['cmean']),np.nanmin(metrics['r']['cmean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax2.scatter(np.nanargmax(metrics['r']['cmean']),
                        np.nanmax(metrics['r']['cmean']),
                        marker='o', color='red', s=120)
        ax2.annotate(f'worst env',
                     (np.nanargmax(metrics['r']['cmean']),np.nanmax(metrics['r']['cmean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax2.axhspan(np.nanpercentile(metrics['r']['cmean'],25), 
                        np.nanpercentile(metrics['r']['cmean'],75), 
                        facecolor='gold', alpha=0.2)
        ax2.grid(visible=True, which='both')
        ax2.set_title('$RMSE$ Value')
        ax2.set(xlabel='Environment')


        ax3.plot(metrics['f']['cmean'],
                        linewidth=2.0)
        ax3.axhline(np.mean(metrics['f']['cmean']),c='r',
                        linewidth=2.0,linestyle='solid') 
        ax3.scatter(np.nanargmin(metrics['f']['cmean']),
                        np.nanmin(metrics['f']['cmean']),
                        marker='o', color='red', s=120)
        ax3.annotate(f'worst env.',
                     (np.nanargmin(metrics['f']['cmean']),np.nanmin(metrics['f']['cmean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax3.scatter(np.nanargmax(metrics['f']['cmean']),
                        np.nanmax(metrics['f']['cmean']),
                        marker='o', color='red', s=120)
        ax3.annotate(f'best env.',
                     (np.nanargmax(metrics['f']['cmean']),np.nanmax(metrics['f']['cmean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax3.axhspan(np.nanpercentile(metrics['f']['cmean'],25), 
                        np.nanpercentile(metrics['f']['cmean'],75), 
                        facecolor='gold', alpha=0.2)
        ax3.grid(visible=True, which='both')
        ax3.set_title('$fit$ Value')
        ax3.set(xlabel='Environment')


        ax4.plot(metrics['nr']['cmean'],
                        linewidth=2.0)
        ax4.axhline(np.mean(metrics['nr']['cmean']),c='r',
                        linewidth=2.0,linestyle='solid') 
        ax4.scatter(np.nanargmin(metrics['nr']['cmean']),
                        np.nanmin(metrics['nr']['cmean']),
                        marker='o', color='red', s=120)
        ax4.annotate(f'best env.',
                     (np.nanargmin(metrics['nr']['cmean']),np.nanmin(metrics['nr']['cmean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax4.scatter(np.nanargmax(metrics['nr']['cmean']),
                        np.nanmax(metrics['nr']['cmean']),
                        marker='o', color='red', s=120)
        ax4.annotate(f'worst env.',
                     (np.nanargmax(metrics['nr']['cmean']),np.nanmax(metrics['nr']['cmean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax4.axhspan(np.nanpercentile(metrics['nr']['cmean'],25), 
                        np.nanpercentile(metrics['nr']['cmean'],75), 
                        facecolor='gold', alpha=0.2)
        ax4.grid(visible=True, which='both')
        ax4.set_title('$NRMSE$ Value')
        ax4.set(xlabel='Environment')

        score = np.round(self.metricdict[model][test]['score'],decimals=2)

        fig.tight_layout()
        fig.suptitle(title + 
                     f'\n\nTest Name: {test}\n\n Model Name: {model}'
                     f'\n\nTest Score: {score}/10.0', y=1.15, weight='bold')
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/{model}/{test}/metrics_overenvironments.png',
                        bbox_inches='tight')
        else:
            plt.show()

    def plotvariation_overenvironments(self, modelname=None, testname=None, save=False, 
                                       title='Cumulative Metric Variation Projected on All Environments'):
        """
        Plots the variation of metrics over a single test on a single model over the available 
        environments.

        Parameters
        ---
            modelname (string): name of the current model
            testname (string): name of the current string
            save (bool): if True, save the model under figpath (/plots/{data_name})
            title (string): title variability
        """
        model = self.modelname if modelname is None else modelname
        test = self.testname if testname is None else testname
        metrics = self.metricdict[model][test]['metrics']
        m1list = [metrics['r']['valuee'].T,metrics['r2']['valuee'].T,
                    metrics['nr']['valuee'].T,metrics['f']['valuee'].T]

        fig1 = plt.figure(figsize = (20,20))
        gs1 = matplotlib.gridspec.GridSpec(4,1, figure=fig1)
        for j in range(4):
            ax1 = fig1.add_subplot(gs1[j])
            ax1.boxplot(m1list[j],patch_artist = True, 
                            boxprops = dict(facecolor = "lightblue"), 
                            medianprops = dict(color = "green", linewidth = 1.5), 
                            whiskerprops = dict(color = "red", linewidth = 2), 
            )
            ax1.set_title(f'${self.LABELID[j]}$')
            ax1.set_xticks([])
            ax1.grid(alpha=0.5)
            ax1.set_xlabel('Environments')

        score = np.round(self.metricdict[model][test]['score'],decimals=2)

        fig1.tight_layout()
        fig1.suptitle(title + 
                     f'\n\nTest Name: {test}\n\n Model Name: {model}'
                     f'\n\nTest Score: {score}/10.0', y=1.18, weight='bold')
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/{model}/{test}/variation_overenvironemnts.png',
                        bbox_inches='tight')
        else:
            plt.show()

    def plotmetrics_overjoints(self, modelname=None, testname=None, save=False, 
                                title='Cumulative Accuracy Metrics Mean over Environments Projected on All Coordinates'):
        """
        Plots the variation of metrics over a single test on a single model over the available 
        environments.

        Parameters
        ---
            modelname (string): name of the current model
            testname (string): name of the current string
            save (bool): if True, save the model under figpath (/plots/{data_name})
            title (string): title variability
        """
        model = self.modelname if modelname is None else modelname
        test = self.testname if testname is None else testname
        metrics = self.metricdict[model][test]['metrics']
        fig,[[ax1, ax2],[ax3, ax4]]  = plt.subplots(2, 2,figsize=(20, 20))
        
        ax1.plot(metrics['r2']['emean'],
                        linewidth=2)
        ax1.axhline(np.mean(metrics['r2']['emean']),c='r',
                        linewidth=2.0,linestyle='solid') 
        ax1.scatter(np.nanargmin(metrics['r2']['emean']),
                        np.nanmin(metrics['r2']['emean']),
                        marker='o', color='red', s=120)
        ax1.annotate(f'worst coord',
                     (np.nanargmin(metrics['r2']['emean']),np.nanmin(metrics['r2']['emean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax1.scatter(np.nanargmax(metrics['r2']['emean']),
                        np.nanmax(metrics['r2']['emean']),
                        marker='o', color='red', s=120)
        ax1.annotate(f'best coord',
                     (np.nanargmax(metrics['r2']['emean']),np.nanmax(metrics['r2']['emean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax1.axhspan(np.nanpercentile(metrics['r2']['emean'],25), 
                        np.nanpercentile(metrics['r2']['emean'],75), 
                        facecolor='gold', alpha=0.2)
        ax1.grid(visible=True, which='both')
        ax1.set_title('$R^{2}$ Value')
        ax1.set(xlabel='Coordinate')
        

        ax2.plot(metrics['r']['emean'],
                        linewidth=2.0)
        ax2.axhline(np.mean(metrics['r']['emean']),c='r',
                        linewidth=2.0,linestyle='solid') 
        ax2.scatter(np.nanargmin(metrics['r']['emean']),
                        np.nanmin(metrics['r']['emean']),
                        marker='o', color='red', s=120)
        ax2.annotate(f'best coord.',
                     (np.nanargmin(metrics['r']['emean']),np.nanmin(metrics['r']['emean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax2.scatter(np.nanargmax(metrics['r']['emean']),
                        np.nanmax(metrics['r']['emean']),
                        marker='o', color='red', s=120)
        ax2.annotate(f'worst coord.',
                     (np.nanargmax(metrics['r']['emean']),np.nanmax(metrics['r']['emean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax2.axhspan(np.nanpercentile(metrics['r']['emean'],25), 
                        np.nanpercentile(metrics['r']['emean'],75), 
                        facecolor='gold', alpha=0.2)
        ax2.grid(visible=True, which='both')
        ax2.set_title('$RMSE$ Value')
        ax2.set(xlabel='Coordinate')


        ax3.plot(metrics['f']['emean'],
                        linewidth=2.0)
        ax3.axhline(np.mean(metrics['f']['emean']),c='r',
                        linewidth=2.0,linestyle='solid') 
        ax3.scatter(np.nanargmin(metrics['f']['emean']),
                        np.nanmin(metrics['f']['emean']),
                        marker='o', color='red', s=120)
        ax3.annotate(f'worst coord.',
                     (np.nanargmin(metrics['f']['emean']),np.nanmin(metrics['f']['emean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax3.scatter(np.nanargmax(metrics['f']['emean']),
                        np.nanmax(metrics['f']['emean']),
                        marker='o', color='red', s=120)
        ax3.annotate(f'best coord.',
                     (np.nanargmax(metrics['f']['emean']),np.nanmax(metrics['f']['emean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax3.axhspan(np.nanpercentile(metrics['f']['emean'],25), 
                        np.nanpercentile(metrics['f']['emean'],75), 
                        facecolor='gold', alpha=0.2)
        ax3.grid(visible=True, which='both')
        ax3.set_title('$fit$ Value')
        ax3.set(xlabel='Coordinate')


        ax4.plot(metrics['nr']['emean'],
                        linewidth=2.0)
        ax4.axhline(np.mean(metrics['nr']['emean']),c='r',
                        linewidth=2.0,linestyle='solid') 
        ax4.scatter(np.nanargmin(metrics['nr']['emean']),
                        np.nanmin(metrics['nr']['emean']),
                        marker='o', color='red', s=120)
        ax4.annotate(f'best coord',
                     (np.nanargmin(metrics['nr']['emean']),np.nanmin(metrics['nr']['emean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax4.scatter(np.nanargmax(metrics['nr']['emean']),
                        np.nanmax(metrics['nr']['emean']),
                        marker='o', color='red', s=120)
        ax4.annotate(f'worst coord',
                     (np.nanargmax(metrics['nr']['emean']),np.nanmax(metrics['nr']['emean'])),
                     xytext=(-2*self.OFFSET,self.OFFSET),
                     textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
        ax4.axhspan(np.nanpercentile(metrics['nr']['emean'],25), 
                        np.nanpercentile(metrics['nr']['emean'],75), 
                        facecolor='gold', alpha=0.2)
        ax4.grid(visible=True, which='both')
        ax4.set_title('$NRMSE$ Value')
        ax4.set(xlabel='Coordinate')

        score = np.round(self.metricdict[model][test]['score'],decimals=2)

        fig.tight_layout()
        fig.suptitle(title + 
                     f'\n\nTest Name: {test}\n\n Model Name: {model}'
                     f'\n\nTest Score: {score}/10.0', y=1.15, weight='bold')
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/{model}/{test}/metrics_overcoordinates.png',
                        bbox_inches='tight')
        else:
            plt.show()

    def plotvariation_overjoints(self, modelname=None, testname=None, save=False, 
                                 title='Cumulative Metric Variation Projected on All Coordinates'):
        """
        Plots the variation of metrics over a single test on a single model over the available 
        coordinates.

        Parameters
        ---
            modelname (dict): metric dictionary of a single test under a single model
            save (bool): if True, save the model under figpath (/plots/{data_name})
            title (string): title variability
        """
        model = self.modelname if modelname is None else modelname
        test = self.testname if testname is None else testname
        metrics = self.metricdict[model][test]['metrics']
        m1list = [metrics['r']['valuec'],metrics['r2']['valuec'],
                    metrics['nr']['valuec'],metrics['f']['valuec']]
        
        fig1 = plt.figure(figsize = (20,20))
        gs1 = matplotlib.gridspec.GridSpec(4,1, figure=fig1)
        for j in range(4):
            ax1 = fig1.add_subplot(gs1[j])
            ax1.boxplot(m1list[j],patch_artist = True, 
                            boxprops = dict(facecolor = "lightblue"), 
                            medianprops = dict(color = "green", linewidth = 1.5), 
                            whiskerprops = dict(color = "red", linewidth = 2), 
            )
            ax1.set_title(f'${self.LABELID[j]}$')
            ax1.grid(alpha=0.5)
            ax1.set_xlabel('Coordinates')

        score = np.round(self.metricdict[model][test]['score'],decimals=2)

        plt.tight_layout()
        fig1.suptitle(title + 
                     f'\n\nTest Name: {test}\n\n Model Name: {model}'
                     f'\n\nTest Score: {score}/10.0', y=1.15, weight='bold')
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/{model}/{test}/variation_overcoordinates.png',
                        bbox_inches='tight')
        else:
            plt.show()

    def plotvariation_tests(self, modelname=None, testnames=None, metric='f', save=False, labels=None, include_legend=True,
                            title = 'Cumulative Metric Variation for All Available Tests of a Single Model',
                            legendentries=None, omitscores=False):
        """
        Plots the test variation over all joints and all environments for a given model
        pertaining to all tests performed. The relevant distribution of a test dataset 
        related to the training dataset is incurred from a score from 1 to 10 as acquired
        from preprocessing information of a testing dataset.

        Parameters
        ---
            metric (string): chosen metric of assessment
            save (bool): if True, save the model under figpath (/plots/{data_name})
            labels (list): naming convention of tests
            include_legend (bool): if True name tests on the ploent
            legendetnries (list): if not None put legend entries into legend
            title (string): title variability
        """
        model = self.modelname if modelname is None else modelname
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
        cur = []
        scores = []
        for i, test in enumerate(self.testlist if testnames is None else testnames):
            score = str(np.round(self.metricdict[model][test]['score'],decimals=2))
            if metric=='f':
                cur.append(self.metricdict[model][test]['metrics']['f']['cmean'])
            elif metric=='r':
                cur.append(self.metricdict[model][test]['metrics']['r']['cmean'])
            elif metric=='nr':
                cur.append(self.metricdict[model][test]['metrics']['nr']['cmean'])
            elif metric=='r2':
                cur.append(self.metricdict[model][test]['metrics']['r2']['cmean'])
            elif metric=='aic':
                aic_act = self.metricdict[model][test]['metrics']['aic']['cmean']
                aic_min = np.min(aic_act)
                aic_delta = aic_act - aic_min
                cur.append(aic_delta)
            elif metric=='fpe':
                cur.append(self.metricdict[model][test]['metrics']['fpe']['cmean'])
            scores.append(score)

        bplot = axes.boxplot(cur,1,'',patch_artist = True, medianprops = dict(color = "green", 
                        linewidth = 1.5), whiskerprops = dict(color = "red", linewidth = 2),
                        labels=scores)
        ticklocs = np.arange(len(self.testlist))+1

        axes.grid(True,alpha=0.6)
        axes.set_xlabel('Test Scores [0 - 10]')
        labels = scores if labels is None else labels
        axes.set_xticks(ticklocs,labels)
        axes.set_title(f'{metric} Variation')

        for patch, color in zip(bplot['boxes'], self.COLORS):
            patch.set_facecolor(color)
        if include_legend:
            if legendentries:
                if not omitscores:
                    l = [f'{l}\nscore: {scores[i]}' for i, l in enumerate(legendentries)]
                else:
                    l = [l for i, l in enumerate(legendentries)]
            else:
                l = self.testlist
            leg = axes.legend(bplot['boxes'], l, loc='best',fancybox=True,
                        framealpha=1, shadow=True, borderpad=1, fontsize='small', title='test type')
            lframe = leg.get_frame()
            lframe.set_facecolor('#b4aeae')
            lframe.set_edgecolor('black')
            lframe.set_alpha(1)

        fig.suptitle(title + 
                     f'\n\nModel Name: {model}\n\nMetric : {metric}',
                     fontsize=25, y=1.02, weight='bold') 
            
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/{model}/variation_overtests.png',
                        bbox_inches='tight')
        else:
            plt.show()  
 
    def plotvariation_models(self, modelnames=None, metric='f', save=False, labels=None, include_legend=True,
                             title = 'Cumulative Metric Variation Over All Tests of All Models',
                             legendentries=None, savelegend=None, omitscores=False,limitaxes=False):
        """
        Plots the test variation over all joints and all environments of all the models
        pertaining to all tests performed. The relevant distribution of a test dataset 
        related to the training dataset is incurred from a score from 1 to 10 as acquired
        from preprocessing information of a testing dataset. Assessment is only through a single
        chosen metric.

        Parameters
        ---
            modelnames (string): chosen metric of assessment
            save (bool): if True, save the model under figpath (/plots/{data_name})
            labels (list): naming convention of tests
            include_legend (bool): if True name tests on the ploent
            legendetnries (list): if not None put legend entries into legend
            title (string): title variability
        """
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
        for i, nmodel in enumerate(self.modellist if modelnames is None else modelnames):
            cur = []
            scores = []
            for j, test in enumerate(self.testlist):
                score = str(np.round(self.metricdict[nmodel][test]['score'],decimals=2))
                if metric=='f':
                    cur.append(self.metricdict[nmodel][test]['metrics']['f']['cmean'])
                elif metric=='r':
                    cur.append(self.metricdict[nmodel][test]['metrics']['r']['cmean'])
                elif metric=='nr':
                    cur.append(self.metricdict[nmodel][test]['metrics']['nr']['cmean'])
                elif metric=='r2':
                    cur.append(self.metricdict[nmodel][test]['metrics']['r2']['cmean'])
                elif metric=='aic':
                    aic_act = self.metricdict[nmodel][test]['metrics']['aic']['cmean']
                    aic_min = np.min(aic_act)
                    aic_delta = aic_act - aic_min
                    cur.append(aic_delta)
                elif metric=='fpe':
                    cur.append(self.metricdict[nmodel][test]['metrics']['fpe']['cmean'])
                scores.append(score)

            bplot = axes.boxplot(cur,1,'',patch_artist = True, medianprops = dict(color = "green", 
                            linewidth = 1.5), whiskerprops = dict(color = "red", linewidth = 2),
                            positions=(np.arange(len(self.testlist)) + i*(len(self.testlist)+1)),
                            widths=0.5)
            ticklocs = np.arange(len(self.modellist))*(len(self.testlist)+1) + np.mean(np.arange(len(self.testlist)))

            for pos, (patch, color) in enumerate(zip(bplot['boxes'], self.COLORS)):
                patch.set_facecolor(color)
                #axes.text(x=ticklocs[i],y=patch.get_ydata()[2],
                #          s=scores[i]) 
            axes.set_xlabel('Model Type')
            axes.set_title(f'{metric} Variation')

            labels = self.modellist if labels is None else labels
            axes.set_xticks(ticklocs,labels)
            axes.grid(True,alpha=0.6)

        if include_legend:
            if legendentries:
                if not omitscores:
                    l = [f'{l}\nscore: {scores[i]}' for i, l in enumerate(legendentries)]
                else:
                    l = [l for i, l in enumerate(legendentries)]
            else:
                l = self.testlist
            leg = axes.legend(bplot['boxes'], l, loc='best',fancybox=True, 
                             framealpha=1, shadow=True, borderpad=1, fontsize='small', title='test type')
            lframe = leg.get_frame()
            lframe.set_facecolor('#b4aeae')
            lframe.set_edgecolor('black')
            lframe.set_alpha(1)

            if savelegend:
                leg.savefig('modellegend.png')

        fig.suptitle(title +
                     f'\n\nMetric : {metric}',
                     fontsize=25, y=1.02, weight='bold') 
        if limitaxes:
            axes.set_ylim(bottom=-3,top=1)
        plt.setp(axes.get_xticklabels(), rotation=0, horizontalalignment='center')    
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/variation_overmodels.png',
                        bbox_inches='tight')
        else:
            plt.show()

    def plotvariation_architectures(self,archnames=None, modelnames=None, metric='f', save=False, labels=None, include_legend=True,
                                    title='Cumulative Metric Variation of All Architectures of Chosen Datasets over Chosen Tests',
                                    legend_entries=None, savelegend=None):
        pass  
    
    def plotcomparison_contexts(self, modelnames, ctx, iter, save=False, of='best',
                                title='Horizon Prediction Comparison Between Various Models'):
        """
        Plots the horizon prediction comparison of a choice of models over a single test for contextual
        comparisons.

        Parameters
        ---
            modelnames (list): list of modelnames included in the comparison of ctxs
            of (string): best-worst-median-random plots of sim2sim
            dim (int): 
            ctx (int):
            iter (int):
            save (bool): if True, save the model under figpath (/plots/{data_name})
            title (string): title variability
        """
        fig, ax = plt.subplots(7, 2, figsize=(20, 20))

        for model in modelnames:
            if of=='best':
                idx1 = np.argmax(self.metricdict[model][self.testname]['metrics']['r2']['cmean'])
            elif of=='worst':
                idx1 = np.argmin(self.metricdict[model][self.testname]['metrics']['r2']['cmean'])
            elif of=='median':
                idx1 = np.argsort(self.metricdict[model][self.testname]['metrics']['r2']['cmean'])[len(self.metricdict[model][self.testname]['metrics']['r2']['cmean'])//2]
            elif of=='random':
                idx1 = np.random.random_integers(low = 0, high=len(self.metricdict[model][self.testname]['metrics']['r2']['cmean']))
            else:
                raise ValueError
            
            sim = self.metricdict[model][self.testname]['ysim'][idx1,:,:]
            err = self.metricdict[model][self.testname]['yerr'][idx1,:,:]
            true = self.metricdict[model][self.testname]['ytrue'][idx1,:,:]
            ctx = self.metricdict[model][self.testname]['yctx'][idx1,:,:]
            total = np.concatenate([ctx,true])

            t_context = np.arange(1, ctx)
            t_prediction = np.arange(1, iter-ctx+1) + ctx
            t_total = np.arange(0, iter)

            k = 0
            for j in range(2):
                for i in range(7):
                    ax[i,j].axvline(x = t_prediction[-1], 
                        color = 'k', linestyle='--')
                    ax[i,j].axvline(x = t_context[-1], 
                        color = 'k', linestyle='--')
                    ax[i,j].axvspan(t_context[0], t_context[-1],
                        facecolor='lime', alpha=0.2)
                    ax[i,j].plot(t_total,total[:,k] ,'k', 
                        label=self.LABELCOORDS4D[k],linewidth=2)
                    ax[i,j].plot(t_prediction,sim[:,k], 'b', 
                        label=self.LABELPRED4D[k],linewidth=2)
                    ax[i,j].grid(True)
                    ax[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax[i,j].set_xlabel('Iterations')
                    k=k+1

        fig.suptitle(title + 
                    f'\nTest Name: {self.testname}\n',size = 20,y = .99)
        fig.tight_layout( pad = 2)
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/context_comparison.png',
                        bbox_inches='tight')
        else:
            plt.show()

    def plotcomparison_horizon(self, modelnames, testname, of, dim, ctx, iter,  save=False, 
                            title='Horizon Prediction Comparison Between Two Different Models'):
        """
        Plots the horizon prediction comparison of two different models over a single test. Possible use
        cases are finetuned, oneshot and fewshot model comparisons.

        Parameters
        ---
            true (np.array): true horizon
            sim1 (np.array): 1st simulation over a single horizon
            err1 (np.array): 1st error over a single horizon
            sim2 (np.array): 2nd simulation over a single horizon
            err2 (np.array): 2nd error over a single horizon
            of (string): best-worst-median-random plots of sim2sim
            dim (int): 
            ctx (int):
            iter (int):
            save (bool): if True, save the model under figpath (/plots/{data_name})
            title (string): title variability
        """
        modelname1 = modelnames[0]
        modelname2 = modelnames[1]
        if of=='best':
            idx1 = np.argmax(self.metricdict[modelname1][testname]['metrics']['r2']['cmean'])
            idx2 = np.argmax(self.metricdict[modelname2][testname]['metrics']['r2']['cmean'])
        elif of=='worst':
            idx1 = np.argmin(self.metricdict[modelname1][testname]['metrics']['r2']['cmean'])
            idx2 = np.argmin(self.metricdict[modelname2][testname]['metrics']['r2']['cmean'])
        elif of=='median':
            idx1 = np.argsort(self.metricdict[modelname1][testname]['metrics']['r2']['cmean'])[len(self.metricdict[modelname1][testname]['metrics']['r2']['cmean'])//2]
            idx2 = np.argsort(self.metricdict[modelname2][testname]['metrics']['r2']['cmean'])[len(self.metricdict[modelname2][testname]['metrics']['r2']['cmean'])//2]
        elif of=='random':
            idx1 = np.random.random_integers(low = 0, high=len(self.metricdict[modelname1][testname]['metrics']['r2']['cmean']))
            idx2 = np.random.random_integers(low = 0, high=len(self.metricdict[modelname1][testname]['metrics']['r2']['cmean']))
        else:
            raise ValueError
        
        sim1 = self.metricdict[modelname1][testname]['ysim'][idx1,:,:]
        err1 = self.metricdict[modelname1][testname]['yerr'][idx1,:,:]
        true1 = self.metricdict[modelname1][testname]['ytrue'][idx1,:,:]
        ctx1 = self.metricdict[modelname1][testname]['yctx'][idx1,:,:]
        sim2 = self.metricdict[modelname2][testname]['ysim'][idx2,:,:]
        err2 = self.metricdict[modelname2][testname]['yerr'][idx2,:,:]
        total = np.concatenate([ctx1,true1])

        t_context = np.arange(1, ctx)
        t_prediction = np.arange(1, iter-ctx+1) + ctx
        t_total = np.arange(0, iter)

        if dim==4:
            fig, ax = plt.subplots(7, 2, figsize=(20, 20))

            k = 0
            for j in range(2):
                for i in range(7):
                    ax[i,j].axvline(x = t_prediction[-1], 
                        color = 'k', linestyle='--')
                    ax[i,j].axvline(x = t_context[-1], 
                        color = 'k', linestyle='--')
                    ax[i,j].axvspan(t_context[0], t_context[-1],
                        facecolor='lime', alpha=0.2)
                    ax[i,j].plot(t_total,total[:,k] ,'k', 
                        label=self.LABELCOORDS4D[k],linewidth=2)
                    ax[i,j].plot(t_prediction,sim1[:,k], 'b', 
                        label=self.LABELPRED4D[k],linewidth=2)
                    ax[i,j].plot(t_prediction,err1[:,k], 'r', 
                        label=self.LABELERROR4D[k])
                    ax[i,j].plot(t_prediction,sim2[:,k], 'g', 
                        label=self.LABELPRED4D[k]+'new',linewidth=2)
                    ax[i,j].plot(t_prediction,err2[:,k], 'm', 
                        label=self.LABELERROR4D[k]+'new')
                    ax[i,j].grid(True)
                    ax[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax[i,j].set_xlabel('Iterations')
                    k=k+1

        elif dim==6:
            fig = plt.figure(figsize = (20,20))
            gs = matplotlib.gridspec.GridSpec(9,2, figure=fig)

            k = 0
            for i1 in range(9):
                ax1 = fig.add_subplot(gs[i1,0])
                ax1.axvline(x = t_prediction[-1], 
                    color = 'k', linestyle='--')
                ax1.axvline(x = t_context[-1], 
                    color = 'k', linestyle='--')
                ax1.axvspan(t_context[0], t_context[-1],
                        facecolor='lime', alpha=0.2)

                ax1.plot(t_total,total[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=2)
                ax1.plot(t_prediction,sim1[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=2)
                ax1.plot(t_prediction,err1[:,k], 'r', 
                    label=self.LABELERROR6D[k])
                ax1.plot(t_prediction,sim2[:,k], 'g', 
                    label=self.LABELPRED6D[k]+'new',linewidth=2)
                ax1.plot(t_prediction,err2[:,k], 'm', 
                    label=self.LABELERROR6D[k]+'new')
                ax1.grid(True)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax1.set_xlabel('Iterations')
                k=k+1

            for i2 in range(7):
                ax2 = fig.add_subplot(gs[i2,1])
                ax2.axvline(x = t_prediction[-1], 
                    color = 'k', linestyle='--')
                ax2.axvline(x = t_context[-1], 
                    color = 'k', linestyle='--')
                ax2.axvspan(t_context[0], t_context[-1],
                        facecolor='lime', alpha=0.2)

                ax2.plot(t_total,total[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=2)
                ax2.plot(t_prediction,sim1[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=2)
                ax2.plot(t_prediction,err1[:,k], 'r', 
                    label=self.LABELERROR6D[k])
                ax2.plot(t_prediction,sim2[:,k], 'g', 
                    label=self.LABELPRED6D[k]+'new',linewidth=2)
                ax2.plot(t_prediction,err2[:,k], 'm', 
                    label=self.LABELERROR6D[k]+'new')
                ax2.grid(True)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax2.set_xlabel('Iterations')
                k=k+1

        fig.suptitle(title + 
                    f'\nTest Name: {testname}\n Model Name: {modelname1}',size = 20,y = .99)
        fig.tight_layout( pad = 2)
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/horizon_comparison.png',
                        bbox_inches='tight')
        else:
            plt.show()

    def plotcomparison_variation_overjoints(self, modelnames, testname, save=False, 
                                            title='Accuracy Variation Comparison Between Two Different Models'):
        """
        Plots the horizon prediction comparison of two different models over a single test. Possible use
        cases are finetuned, oneshot and fewshot model comparisons.

        Parameters
        ---
            modelnames (string)
            testname (string)
            save (bool): if True, save the model under figpath (/plots/{data_name})
            title (string): title variability
        """
        modelname1 = modelnames[0]
        modelname2 = modelnames[1]
        metric1 = self.metricdict[modelname1][testname]['metrics']
        metric2 = self.metricdict[modelname2][testname]['metrics']

        m1list = [metric1['r']['valuec'],metric1['r2']['valuec'],
                    metric1['nr']['valuec'],metric1['f']['valuec']]
        m2list = [metric2['r']['valuec'],metric2['r2']['valuec'],
                    metric2['nr']['valuec'],metric2['f']['valuec']]
        
        #fig1, axes1 = plt.subplots(nrows=4, ncols=1, figsize=(18,18))
        fig1 = plt.figure(figsize = (20,20))
        gs1 = matplotlib.gridspec.GridSpec(4,1, figure=fig1)
        for j in range(4):
            ax1 = fig1.add_subplot(gs1[j])
            ax1.boxplot(m1list[j],patch_artist = True, 
                            boxprops = dict(facecolor = "lightblue"), 
                            medianprops = dict(color = "green", linewidth = 1.5), 
                            whiskerprops = dict(color = "red", linewidth = 2), 
            )
            ax1.boxplot(m2list[j],patch_artist = True, 
                            boxprops = dict(facecolor = "lightblue"), 
                            medianprops = dict(color = "green", linewidth = 1.5), 
                            whiskerprops = dict(color = "red", linewidth = 2), 
            )
            ax1.set_title(f'${self.LABELID[j]}$')
            ax1.grid(alpha=0.5)

        fig1.suptitle(title + 
                    f'\nTest Name: {testname}\n Model Name: {modelname1}')
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/variation_comparison.png',
                        bbox_inches='tight')
        else:
            plt.show()

    def plotcomparison_metrics_overjoints(self, modelnames, testname, save=False, 
                                        title=''):
        """
        Plots the horizon prediction comparison of two different models over a single test. Possible use
        cases are finetuned, oneshot and fewshot model comparisons.

        Parameters
        ---
            metric1 (dict): metric dict of model 1
            metric2 (dict): metric dict of model 2
            save (bool): if True, save the model under figpath (/plots/{data_name})
            title (string): title variability
        """
        modelname1 = modelnames[0]
        modelname2 = modelnames[1]
        metric1 = self.metricdict[modelname1][testname]['metrics']
        metric2 = self.metricdict[modelname2][testname]['metrics']

        fig,[[ax1, ax2],[ax3, ax4]]  = plt.subplots(2, 2,figsize=(15, 15), dpi=200)

        ax1.plot(metric1['r2']['emean'],
                        linewidth=1.5)
        ax1.plot(metric2['r2']['emean'],
                        linewidth=1.5)
        ax1.plot(np.mean(metric1['r2']['emean']),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax1.scatter(np.nanargmin(metric1['r2']['emean']),
                        np.nanmin(metric1['r2']['emean']),
                        marker='o', color='red', s=30)
        ax1.scatter(np.nanargmax(metric1['r2']['emean']),
                        np.nanmax(metric1['r2']['emean']),
                        marker='o', color='red', s=30)
        ax1.axhspan(np.nanpercentile(metric1['r2']['emean'],25), 
                        np.nanpercentile(metric1['r2']['emean'],75), 
                        facecolor='yellow', alpha=0.2)
        ax1.grid(visible=True, which='both')
        ax1.set_title('$R^{2}$ Value')
        ax1.set(xlabel='Environment Index')
        

        ax2.plot(metric1['r']['emean'],
                        linewidth=1.5)
        ax2.plot(metric2['r']['emean'],
                        linewidth=1.5)
        ax2.plot(np.mean(metric1['r']['emean']),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax2.scatter(np.nanargmin(metric1['r']['emean']),
                        np.nanmin(metric1['r']['emean']),
                        marker='o', color='red', s=30)
        ax2.scatter(np.nanargmax(metric1['r']['emean']),
                        np.nanmax(metric1['r']['emean']),
                        marker='o', color='red', s=30)
        ax2.axhspan(np.nanpercentile(metric1['r']['emean'],25), 
                        np.nanpercentile(metric1['r']['emean'],75), 
                        facecolor='yellow', alpha=0.2)
        ax2.grid(visible=True, which='both')
        ax2.set_title('$RMSE$ Value')
        ax2.set(xlabel='Environment Index')


        ax3.plot(metric1['f']['emean'],
                        linewidth=1.5)
        ax3.plot(metric2['f']['emean'],
                        linewidth=1.5)
        ax3.plot(np.mean(metric1['f']['emean']),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax3.scatter(np.nanargmin(metric1['f']['emean']),
                        np.nanmin(metric1['f']['emean']),
                        marker='o', color='red', s=30)
        ax3.scatter(np.nanargmax(metric1['f']['emean']),
                        np.nanmax(metric1['f']['emean']),
                        marker='o', color='red', s=30)
        ax3.axhspan(np.nanpercentile(metric1['f']['emean'],25), 
                        np.nanpercentile(metric1['f']['emean'],75), 
                        facecolor='yellow', alpha=0.2)
        ax3.grid(visible=True, which='both')
        ax3.set_title('$fit$ Value')
        ax3.set(xlabel='Environment Index')


        ax4.plot(metric1['nr']['emean'],
                        linewidth=1.5)
        ax4.plot(metric2['nr']['emean'],
                        linewidth=1.5)
        ax4.plot(np.mean(metric1['nr']['emean']),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax4.scatter(np.nanargmin(metric1['nr']['emean']),
                        np.nanmin(metric1['nr']['emean']),
                        marker='o', color='red', s=30)
        ax4.scatter(np.nanargmax(metric1['nr']['emean']),
                        np.nanmax(metric1['nr']['emean']),
                        marker='o', color='red', s=30)
        ax4.axhspan(np.nanpercentile(metric1['nr']['emean'],25), 
                        np.nanpercentile(metric1['nr']['emean'],75), 
                        facecolor='yellow', alpha=0.2)
        ax4.grid(visible=True, which='both')
        ax4.set_title('$NRMSE$ Value')
        ax4.set(xlabel='Environment Index')

        fig.tight_layout(pad = 1.2 )
        fig.suptitle(title + 
                    f'\nTest Name: {testname}\n Model Name: {modelname1}',size = 20,y = .99)
        if save:
            plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/metrics_comparison.png',
                        bbox_inches='tight')
        else:
            plt.show()
    
    def plotmetric(self, modelname=None, testname=None, metric='f', jointmean=True, envmean=True, save=False):
        """
        Applies a chosen metric according to specified mean parameters to a batch of simulations.
        if jointmean=False, envmean=False:
            INVALID
        if jointmean=True, envmean=False:
            mean over all joints projected onto environments space
        if jointmean=False, envmean=True:
            mean over all environments projected onto coordinate space
        if jointmean=True, envmean=True:
            mean metrics over all dimensions

        Parameters:
            metrics: metric list of a test
            metric: chosen metric to be assessed
            jointmean=False:
            envmean=False:
            save=False: if True, save the model under figpath (/plots/{data_name})
        """
        model = self.modelname if modelname is None else modelname
        test = self.testname if testname is None else testname
        metrics = self.metricdict[model][test]['metrics']

        if not jointmean and not envmean:
            print('Impossible to parse without projecting onto coordinate space or environment space')
            return
        
        elif jointmean and not envmean:
            if metric=='f':
                c = metrics['f']['cmean']
            elif metric=='r':
                c = metrics['r']['cmean']
            elif metric=='nr':
                c = metrics['nr']['cmean']
            elif metric=='r2':
                c = metrics['r2']['cmean']
            elif metric=='aic':
                aic_act = metrics['aic']['cmean']
                aic_min = np.nanmin(aic_act)
                c = aic_act - aic_min
            elif metric=='fpe':
                c = metrics['fpe']['cmean']

            fig, ax1 = plt.subplots(1, figsize=(15,15), dpi=200)

            ax1.plot(c,
                            linewidth=2)
            ax1.axhline(np.mean(c),c='r',
                            linewidth=2.0,linestyle='solid') 
            ax1.scatter(np.nanargmin(c),
                            np.nanmin(c),
                            marker='o', color='red', s=120)
            ax1.annotate(f'worst env.' if metric=='fpe' else f'best env.',
                        (np.nanargmin(c),np.nanmin(c)),
                        xytext=(-2*self.OFFSET,self.OFFSET),
                        textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
            ax1.scatter(np.nanargmax(c),
                            np.nanmax(c),
                            marker='o', color='red', s=120)
            ax1.annotate(f'best env.' if metric=='fpe' else f'worst env.',
                        (np.nanargmax(c),np.nanmax(c)),
                        xytext=(-2*self.OFFSET,self.OFFSET),
                        textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
            ax1.axhspan(np.nanpercentile(c,25), 
                            np.nanpercentile(c,75), 
                            facecolor='gold', alpha=0.2)
            ax1.grid(visible=True, which='both')
            ax1.set_title(f'{metric} Value')
            ax1.set(xlabel='Environment')

            score = np.round(self.metricdict[model][test]['score'],decimals=2)

            fig.tight_layout()
            fig.suptitle(f'Cumulative Accuracy Projected on Environments'
                     f'\n\nTest Name: {test}\n\n Model Name: {model}'
                     f'\n\nTest Score: {score}/10.0', y=1.25, weight='bold')
            if save:
                plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/{model}/{test}/{metric}_overenvironments.png',
                        bbox_inches='tight')
            else:
                plt.show()

        elif envmean and not jointmean:
            if metric=='f':
                c = metrics['f']['emean'].T
            elif metric=='r':
                c = metrics['r']['emean'].T
            elif metric=='nr':
                c = metrics['nr']['emean'].T
            elif metric=='r2':
                c = metrics['r2']['emean'].T
            elif metric=='aic':
                aic_act = metrics['aic']['emean'].T
                aic_min = np.min(aic_act)
                c = aic_act - aic_min
            elif metric=='fpe':
                c = metrics['fpe']['emean'].T

            fig, ax1 = plt.subplots(1, figsize=(15,15), dpi=200)

            ax1.plot(c,
                            linewidth=2)
            ax1.axhline(np.mean(c),c='r',
                            linewidth=2.0,linestyle='solid') 
            ax1.scatter(np.nanargmin(c),
                            np.nanmin(c),
                            marker='o', color='red', s=120)
            ax1.annotate(f'worst coord.' if metric=='fpe' else f'best coord.',
                        (np.nanargmin(c),np.nanmin(c)),
                        xytext=(-2*self.OFFSET,self.OFFSET),
                        textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
            ax1.scatter(np.nanargmax(c),
                            np.nanmax(c),
                            marker='o', color='red', s=120)
            ax1.annotate(f'best coord.' if metric=='fpe' else f'worst coord.',
                        (np.nanargmax(c),np.nanmax(c)),
                        xytext=(-2*self.OFFSET,self.OFFSET),
                        textcoords='offset points', bbox=self.BBOX, arrowprops=self.ARROWPROPS)
            ax1.axhspan(np.nanpercentile(c,25), 
                            np.nanpercentile(c,75), 
                            facecolor='gold', alpha=0.2)
            ax1.grid(visible=True, which='both')
            ax1.set_title(f'{metric} Value')
            ax1.set(xlabel='Coordinate')    

            score = np.round(self.metricdict[model][test]['score'],decimals=2)

            fig.tight_layout()
            fig.suptitle(f'Cumulative Accuracy Projected on Coordinates'
                     f'\n\nTest Name: {test}\n\n Model Name: {model}'
                     f'\n\nTest Score: {score}/10.0', y=1.25, weight='bold')
            if save:
                plt.savefig(f'{self.figpath}/{self.dataname}/{self.savename}/{model}/{test}/{metric}_overcoordinates.png',
                        bbox_inches='tight')
            else:
                plt.show()
            
        elif jointmean and envmean:
            r = np.mean(metrics['r']['cmean']), np.mean(metrics['r']['emean'])
            r2 = np.mean(metrics['r2']['cmean']), np.mean(metrics['r2']['emean'])
            f = np.mean(metrics['f']['cmean']), np.mean(metrics['f']['emean'])
            nr = np.mean(metrics['nr']['cmean']), np.mean(metrics['nr']['emean'])
            naic = np.mean(metrics['aic']['cmean']), np.mean(metrics['aic']['emean'])
            fpe = np.mean(metrics['fpe']['cmean']), np.mean(metrics['fpe']['emean'])

            print(f'RMSE(env, coord): {r}\nr^2(env, coord): {r2}\nNRMSE(env, coord): {nr}'
                  f'\nfit index(env, coord): {f}\nnaic(env, coord): {naic}\nfpe(env, coord): {fpe}\n\n\n')
        
    def zeroshot_horizon(self, model, test, of, dim, ctx=200, iter=1000, save=False,
                         title='ZeroShot Horizon Prediction'):
        """
        Alias for plotting horizon over a single environment.
        
        Zero shot config:
            ctx=200
            iter=1000
        """
        return self.plothorizon(modelname=model, testname=test, of=of, dim=dim, ctx=ctx, iter=iter, save=save, title=title)

    def zeroshot_metrics_overjoints(self, model, test, save=False, 
                                  title='Cumulative ZeroShot Accuracy Metrics Projected on All Environments'):
        """
        Alias for jointmean plot of available metrics. 
        
        Zero shot config:
            ctx=200
            iter=1000
        """
        return self.plotmetrics_overjoints(modelname=model, testname=test, save=save, title=title)

    def zeroshot_variation_overjoints(self, model, test, save=False, 
                                  title='Cumulative ZeroShot Metrics Variation Projected on All Environments'):
        """
        Alias for jointmean plot of available metrics. 
        
        Zero shot config:
            ctx=200
            iter=1000
        """
        return self.plotvariation_overjoints(modelname=model, testname=test, save=save, title=title)

    def oneshot_horizon(self, model, test, of, dim, ctx=1200, iter=2000, save=False, 
                        title='OneShot Horizon Prediction'):
        """
        Alias for plotting horizon over a single environment. 
        
        One shot config:
            ctx=1200
            iter=2000
        """
        return self.plothorizon(modelname=model, testname=test, of=of, dim=dim, ctx=ctx, iter=iter, save=save, title=title)
    
    def oneshot_horizon_comparison(self, models, test, of, dim, ctx=1200, iter=2000, save=False,
                                   title='OneShot Horizon Comparison'):
        """
        Alias for plotting horizon comparison over a single environment. 
        
        One shot config:
            ctx=1200
            iter=2000
        """
        return self.plotcomparison_horizon(modelnames=models, testname=test, of=of, dim=dim, ctx=ctx, iter=iter, save=save, title=title)

    def oneshot_metrics_overjoints(self, model, test, save=False, 
                                 title='Cumulative OneShot Accuracy Metrics Projected on All Environments'):
        """
         Alias for jointmean plot over a single environment. 
         
         One shot config:
            ctx=1200
            iter=2000
        """
        return self.plotmetrics_overjoints(modelname=model, testname=test, save=save, title=title)

    def oneshot_variation_overjoints(self, model, test, save=False, 
                                 title='Cumulative OneShot Metrics Variation Projected on All Environments'):
        """
         Alias for jointmean plot over a single environment. 
         
         One shot config:
            ctx=1200
            iter=2000
        """
        return self.plotvariation_overjoints(modelname=model, testname=test, save=save, title=title)

    def fewshot_horizon(self, model, test, of, dim, ctx=5200, iter=6000, save=False, 
                        title='FewShot Horizon Prediction'):
        """
        Alias for plotting horizon over a single environment. 
        
        Few shot config (n=shot):
            ctx = n*1000 + 200
            iter = (n+1)*1000
        """
        return self.plothorizon(modelname=model, testname=test, of=of, dim=dim, ctx=ctx, iter=iter, save=save, title=title)

    def fewshot_horizon_comparison(self, models, test, of, dim, ctx=5200, iter=6000, save=False,
                                   title='FewShot Horizon Comparison'):
        """
        Alias for plotting horizon comparison over a single environment. 
        
        Few shot config (n=shot):
            ctx = n*1000 + 200
            iter = (n+1)*1000
        """
        return self.plotcomparison_horizon(modelnames=models, testname=test, of=of, dim=dim, ctx=ctx, iter=iter, save=save, title=title)

    def fewshot_metrics_overjoints(self, model, test, save=False, 
                                 title='Cumulative FewShot Accuracy Metrics Projected on All Environments'):
        """
        Alias for jointmean plot over a single environment. 
        
        Few shot config (n=shot):
            ctx = n*1000 + 200
            iter = (n+1)*1000
        """
        return self.plotmetrics_overjoints(modelname=model, testname=test, save=save, title=title)
    
    def fewshot_variation_overjoints(self, model, test, save=False, 
                                 title='Cumulative FewShot Metrics Variation Projected on All Environments'):
        """
        Alias for jointmean plot over a single environment. 
        
        Few shot config (n=shot):
            ctx = n*1000 + 200
            iter = (n+1)*1000
        """
        return self.plotvariation_overjoints(modelname=model, testname=test, save=save, title=title)

    def plotfinetune_horizon(self, model, test, of='median', dim=4, save=False, 
                             title='FineTuned Model Horizon Prediction'):
        """
        Alias for horizon plot of a finetuned model.
        """
        return self.plothorizon(modelname=model, testname=test, of=of, dim=dim, save=save, title=title)

    def plotfinetune_variation_overjoints(self, model, test, save=False,
                             title='Cumulative FineTune Metrics Variation Projected on All Environments'):
        """
        Alias for joint accuracy variation of a finetuned model.
        """
        return self.plotvariation_overjoints(modelname=model, testname=test, save=save, title=title)

    def plotfinetune_metrics_overjoints(self, model, test, save=False,
                             title='Cumulative FineTune Accuracy Metrics Projected on All Environments'):
        """
        Alias for accuracy of a finetuned model.
        """
        return self.plotmetrics_overjoints(modelname=model, testname=test, save=save, title=title)
    
    def plotfinetune_horizon_comparison(self, models, test, of='median', dim=4, save=False, ctx=200, iter=1000,
                             title='FineTuned Model Horizon Prediction Comparison'):
        """
        Alias for horizon prediction comparison of a finetuned and pretrained model.
        """
        return self.plotcomparison_horizon(modelnames=models, testname=test, of=of, dim=dim, save=save, title=title, ctx=ctx, iter=iter)

    def plotfinetune_variation_overjoints_comparison(self, models, test, save=False,
                             title='Cumulative FineTune Metrics Variation Comparison'):
        """
        Alias for joint accuracy variation comparison of a finetuned and pretrained model.
        """
        return self.plotcomparison_variation_overjoints(modelnames=models, testname=test, save=save, title=title)

    def plotfinetune_metrics_overjoints_comparison(self, models, test, save=False,
                             title='Cumulative FineTune Accuracy Metrics Comparsion'):
        """
        Alias for joint accuracy comparison of a finetuned and pretrained model.
        """
        return self.plotcomparison_metrics_overjoints(modelnames=models, testname=test, save=save, title=title)
    
