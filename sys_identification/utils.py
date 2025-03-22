"""
EXPLAIN MODULE
check methods
fix plots
add logs

loss plateau
optimizer hyperparameters
model hyperparameters
training time **
"""
import matplotlib.gridspec
import torch
import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from torch.utils.tensorboard import SummaryWriter
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy import interpolate
from scipy.ndimage import uniform_filter1d

from pathlib import Path
import argparse

from architectures.transformer.transformer_sim import Config, TSTransformer
from metrics import *
from datasets import dataset

class arguments():
    """
    Custom parser, a merged script of older training programs, model hyper-parameters,
    training parameters and testing parameters are obtained through the parser - refer
    to README.md
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
        Possible arguments are: COULD BE DEPRECATED
        --
        """
        self.parser.add_argument('-init','--init-type', type=str, default='scratch', choices=['scratch',
                                                                                              'resume',
                                                                                              'pretrained'],
                            help='init from (scratch|resume|pretrained)')
        self.parser.add_argument('-dn','--data-name', type=str, default='MG1_rep',
                            help='name of the file in which the models are to be saved')
        self.parser.add_argument('-mn','--model-name', type=str, default='',
                            help='name of the model in which the models are to be saved')
        self.parser.add_argument('-dc','--disable_cuda', action='store_true',
                            help="tensor process device")   
        self.parser.add_argument('-im','--include-mass-vectors', action='store_true',
                            help="includes mass vectors in training")     
        self.parser.add_argument('-id','--include-control-diffs', action='store_true',
                            help="include control derivatives in training")
        self.parser.add_argument('-nj','--num-of-joints', type=int, default=7,
                            help="umber of joints of interest of the dataset")   
        self.parser.add_argument('-nc','--num-of-coordinates', type=int, default=14,
                            help="number of coordinates of interest of the dataset")
        self.parser.add_argument('-tsi','--total-sim-iterations', type=int, default=1000,
                            help="number of total simulation iterations of the dataset")                     
        
        self.parser.add_argument('-trb','--training-batch-size',type=int,default=8,
                            help='batch size for training data')
        self.parser.add_argument('-vlb','--validation-batch-size',type=int,default=8,
                            help='batch size for validation data')
        self.parser.add_argument("-lf",'--loss-function', type=str, default='MSE', choices=["MAE",
                                                                                         "MSE",
                                                                                         "Huber"],
                            help="loss function: 'MAE'-'MSE'-'Huber'")
        
        self.parser.add_argument("-evint",'--eval-interval', type=int, default=50,
                            help='evaluation interval')
        self.parser.add_argument("-evitr",'--validate-at', type=int, default=50,
                            help='evaluation iteraration')
        self.parser.add_argument("-il",'--log-at', type=int, default=50,
                            help='log to wandb at iteration num')

        self.parser.add_argument("-lr",'--learning-rate', type=float, default=5e-3,
                            help='learning rate')
        self.parser.add_argument("-exp",'--exponential-decay', action='store_true',
                            help='learning rate') 
        self.parser.add_argument("-stp",'--step-decay', action='store_true',
                            help='learning rate') 
        self.parser.add_argument("-cos",'--cosine-annealing', action='store_true',
                            help='learning rate')         
        self.parser.add_argument("-wd",'--weight-decay', type=float, default=0.0,
                            help='weight decay')
        self.parser.add_argument("-b1",'--beta1', type=float, default=.9,
                            help="loss function ext 1: 'MAE'-'MSE'-'Huber'")
        self.parser.add_argument("-b2",'--beta2', type=float, default=.95,
                            help="loss function ext 2: 'MAE'-'MSE'-'Huber'")

        self.parser.add_argument("-mi",'--max-iters', type=int, default=1_000_000,
                            help='number of max iterations')
        self.parser.add_argument('-s','--seed', default=False,
                            help='seed for random number generation')
        self.parser.add_argument('-wandb','--log-wandb', action='store_true', 
                            help='Live log')


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
        

        parserd = self.subparser.add_parser("diffusion", help="diffusion doc")
        parserd.add_argument('-nl','--n-layer', type=int, default=12,
                            help='number of transformer layers')
        parserd.add_argument('-nb','--bias', action='store_false', default=True,
                    help='disable nn biases')
        
        parserm = self.subparser.add_parser("meta", help="metalearner doc")
        parserm.add_argument('-nl','--n-layer', type=int, default=12,
                            help='number of transformer layers')
        parserm.add_argument('-nb','--bias', action='store_false', default=True,
                    help='disable nn biases')
        
        parsertest = self.subparser.add_parser('testing', help='test doc')
        parsertest.add_argument('--skip', type=int, default=0, metavar='num',
                    help='number of timesteps to skip')
        parsertest.add_argument('--test-folder', type=str, default="train", metavar='test',
                    help='test-folder')
        parsertest.add_argument('--figure-folder', type=str, default="fig_prova", metavar='figure',
                    help='Figure output')
        parsertest.add_argument('--model-folder', type=str, default="out_ds_big", metavar='model',
                    help='Figure output')
        parsertest.add_argument('--save-figure', action='store_true', default=True,
                    help='Figure output')
        parsertest.add_argument('--plot-predictions-and-metrics', action='store_true', default=True,
                    help='Figure output')
        parsertest.add_argument('--num-test', type=int, default=1000, metavar='num',
                    help='number of wanted test trajectories for each file')
        

        args = self.parser.parse_args()
        return args

class preprocess(dataset):
    """
    Preprocessing class used to manipulate created models and datasets. Current use is only
    in testing. 
    """
    def __init__(self, args):
        super().__init__(args)

        self.yctx = torch.empty(size=(1,self.sqlctx,self.args.num_of_coordinates),
                            device=self.device)
        self.ytrue = torch.empty(size=(1,self.sqlpar,self.args.num_of_coordinates),
                            device=self.device)
        self.ysim = torch.empty(size=(1,self.sqlpar,self.args.num_of_coordinates),
                            device=self.device)
        self.err = torch.empty(size=(1,self.sqlpar,self.args.num_of_coordinates),
                            device=self.device)

        self.metricdict = {}
        self.currentdict = {}
        self.accusation = {}

    def __str__(self):
        print("Preprocess Object Instantiated")

    
    def settime(self, time):
        return super().settime(time)
    
    def gettime(self, time):
        return super().gettime(time)
    

    def getmetadata(self):
        return super().getmetadata()
    
    def setmetadata(self, 
                    datadict_):
        return super().setmetadata(datadict_)
    

    def getdataset(self):
        return super().getdataset()
    
    def setdataset(self, training_dataset_=None, validation_dataset_=None, test_dataset_=None):
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
    

    def resolve_datasets(self, eval=True):
        return super().resolve_datasets(eval)
    
    def initialize_model(self, modelname=True):
        return super().initialize_model(modelname)
    
    def configure_dataset(self, eval=True):
        return super().configure_dataset(eval)
    
    def check_distribution(self, train, test):
        """
        Checks the distribution of a training dataset that a given model is trained on with
        the distribution of a test dataset, the return value is a distribution score which
        assesses the closeness of each dataset
        """
        
        trainstr = [str.split(x, '_') for x in train]
        teststr = str.split(test, '_')
        compare = np.array([x==y for tr in trainstr for x,y in zip(tr,teststr)])

        score = (compare==True).sum()/len(compare)*10
        scoreinterp = 'out of distribution' if score<=5 else 'in distribution'

        print(f'Train/Test Dataset Match Score: {score}\nCurrent dataset is {scoreinterp}')
        return score
    
    def cast2original(self, yctx, ytrue, ysim, yerr=None):
        """
        Casts single position pairs into the original deconstructed dataset
        """
        self.yctx = torch.cat((self.yctx,yctx),dim=0)

        self.ytrue = torch.cat((self.ytrue,ytrue),dim=0)

        self.ysim = torch.cat((self.ysim,ysim),dim=0)

        if yerr!=None:
            self.err = torch.cat((self.err,yerr),dim=0)

    def cast2dict(self, rmse, nrmse, r2, fit, aic, fpe, modelname, testname):
        """
        Casts metric and model information of a single test into a suitable data array, useful
        for postprocessing
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
        RMSE:
        NRMSE:
        R2:
        FIT IDX:
        Norm-AIC:
        FPE:
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
        Conditions tensors for postprocessing
        """
        self.yctx = self.yctx.to("cpu").detach().numpy()
        self.ytrue = self.ytrue.to("cpu").detach().numpy()
        self.ysim = self.ysim.to("cpu").detach().numpy() 
        self.err = self.err.to("cpu").detach().numpy()
    
    def test(self, method=None):
        """
        Tests a given test dataset on a given model to produce a metric tensor which is also 
        further elaborated statistically according to its joint mean and variance as well as
        environment mean and variance
        """
        testmetric = self.parse_metrics(method)

        ry = testmetric(self.ytrue, self.ysim, time_axis=1)

        rync = np.array([np.nan_to_num(col, np.mean(col)) for col in ry.T]).T
        ry_mc = np.nanmean(rync,axis=1) 
        ry_stdc = np.nanstd(rync,axis=1)

        ryne = np.array([np.nan_to_num(row, np.mean(row)) for row in ry]).T
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

    def load(self, data, eval=False):
        return super().load(data, eval)
    
    def normalizeststd(self, x):
        return super().normalizestd(x)
    
    def denormalizestd(self, nx, xmean, xstd):
        return super().denormalizestd(nx, xmean, xstd)

    def normalizelin(self, x):
        return super().normalizelin(x)
    
    def denormalizelin(self, nx, xmin, xmax):
        return super().denormalizelin(nx, xmin, xmax)
    
    def seperate_context(self, x):
        return super().seperate_context(x)

class plotcfg():
    
    CTX = 200
    ITER = 1000

    FONT = {'family' : 'DejaVu Sans',
                        'weight' : 'normal',
                        'size'   : 25}
    FONTDIM = 15

    LABELCOORDS4D = ["$x$","$y$","$z$" ,"$X$","$Y$","$Z$" ,"$W$",
                        "q0","q1","q2","q3","q4","q5","q6"]
    LABELCOORDS6D = ["$x$","$y$","$z$" ,"$d11$","$d12$","$d13$","$d22$","$d23$" ,"$d33$",
                        "q0","q1","q2","q3","q4","q5","q6"]

    LABELPRED4D = ["$\hat x$","$\hat y$","$\hat z$","$\hat X$","$ \hat Y$","$\hat Z$" ,"$ \hat W$"
                        ,"$ \hat q0$","$ \hat q1$","$ \hat q2$","$ \hat q3$","$ \hat q4$","$ \hat q5$","$ \hat q6$"]
    LABELPRED6D = ["$\hat x$","$\hat y$","$\hat z$","$\hat d11$","$ \hat d12$","$\hat d13$" ,"$ \hat d22$"
                        ,"$\hat d23$" ,"$ \hat d33$","$ \hat q0$","$ \hat q1$","$ \hat q2$","$ \hat q3$"
                        ,"$ \hat q4$","$ \hat q5$","$ \hat q6$"]

    LABELERROR4D = ["$x - \hat x$","$y - \hat y$","$z - \hat z$","$X - \hat X$","$ Y - \hat Y$",
                        "$Z - \hat Z$" ,"$ W -  \hat W$","$ q0 -  \hat q0$","$ q1 -  \hat q1$","$ q2 -  \hat q2$"
                        ,"$ q3 -  \hat q3$","$ q4 -  \hat q4$","$ q5 -  \hat q5$","$ q6 -  \hat q6$"]
    LABELERROR6D = ["$x - \hat x$","$y - \hat y$","$z - \hat z$","$d11 - \hat d11$","$ d12 - \hat d12$",
                        "$d13 - \hat d13$" ,"$ d22 -  \hat d22$","$d23 - \hat d23$" ,"$ d33 -  \hat d33$",
                        "$ q0 -  \hat q0$","$ q1 -  \hat q1$","$ q2 -  \hat q2$","$ q3 -  \hat q3$",
                        "$ q4 -  \hat q4$","$ q5 -  \hat q5$","$ q6 -  \hat q6$"] 

    TEXTID = []

    LABELID =['RMSE',
              'R²',
              'NRMSE',
              'Fit Index']  

    values = []    

    BBOX = dict(facecolor = 'yellow', alpha = 1)

class postprocess(dataset, plotcfg):
    """
    Postprocessor object to log and plot the acquired data. Works in a script with the metric
    dictionary if preprocessor is already instantiated, otherwise metrics should be inputted into
    class methods

    FOR A MODEL:
    SINGLE ENVIRONMENT OVER ALL JOINTS
    SINGLE JOINT OVER ALL ENVIRONMENTS

    ALL JOINTS MEAN - ENVIRONMENT
    ALL ENVIRONMENTS MEAN - JOINT

    ALL JOINTS & ENVIRONMENTS MEAN
    """
    def __init__(self, args, metrics, gendict):
        self.args = args
        self.metrics = metrics
        self.gendict = gendict

        self.logpath = f'sys_identification/logs'
        self.summarypath = f'sys_identification/summary'
        self.figpath = f'sys_identification/plots'

        plt.rc('font', **self.FONT)

    def __str__(self):
        return 'Posprocess Object Instantiated'
    
    def configure_postprocess(self):
        pass

    def logdata(self, meanmetrics):
        """
        Data logger into txt file, verbose version of summarize
        """
        pass

    def summarize(self, meanmetrics):
        """
        Summarizes training and testing progression for a given training or testing scripts, logs directly
        to the terminal
        """
        pass

    def tabulate(self, meanmetrics):
        """
        Tabulates testing metrics and logs table
        """
        pass

    def spike_removal(y, 
                  width_threshold, 
                  prominence_threshold=None, 
                  moving_average_window=10, 
                  width_param_rel=0.8, 
                  interp_type='linear'):
        """
        Detects and replaces spikes in the input spectrum with interpolated values. Algorithm first 
        published by N. Coca-Lopez in Analytica Chimica Acta. https://doi.org/10.1016/j.aca.2024.342312

        Parameters:
        y (numpy.ndarray): Input spectrum intensity.
        width_threshold (float): Threshold for peak width.
        prominence_threshold (float): Threshold for peak prominence.
        moving_average_window (int): Number of points in moving average window.
        width_param_rel (float): Relative height parameter for peak width.
        tipo: type of interpolation (linear, quadratic, cubic)
        
        Returns:
        numpy.ndarray: Signal with spikes replaced by interpolated values.
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

    def plotlosses(self, training_loss_list, validation_loss_list, method, save=False):
        """
        Plots the training, validation or testing loss function according given a user-defined loss metric, plots
        are generated on the aggregated loss over multiple datasets - if available
        """
        fig, ax = plt.subplots(1, 2, sharey=True)

        ax[0].plot(training_loss_list)
        ax[0].set_title(f'Training Loss Over Iterations')
        ax[0].grid(True)
        ax[0].legend(['Training Loss'])
        ax[0].set_xlabel('Iterations')
        ax[0].set_ylabel(f'Loss - {method}')

        ax[1].plot(validation_loss_list)
        ax[1].set_title(f'Validation Loss Over Iterations')
        ax[1].grid(True)
        ax[1].legend(['Validation Loss'])
        ax[1].set_xlabel('Iterations')
        ax[1].set_ylabel(f'Loss - {method}')

        fig.suptitle(f'{method} Loss Variation Over Iterations')
        if save:
            plt.savefig(f'{self.figpath}/lossvariation.png')
        plt.show()

    def plotsim2sim(self, sim, true, err, dim, save=False):
        """
        Plots the true generated pose, estimated pose and relative simulation error on the original
        time frame defined by the true generated pose.
        """
        if dim==4:
            fig, ax = plt.subplots(7, 2, figsize=(20, 20))

            k = 0
            for j in range(2):
                for i in range(7):
                    ax[i,j].plot(true[:,k] ,'k', 
                        label=self.LABELCOORDS4D[k],linewidth=2)
                    ax[i,j].plot(sim[:,k], 'b', 
                        label=self.LABELPRED4D[k],linewidth=2)
                    ax[i,j].plot(err[:,k], 'r', 
                        label=self.LABELERROR4D[k])
                    ax[i,j].grid(True)
                    ax[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax[i,j].set_xlabel('Iterations')
                    k=k+1

        elif dim==6:
            fig = plt.figure(figsize = (20,20))
            gs = matplotlib.gridspec.GridSpec(9,2, figure=fig)

            k = 0
            for i1 in range(9):
                ax1 = fig.addsubplot(gs[i1,1])
                ax1.plot(true[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=2)
                ax1.plot(sim[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=2)
                ax1.plot(err[:,k], 'r', 
                    label=self.LABELERROR6D[k])
                ax1.grid(True)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax1.set_xlabel('Iterations')
                k=k+1
            
            for i2 in range(7):
                ax2 = fig.addsubplot(gs[i2,2])
                ax2.plot(true[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=2)
                ax2.plot(sim[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=2)
                ax2.plot(err[:,k], 'r', 
                    label=self.LABELERROR6D[k])
                ax2.grid(True)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax2.set_xlabel('Iterations')
                k=k+1

        fig.suptitle('Ground Truth vs Model Prediction',size = 20,y = .99)
        fig.tight_layout( pad = 2)
        if save:
            plt.savefig(self.figpath/f"{self.labels}.png")
        plt.show()

    def plotsim2real(self, sim, real, err, dim, save=False):
        """
        Plots the true acquired and conditioned pose, estimated pose and relative simulation error 
        on the original time frame defined by the true acquired pose.
        """
        if dim==4:
            fig, ax = plt.subplots(7, 2, figsize=(20, 20))

            k = 0
            for j in range(2):
                for i in range(7):
                    ax[i,j].plot(real[:,k] ,'k', 
                        label=self.LABELCOORDS4D[k],linewidth=2)
                    ax[i,j].plot(sim[:,k], 'b', 
                        label=self.LABELPRED4D[k],linewidth=2)
                    ax[i,j].plot(err[:,k], 'r', 
                        label=self.LABELERROR4D[k])
                    ax[i,j].grid(True)
                    ax[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax[i,j].set_xlabel('Iterations')
                    k=k+1

        elif dim==6:
            fig = plt.figure(figsize = (20,20))
            gs = matplotlib.gridspec.GridSpec(9,2, figure=fig)

            k = 0
            for i1 in range(9):
                ax1 = fig.addsubplot(gs[i1,1])
                ax1.plot(real[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=2)
                ax1.plot(sim[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=2)
                ax1.plot(err[:,k], 'r', 
                    label=self.LABELERROR6D[k])
                ax1.grid(True)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax1.set_xlabel('Iterations')
                k=k+1
            
            for i2 in range(7):
                ax2 = fig.addsubplot(gs[i2,2])
                ax2.plot(real[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=2)
                ax2.plot(sim[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=2)
                ax2.plot(err[:,k], 'r', 
                    label=self.LABELERROR6D[k])
                ax2.grid(True)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax2.set_xlabel('Iterations')
                k=k+1

        fig.suptitle('Acquisation vs Model Prediction',size = 20,y = .99)
        fig.tight_layout( pad = 2)
        if save:
            plt.savefig(self.figpath/f"{self.labels}.png")
        plt.show()

    def plothorizon(self, sim, total, err, dim, iter, ctx, save=False):
        """
        Plots the horizon prediction for a given simulation determined by the user.
        """
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
                    ax[i,j].plot(t_prediction,sim[:,k], 'b', 
                        label=self.LABELPRED4D[k],linewidth=2)
                    ax[i,j].plot(t_prediction,err[:,k], 'r', 
                        label=self.LABELERROR4D[k])
                    ax[i,j].grid(True)
                    ax[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax[i,j].set_xlabel('Iterations')
                    k=k+1

        elif dim==6:
            fig = plt.figure(figsize = (20,20))
            gs = matplotlib.gridspec.GridSpec(9,2, figure=fig)

            k = 0
            for i1 in range(9):
                ax1 = fig.addsubplot(gs[i1,1])
                ax1.axvline(x = t_prediction[-1], 
                    color = 'k', linestyle='--')
                ax1.axvline(x = t_context[-1], 
                    color = 'k', linestyle='--')
                ax1.axvspan(t_context[0], t_context[-1],
                        facecolor='lime', alpha=0.2)

                ax1.plot(t_total,total[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=2)
                ax1.plot(t_prediction,sim[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=2)
                ax1.plot(t_prediction,err[:,k], 'r', 
                    label=self.LABELERROR6D[k])
                ax1.grid(True)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax1.set_xlabel('Iterations')
                k=k+1

            for i2 in range(7):
                ax2 = fig.addsubplot(gs[i2,2])
                ax2.axvline(x = t_prediction[-1], 
                    color = 'k', linestyle='--')
                ax2.axvline(x = t_context[-1], 
                    color = 'k', linestyle='--')
                ax2.axvspan(t_context[0], t_context[-1],
                        facecolor='lime', alpha=0.2)

                ax2.plot(t_total,total[:,k] ,'k', 
                    label=self.LABELCOORDS6D[k],linewidth=2)
                ax2.plot(t_prediction,sim[:,k], 'b', 
                    label=self.LABELPRED6D[k],linewidth=2)
                ax2.plot(t_prediction,err[:,k], 'r', 
                    label=self.LABELERROR6D[k])
                ax2.grid(True)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax2.set_xlabel('Iterations')
                k=k+1

        fig.suptitle('Model Prediction of Time Horizon',size = 20,y = .99)
        fig.tight_layout( pad = 2)
        if save:
            plt.savefig(self.figpath/f"{self.labels}.png")
        plt.show()

    def plotpredictionerror(self, sim, total, dim, save=False):
        """
        Plots the percent prediction error according to user given prediciton and
        ground truth simulation
        """
        err = np.abs((sim-total)/total)*100

        if dim==4:
            fig, ax = plt.subplots(7, 2, figsize=(20, 20))

            k = 0
            for j in range(2):
                for i in range(7):
                    errc = uniform_filter1d(err[:,k],11)
                    ax[i,j].plot(errc, 'r', 
                        label=self.LABELERROR4D[k])
                    ax[i,j].grid(True)
                    ax[i,j].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    ax[i,j].set_xlabel('Iterations')
                    ax[i,j].set_ylim(-10,10)
                    k=k+1

        elif dim==6:
            fig = plt.figure(figsize = (20,20))
            gs = matplotlib.gridspec.GridSpec(9,2, figure=fig)

            k = 0
            for i1 in range(9):
                ax1 = fig.addsubplot(gs[i1,1])
                errc = uniform_filter1d(err[:,k],11)
                ax1.plot(errc, 'r', 
                    label=self.LABELERROR4D[k])
                ax1.grid(True)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax1.set_xlabel('Iterations')
                ax1.set_ylim(-10,10)
                k=k+1
            
            for i2 in range(7):
                ax2 = fig.addsubplot(gs[i2,2])
                errc = uniform_filter1d(err[:,k],11)
                ax2.plot(errc, 'r', 
                    label=self.LABELERROR4D[k])
                ax2.grid(True)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                ax2.set_xlabel('Iterations')
                ax2.set_ylim(-10,10)
                k=k+1

        fig.suptitle('Percent Relative Error for a Single Simulation',size = 20,y = .99)
        fig.tight_layout( pad = 2)
        if save:
            plt.savefig(self.figpath/f"{self.labels}.png")
        plt.show()

    def plotmetrics_overtime(self, ysim, ytrue, save=False):
        """
        Plots the variation of metrics over a single test on a single model over the available 
        environments. Variation over environments is demonstrated through percentiles and 
        quantiles.
        """
        r2 = r_squared(ytrue, ysim, time_axis=0)
        r2m = np.mean(r2, axis=1)

        r = rmse(ytrue, ysim, time_axis=0)
        rm = np.mean(r, axis=1)

        f = fit_index(ytrue, ysim, time_axis=0)
        fm = np.mean(f, axis=1)

        nr = nrmse(ytrue, ysim, time_axis=0)
        nrm = np.mean(nr, axis=1)       

        fig,[[ax1, ax2],[ax3, ax4]]  = plt.subplots(2, 2,figsize=(15, 15), dpi=200)

        ax1.plot(r2m,
                        linewidth=1.5)
        ax1.plot(np.mean(r2m),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax1.scatter(np.nanargmin(r2m),
                        np.nanmin(r2m),
                        marker='o', color='red', s=30)
        ax1.scatter(np.nanargmax(r2m),
                        np.nanmax(r2m),
                        marker='o', color='red', s=30)
        ax1.axhspan(np.nanpercentile(r2m,25), 
                        np.nanpercentile(r2m,75), 
                        facecolor='yellow', alpha=0.2)
        ax1.grid(visible=True, which='both')
        ax1.set_title('$R^{2}$ Value')
        ax1.set(xlabel='Environment Index')
        

        ax2.plot(rm,
                        linewidth=1.5)
        ax2.plot(np.mean(rm),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax2.scatter(np.nanargmin(rm),
                        np.nanmin(rm),
                        marker='o', color='red', s=30)
        ax2.scatter(np.nanargmax(rm),
                        np.nanmax(rm),
                        marker='o', color='red', s=30)
        ax2.axhspan(np.nanpercentile(rm,25), 
                        np.nanpercentile(rm,75), 
                        facecolor='yellow', alpha=0.2)
        ax2.grid(visible=True, which='both')
        ax2.set_title('$RMSE$ Value')
        ax2.set(xlabel='Environment Index')


        ax3.plot(fm,
                        linewidth=1.5)
        ax3.plot(np.mean(fm),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax3.scatter(np.nanargmin(fm),
                        np.nanmin(fm),
                        marker='o', color='red', s=30)
        ax3.scatter(np.nanargmax(fm),
                        np.nanmax(fm),
                        marker='o', color='red', s=30)
        ax3.axhspan(np.nanpercentile(fm,25), 
                        np.nanpercentile(fm,75), 
                        facecolor='yellow', alpha=0.2)
        ax3.grid(visible=True, which='both')
        ax3.set_title('$fit$ Value')
        ax3.set(xlabel='Environment Index')


        ax4.plot(nrm,
                        linewidth=1.5)
        ax4.plot(np.mean(nrm),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax4.scatter(np.nanargmin(nrm),
                        np.nanmin(nrm),
                        marker='o', color='red', s=30)
        ax4.scatter(np.nanargmax(nrm),
                        np.nanmax(nrm),
                        marker='o', color='red', s=30)
        ax4.axhspan(np.nanpercentile(nrm,25), 
                        np.nanpercentile(nrm,75), 
                        facecolor='yellow', alpha=0.2)
        ax4.grid(visible=True, which='both')
        ax4.set_title('$NRMSE$ Value')
        ax4.set(xlabel='Environment Index')

        fig.tight_layout(pad = 1.2 )
        fig.suptitle('Cumulative Accuracy Metrics Projected on All Environments',size = 20,y = .99)
        if save:
            plt.savefig(self.figpath/f"metrics.png")
        plt.show()


    def plotmetrics_overjoints(self, metrics, save=False):
        """
        Plots the variation of metrics over a single test on a single model over the available 
        environments. Variation over environments is demonstrated through percentiles and 
        quantiles.
        """
        fig,[[ax1, ax2],[ax3, ax4]]  = plt.subplots(2, 2,figsize=(15, 15), dpi=200)

        ax1.plot(metrics['r2']['cmean'],
                        linewidth=1.5)
        ax1.plot(np.mean(metrics['r2']['cmean']),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax1.scatter(np.nanargmin(metrics['r2']['cmean']),
                        np.nanmin(metrics['r2']['cmean']),
                        marker='o', color='red', s=30)
        ax1.scatter(np.nanargmax(metrics['r2']['cmean']),
                        np.nanmax(metrics['r2']['cmean']),
                        marker='o', color='red', s=30)
        ax1.axhspan(np.nanpercentile(metrics['r2']['cmean'],25), 
                        np.nanpercentile(metrics['r2']['cmean'],75), 
                        facecolor='yellow', alpha=0.2)
        ax1.grid(visible=True, which='both')
        ax1.set_title('$R^{2}$ Value')
        ax1.set(xlabel='Environment Index')
        

        ax2.plot(metrics['r']['cmean'],
                        linewidth=1.5)
        ax2.plot(np.mean(metrics['r']['cmean']),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax2.scatter(np.nanargmin(metrics['r']['cmean']),
                        np.nanmin(metrics['r']['cmean']),
                        marker='o', color='red', s=30)
        ax2.scatter(np.nanargmax(metrics['r']['cmean']),
                        np.nanmax(metrics['r']['cmean']),
                        marker='o', color='red', s=30)
        ax2.axhspan(np.nanpercentile(metrics['r']['cmean'],25), 
                        np.nanpercentile(metrics['r']['cmean'],75), 
                        facecolor='yellow', alpha=0.2)
        ax2.grid(visible=True, which='both')
        ax2.set_title('$RMSE$ Value')
        ax2.set(xlabel='Environment Index')


        ax3.plot(metrics['f']['cmean'],
                        linewidth=1.5)
        ax3.plot(np.mean(metrics['f']['cmean']),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax3.scatter(np.nanargmin(metrics['f']['cmean']),
                        np.nanmin(metrics['f']['cmean']),
                        marker='o', color='red', s=30)
        ax3.scatter(np.nanargmax(metrics['f']['cmean']),
                        np.nanmax(metrics['f']['cmean']),
                        marker='o', color='red', s=30)
        ax3.axhspan(np.nanpercentile(metrics['f']['cmean'],25), 
                        np.nanpercentile(metrics['f']['cmean'],75), 
                        facecolor='yellow', alpha=0.2)
        ax3.grid(visible=True, which='both')
        ax3.set_title('$fit$ Value')
        ax3.set(xlabel='Environment Index')


        ax4.plot(metrics['nr']['cmean'],
                        linewidth=1.5)
        ax4.plot(np.mean(metrics['nr']['cmean']),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax4.scatter(np.nanargmin(metrics['nr']['cmean']),
                        np.nanmin(metrics['nr']['cmean']),
                        marker='o', color='red', s=30)
        ax4.scatter(np.nanargmax(metrics['nr']['cmean']),
                        np.nanmax(metrics['nr']['cmean']),
                        marker='o', color='red', s=30)
        ax4.axhspan(np.nanpercentile(metrics['nr']['cmean'],25), 
                        np.nanpercentile(metrics['nr']['cmean'],75), 
                        facecolor='yellow', alpha=0.2)
        ax4.grid(visible=True, which='both')
        ax4.set_title('$NRMSE$ Value')
        ax4.set(xlabel='Environment Index')

        fig.tight_layout(pad = 1.2 )
        fig.suptitle('Cumulative Accuracy Metrics Projected on All Environments',size = 20,y = .99)
        if save:
            plt.savefig(self.figpath/f"metrics.png")
        plt.show()

    def plotvariation_overjoints(self, metrics, save=False):
        """
        Plots the test variation over all joints and all environments for a given model
        """
        m1list = [metrics['r']['valuec'],metrics['r2']['valuec'],
                    metrics['nr']['valuec'],metrics['f']['valuec']]
        
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
            ax1.set_title(f'${self.LABELID[j]}$')
            ax1.grid(alpha=0.5)

        fig1.suptitle(f'Overall Joint - Environment Variation of a Single Simulation - End Effector and Joints')
        if save:
            plt.savefig(self.figpath/f"metrics_joint_pos.png")
        plt.show()

    def plotmetrics_overenvironments(self, metrics, save=False):
        """
        Plots the variation of metrics over a single test on a single model over the available 
        environments. Variation over environments is demonstrated through percentiles and 
        quantiles.
        """
        fig,[[ax1, ax2],[ax3, ax4]]  = plt.subplots(2, 2,figsize=(15, 15), dpi=200)

        ax1.plot(metrics['r2']['emean'],
                        linewidth=1.5)
        ax1.plot(np.mean(metrics['r2']['emean']),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax1.scatter(np.nanargmin(metrics['r2']['emean']),
                        np.nanmin(metrics['r2']['emean']),
                        marker='o', color='red', s=30)
        ax1.scatter(np.nanargmax(metrics['r2']['emean']),
                        np.nanmax(metrics['r2']['emean']),
                        marker='o', color='red', s=30)
        ax1.axhspan(np.nanpercentile(metrics['r2']['emean'],25), 
                        np.nanpercentile(metrics['r2']['emean'],75), 
                        facecolor='yellow', alpha=0.2)
        ax1.grid(visible=True, which='both')
        ax1.set_title('$R^{2}$ Value')
        ax1.set(xlabel='Environment Index')
        

        ax2.plot(metrics['r']['emean'],
                        linewidth=1.5)
        ax2.plot(np.mean(metrics['r']['emean']),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax2.scatter(np.nanargmin(metrics['r']['emean']),
                        np.nanmin(metrics['r']['emean']),
                        marker='o', color='red', s=30)
        ax2.scatter(np.nanargmax(metrics['r']['emean']),
                        np.nanmax(metrics['r']['emean']),
                        marker='o', color='red', s=30)
        ax2.axhspan(np.nanpercentile(metrics['r']['emean'],25), 
                        np.nanpercentile(metrics['r']['emean'],75), 
                        facecolor='yellow', alpha=0.2)
        ax2.grid(visible=True, which='both')
        ax2.set_title('$RMSE$ Value')
        ax2.set(xlabel='Environment Index')


        ax3.plot(metrics['f']['emean'],
                        linewidth=1.5)
        ax3.plot(np.mean(metrics['f']['emean']),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax3.scatter(np.nanargmin(metrics['f']['emean']),
                        np.nanmin(metrics['f']['emean']),
                        marker='o', color='red', s=30)
        ax3.scatter(np.nanargmax(metrics['f']['emean']),
                        np.nanmax(metrics['f']['emean']),
                        marker='o', color='red', s=30)
        ax3.axhspan(np.nanpercentile(metrics['f']['emean'],25), 
                        np.nanpercentile(metrics['f']['emean'],75), 
                        facecolor='yellow', alpha=0.2)
        ax3.grid(visible=True, which='both')
        ax3.set_title('$fit$ Value')
        ax3.set(xlabel='Environment Index')


        ax4.plot(metrics['nr']['emean'],
                        linewidth=1.5)
        ax4.plot(np.mean(metrics['nr']['emean']),'r',
                        linewidth=1.5,linestyle='dotted') 
        ax4.scatter(np.nanargmin(metrics['nr']['emean']),
                        np.nanmin(metrics['nr']['emean']),
                        marker='o', color='red', s=30)
        ax4.scatter(np.nanargmax(metrics['nr']['emean']),
                        np.nanmax(metrics['nr']['emean']),
                        marker='o', color='red', s=30)
        ax4.axhspan(np.nanpercentile(metrics['nr']['emean'],25), 
                        np.nanpercentile(metrics['nr']['emean'],75), 
                        facecolor='yellow', alpha=0.2)
        ax4.grid(visible=True, which='both')
        ax4.set_title('$NRMSE$ Value')
        ax4.set(xlabel='Environment Index')

        fig.tight_layout(pad = 1.2 )
        fig.suptitle('Cumulative Accuracy Metrics Projected on All Environments',size = 20,y = .99)
        if save:
            plt.savefig(self.figpath/f"metrics.png")
        plt.show()
    
    def plotvariation_overenvironments(self, metrics, save=False):
        """
        Plots the test variation over all joints and all environments for a given model
        """
        m1list = [metrics['r']['valuee'],metrics['r2']['valuee'],
                    metrics['nr']['valuee'],metrics['f']['valuee']]

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
            ax1.set_title(f'${self.LABELID[j]}$')
            ax1.grid(alpha=0.5)

        fig1.suptitle(f'Overall Joint - Environment Variation of a Single Simulation - End Effector')
        if save:
            plt.savefig(self.figpath/f"metrics_joint_pos.png")
        plt.show()

    def plotvariation_total(self, metricsalltests, testlist, testscores, save=False):
        """
        Plots the test variation over all joints and all environments for a given model
        pertaining to all tests performed. The relevant distribution of a test dataset 
        related to the training dataset is incurred from a score from 1 to 10.
        """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18,18))

        axes[0,0].set_title('$R^2$', y=1.02,size=25)
        axes[0,0].boxplot(metricsalltests['r2'],1,'',patch_artist = True, boxprops = dict(facecolor = "lightblue"), 
                        medianprops = dict(color = "green", linewidth = 1.5), whiskerprops = dict(color = "red", linewidth = 2))
        axes[0,0].grid(True,alpha=0.6)    

        axes[1,0].set_title('$fit_index$', y=1.02,size=25)
        axes[1,0].boxplot(metricsalltests['f'],1,'',patch_artist = True, boxprops = dict(facecolor = "lightblue"), 
                        medianprops = dict(color = "green", linewidth = 1.5), whiskerprops = dict(color = "red", linewidth = 2))
        axes[1,0].grid(True,alpha=0.6)  

        axes[0,1].set_title('$NRMSE$', y=1.02,size=25)
        axes[0,1].boxplot(metricsalltests['nr'],1,'',patch_artist = True, boxprops = dict(facecolor = "lightblue"), 
                        medianprops = dict(color = "green", linewidth = 1.5), whiskerprops = dict(color = "red", linewidth = 2))
        axes[0,1].grid(True,alpha=0.6)  

        axes[1,1].set_title('$RMSE$', y=1.02,size=25)
        axes[1,1].boxplot(metricsalltests['r'],1,'',patch_artist = True, medianprops = dict(color = "green", linewidth = 1.5), 
                        boxprops = dict(facecolor = "lightblue"), whiskerprops = dict(color = "red", linewidth = 2))
        axes[1,1].grid(True,alpha=0.6) 
            

        fig.suptitle(f'Variation of Model over All Tests',fontsize=25) 
        plt.setp(axes, xticklabels=testscores)
        plt.tight_layout( pad = 1.2)

        plt.show()     
 
    def plotvariation_total_all(self, metricsallmodels, testlist, modellist, testscores, metric='fit_index', save=False):
        """
        Plots the test variation over all joints and all environments for all performed tests over
        all available models. The metric of is user defined
        """
        if metric=='f':
            c = metricsallmodels['f']
        elif metric=='r':
            c = metricsallmodels['r']
        elif metric=='nr':
            c = metricsallmodels['nr']
        elif metric=='r2':
            c = metricsallmodels['r2']

        fig, axes = plt.subplots(nrows=len(modellist), ncols=1, figsize=(18,18))
        for i,model in enumerate(modellist):
            axes[i].set_title(f'o', y=1.02,size=25)
            axes[i].boxplot(c[i],1,'',patch_artist = True, boxprops = dict(facecolor = "lightblue"), 
                            medianprops = dict(color = "green", linewidth = 1.5), whiskerprops = dict(color = "red", linewidth = 2))
            axes[i].grid(True,alpha=0.6)    
            plt.setp(axes, xticklabels=testscores[i])
        fig.suptitle(f'Variation of Model over All Tests',fontsize=25) 
        
        plt.tight_layout( pad = 1.2)
        if save:
            plt.savefig(f'{self.fig_path}/bababammm.png')
        plt.show()
    
    def plotmetric(self, metrics, jointmean=False, envmean=False, save=False):
        """
        Applies a chosen metric according to specified mean parameters to a batch of simulations.
        """
        if not jointmean and not envmean:
            print('Impossible to parse without projection onto joint space or environment space')
            return
        
        elif jointmean and not envmean:
            r = metrics['r']['cmean']
            r2 = metrics['r2']['cmean']
            f = metrics['f']['cmean']
            nr = metrics['nr']['cmean']

            fig,[[ax1, ax2],[ax3, ax4]]  = plt.subplots(2, 2,figsize=(15, 15), dpi=200)

            ax1.plot(r2,
                            linewidth=1.5)
            ax1.plot(np.mean(r2),'r',
                            linewidth=1.5,linestyle='dotted') 
            ax1.scatter(np.nanargmin(r2),
                            np.nanmin(r2),
                            marker='o', color='red', s=30)
            ax1.scatter(np.nanargmax(r2),
                            np.nanmax(r2),
                            marker='o', color='red', s=30)
            ax1.axhspan(np.nanpercentile(r2,25), 
                            np.nanpercentile(r2,75), 
                            facecolor='yellow', alpha=0.2)
            ax1.grid(visible=True, which='both')
            ax1.set_title('$R^{2}$ Value')
            ax1.set(xlabel='Environment Index')
            

            ax2.plot(r,
                            linewidth=1.5)
            ax2.plot(np.mean(r),'r',
                            linewidth=1.5,linestyle='dotted') 
            ax2.scatter(np.nanargmin(r),
                            np.nanmin(r),
                            marker='o', color='red', s=30)
            ax2.scatter(np.nanargmax(r),
                            np.nanmax(r),
                            marker='o', color='red', s=30)
            ax2.axhspan(np.nanpercentile(r,25), 
                            np.nanpercentile(r,75), 
                            facecolor='yellow', alpha=0.2)
            ax2.grid(visible=True, which='both')
            ax2.set_title('$RMSE$ Value')
            ax2.set(xlabel='Environment Index')


            ax3.plot(f,
                            linewidth=1.5)
            ax3.plot(np.mean(f),'r',
                            linewidth=1.5,linestyle='dotted') 
            ax3.scatter(np.nanargmin(f),
                            np.nanmin(f),
                            marker='o', color='red', s=30)
            ax3.scatter(np.nanargmax(f),
                            np.nanmax(f),
                            marker='o', color='red', s=30)
            ax3.axhspan(np.nanpercentile(f,25), 
                            np.nanpercentile(f,75), 
                            facecolor='yellow', alpha=0.2)
            ax3.grid(visible=True, which='both')
            ax3.set_title('$fit$ Value')
            ax3.set(xlabel='Environment Index')


            ax4.plot(nr,
                            linewidth=1.5)
            ax4.plot(np.mean(nr),'r',
                            linewidth=1.5,linestyle='dotted') 
            ax4.scatter(np.nanargmin(nr),
                            np.nanmin(nr),
                            marker='o', color='red', s=30)
            ax4.scatter(np.nanargmax(nr),
                            np.nanmax(nr),
                            marker='o', color='red', s=30)
            ax4.axhspan(np.nanpercentile(nr,25), 
                            np.nanpercentile(nr,75), 
                            facecolor='yellow', alpha=0.2)
            ax4.grid(visible=True, which='both')
            ax4.set_title('$NRMSE$ Value')
            ax4.set(xlabel='Environment Index')

            fig.tight_layout(pad = 1.2 )
            fig.suptitle('Cumulative Accuracy Metrics Projected on All Environments',size = 20,y = .99)
            if save:
                plt.savefig(self.figpath/f"metrics.png")
            plt.show()

        elif envmean and not jointmean:
            r = metrics['r']['emean'].T
            r2 = metrics['r2']['emean'].T
            f = metrics['f']['emean'].T
            nr = metrics['nr']['emean'].T

            fig,[[ax1, ax2],[ax3, ax4]]  = plt.subplots(2, 2,figsize=(15, 15), dpi=200)

            ax1.plot(r2,
                            linewidth=1.5)
            ax1.plot(np.mean(r2),'r',
                            linewidth=1.5,linestyle='dotted') 
            ax1.scatter(np.nanargmin(r2),
                            np.nanmin(r2),
                            marker='o', color='red', s=30)
            ax1.scatter(np.nanargmax(r2),
                            np.nanmax(r2),
                            marker='o', color='red', s=30)
            ax1.axhspan(np.nanpercentile(r2,25), 
                            np.nanpercentile(r2,75), 
                            facecolor='yellow', alpha=0.2)
            ax1.grid(visible=True, which='both')
            ax1.set_title('$R^{2}$ Value')
            ax1.set(xlabel='Joint Index')
            

            ax2.plot(r,
                            linewidth=1.5)
            ax2.plot(np.mean(r),'r',
                            linewidth=1.5,linestyle='dotted') 
            ax2.scatter(np.nanargmin(r),
                            np.nanmin(r),
                            marker='o', color='red', s=30)
            ax2.scatter(np.nanargmax(r),
                            np.nanmax(r),
                            marker='o', color='red', s=30)
            ax2.axhspan(np.nanpercentile(r,25), 
                            np.nanpercentile(r,75), 
                            facecolor='yellow', alpha=0.2)
            ax2.grid(visible=True, which='both')
            ax2.set_title('$RMSE$ Value')
            ax2.set(xlabel='Joint Index')


            ax3.plot(f,
                            linewidth=1.5)
            ax3.plot(np.mean(f),'r',
                            linewidth=1.5,linestyle='dotted') 
            ax3.scatter(np.nanargmin(f),
                            np.nanmin(f),
                            marker='o', color='red', s=30)
            ax3.scatter(np.nanargmax(f),
                            np.nanmax(f),
                            marker='o', color='red', s=30)
            ax3.axhspan(np.nanpercentile(f,25), 
                            np.nanpercentile(f,75), 
                            facecolor='yellow', alpha=0.2)
            ax3.grid(visible=True, which='both')
            ax3.set_title('$fit$ Value')
            ax3.set(xlabel='Joint Index')


            ax4.plot(nr,
                            linewidth=1.5)
            ax4.plot(np.mean(nr),'r',
                            linewidth=1.5,linestyle='dotted') 
            ax4.scatter(np.nanargmin(nr),
                            np.nanmin(nr),
                            marker='o', color='red', s=30)
            ax4.scatter(np.nanargmax(nr),
                            np.nanmax(nr),
                            marker='o', color='red', s=30)
            ax4.axhspan(np.nanpercentile(nr,25), 
                            np.nanpercentile(nr,75), 
                            facecolor='yellow', alpha=0.2)
            ax4.grid(visible=True, which='both')
            ax4.set_title('$NRMSE$ Value')
            ax4.set(xlabel='Joint Index')

            fig.tight_layout(pad = 1.2 )
            fig.suptitle('Cumulative Accuracy Metrics Projected on All Joints',size = 20,y = .99)
            if save:
                plt.savefig(self.figpath/f"metrics.png")
            plt.show()
            
        elif jointmean and envmean:
            r = np.mean(metrics['r']['cmean'])
            r2 = np.mean(metrics['r2']['cmean'])
            f = np.mean(metrics['f']['cmean'])
            nr = np.mean(metrics['nr']['cmean'])

            print(f'RMSE: {r}\nr^2: {r2}\nNRMSE: {nr}\nfit index: {f}')
        
    def zeroshot_horizon(self, sim, total, error, dim, ctx, iter, save=False):
        """
        Alias for plotting horizon over a single environment
        """
        return self.plothorizon(self, sim, total, error, dim, ctx, iter, save=save)

    def zeroshot_metrics_overenvs(self, metrics, save=False):
        """
        Alias for jointmean plot of available metrics
        """
        return self.plotmetrics_overjoints(self, metrics, save=save)

    def oneshot_horizon(self, sim, total, error, dim, shot, save=False):
        """
        Single example - oneshot 2000 iterations
        """
        pass

    def oneshot_metrics_overenvs(self, sim, total, error, dim, shot, metric, save=False):
        """
        Single example - oneshot 2000 iterations
        """
        pass

    def fewshot_horizon(self, sim, total, error, dim, numofshots, save=False):
        """
        Multiple examples - fewshot 5000-15000 iterations
        """
        pass

    def fewshot_metrics_overenvs(self, sim, total, error, dim, shot, metric, save=False):
        """
        Multiple examples - fewshot 5000-15000 iterations
        """
        pass
    
    def plotfinetune_horizon(self):
        """
        Comparisons between trained and finetuned model on horizon prediction
        """
        pass

    def plotfinetune_metrics_overenvs(self):
        """
        Comparisons between trained and finetuned model on metrics assessment
        """
        pass
    
