"""
EXPLAIN MODULE
check methods
fix learning rate schedulers

loss plateau
optimizer hyperparameters
model hyperparameters
training time **
"""

import torch
import math
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingWarmRestarts, ExponentialLR, StepLR

import datetime
import os
import json
from pathlib import Path

from architectures.transformer.transformer_sim import Config, TSTransformer
from toydataset import *

import wandb

class cfg():
    JOINTS=7
    COORDINATES=14
    TIMESTEPS=1000

    STEPSIZE=1.0

    GAMMA=1.0

    WARMUP_ITERATIONS=200

    TOTAL_ITERATIONS=2000
    
class dataset(cfg):
    """
    Dataset object that utilizes any custom dataset with the prescribed data format to be imposed
    in the training regime. The data format is:
    control: 
    position:
    diff:
    mass:
    Once the dataset object is instantiated, the getdata() method can be used to obtain the metadata
    of a certain dataset whereas the getgen() method can be used to access the specific generation files
    inside the specified dataset. When using getgen() it is important to also call delgen() to remove and 
    clean generation data in each iteration over a dataset including multitudes of generation data.

    Get/Set methods initiate dataset properties such as:
    metadata:
    generation:
    dataset:
    model:
    These properties pertain to every loop and should be called and cast back to the dataset object when
    the loop is over / at the end of each iteration
    """
    def __init__(self,args):
        torch.cuda.empty_cache()

        self.args = args
        if not args.disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.starttime = None

        self.modelpath = ''
        self.traindatapath = ''
        self.testdatapath = ''
        self.metapath = ''

        self.traindatalist = []
        self.testdatalist = []
        self.modellist = []

        self.control = torch.empty(0)
        self.diff = torch.empty(0)
        self.mass = torch.empty(0)
        self.position = torch.empty(0)
        self.gendict = {
            "control" : self.control,
            "position" : self.position,
            "diff" : self.diff,
            "mass" : self.mass
        }

        self.ny = args.num_of_coordinates
        self.nu = args.num_of_joints
        self.sql = args.total_sim_iterations
        self.sqlctx = int(self.args.context/100 * self.sql)
        self.sqlpar = self.sql - self.sqlctx
        self.modelargs = {
            "n_layer" : self.args.n_layer, 
            "n_head" : self.args.n_head, 
            "n_embd" : self.args.n_embd, 
            "n_y" : self.ny, 
            "n_u" : self.nu,
            "seq_len_ctx" : self.sqlctx, 
            "seq_len_new" : self.sqlpar,
            "bias" : self.args.bias, 
            "dropout" : self.args.dropout
        }
        self.datadict = {}
        self.checkpoint = {}
        self.modelname = ''
        self.dataname = ''

        self.training_dataset = torch.empty(0)
        self.current_training_loss = np.inf
        self.training_loss_list = [np.inf]

        self.validation_dataset = torch.empty(0)
        self.current_validation_loss = np.inf
        self.validation_loss_list = [np.inf]
        self.best_validation_loss = [np.inf, np.inf]

        self.test_dataset = torch.empty(0)

        self.iter = 0
        self.learning_rate = args.learning_rate 
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def __str__(self):
        return 'Dataset Object Instantiated'

    def __iter__(self):
        return self
    
    def __next__(self):
        pass

    def __len__(self):
        return len(self.datadict["datalist"])
    
    def __getitem__(self, index):
        return self.datadict["datalist"][index]
    

    def settime(self, time):
        self.starttime = time

    def gettime(self, time):
        dt = datetime.timedelta(seconds=(time - self.starttime))
        print(f"\n{dt} seconds.")
        return dt


    def setmetadata(self, 
                    datadict_):
        self.datadict = datadict_

    def getmetadata(self):
        return self.datadict
    

    def setdataset(self, 
                   training_dataset_=None, 
                   validation_dataset_=None,
                   test_dataset_=None,
                   ):
        self.training_dataset = training_dataset_
        self.validation_dataset = validation_dataset_
        self.test_dataset = test_dataset_

    def getdataset(self):
        return (
            self.training_dataset, 
            self.validation_dataset,
            self.test_dataset 
        )
    

    def setgeneration(self, 
                      gendict_):
        self.gendict = gendict_

    def getgeneration(self):
        return self.gendict
    

    def setmodel(self, 
                 modelargs_, 
                 model_, 
                 optimizer_,
                 scheduler_):
        self.modelargs = modelargs_
        self.model = model_
        self.optimizer = optimizer_
        self.scheduler = scheduler_

    def getmodel(self):
        return (
            self.modelargs, 
            self.model, 
            self.optimizer,
            self.scheduler
        )
    

    def getlosslist(self):
        return (
            self.training_loss_list,
            self.validation_loss_list,
            self.best_validation_loss
        )
        
    def setlosslist(self, 
                    training_loss=None, 
                    validation_loss=None, 
                    best_validation_loss=None):
        if training_loss!=None:
            self.training_loss_list.append(training_loss)
            self.current_training_loss = training_loss
        
        if validation_loss!=None:
            self.validation_loss_list.append(validation_loss)
            self.current_validation_loss = validation_loss

        if best_validation_loss!=None:
            self.best_validation_loss[0] = self.best_validation_loss[1]
            self.best_validation_loss[1] = best_validation_loss


    def reset(self):
        torch.cuda.empty_cache()
        self.gendict = {}
        self.training_dataset = torch.empty(0)
        self.validation_dataset = torch.empty(0)

    def resolve_datasets(self, eval=False):
        """
        Function to resolve model characteristics and parameters as well as script particularities
        during training and testing.
        Eval --> training/testing
        """
        if not self.args.seed:
            self.seed = torch.randint(low=0,high=99999,size=(1,))
        else:
            self.seed = self.args.seed
        torch.manual_seed(seed=self.seed)

        parentpath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

        traindatadir = f'data_generation/data_tensors/train/{self.args.data_name}'
        self.traindatapath = os.path.join(parentpath,traindatadir)
        self.traindatalist = os.listdir(self.traindatapath)
        print(f'Training data is acquired from:\n{self.traindatapath}\n')

        metadir = f'data_generation/data_objects/{self.args.data_name}.json'
        self.metapath = os.path.join(parentpath,metadir)
        with open(self.metapath, 'r') as f:
            metadata  = json.load(f)
        totalsims = len(metadata["dataname"])
        totalenvs = sum(metadata["genenvs"])
        timetaken = datetime.timedelta(seconds=round(sum(metadata["gentime"])))
        print(f'Metadata is acquired from:\n{self.metapath}\n')

        if eval==False:
            modeldir = f'sys_identification/models'
            modelpath = os.path.join(parentpath,modeldir)
            Path(modelpath).mkdir(exist_ok=True)
            modelspecific = f'{modeldir}/{self.args.data_name}'
            self.modelpath = os.path.join(parentpath,modelspecific)
            Path(self.modelpath).mkdir(exist_ok=True)
            print(f'Trained model is to be saved to:\n{self.modelpath}\n')
        
        elif eval==True:
            modeldir = f'sys_identification/models'
            modelpath = os.path.join(parentpath,modeldir)
            modelspecific = f'{modeldir}/{self.args.data_name}'
            self.modelpath = os.path.join(parentpath,modelspecific)
            print(f'Test models are acquired from to:\n{self.modelpath}\n')
            self.modellist = os.listdir(self.modelpath)

            testdatadir = f'data_generation/data_tensors/test/{self.args.data_name}'
            self.testdatapath = os.path.join(parentpath,testdatadir)
            self.testdatalist = os.listdir(self.testdatapath)
            print(f'Test data is acquired from:\n{self.testdatapath}\n')

            print(f'Over {len(self.modellist)} different models trained on {self.args.data_name}')
            print(f'{len(self.testdatalist)} different datasets are to be tested.\nAvailable models and test datasets are:\n')
            print(f'Models-->\n{self.modellist}\n\nTest Datasets-->\n{self.testdatalist}\n')

        self.datadict = {
            "traindatalist" : self.traindatalist,
            "testdatalist" : self.testdatalist,
            "modellist" : self.modellist,
            "tsims" : totalsims,
            "tenvs" : totalenvs,
            "time" : timetaken,
            "mpath" : self.modelpath,
            "dpath" : self.traindatapath,
            "epath" : self.testdatapath
            }   

        print(f'Will use {self.device} for the training/testing\n')   
    
    def initialize_model(self, modelname=None):
        """
        Defines the model that is to be loaded, used and saved at the end of training procedure
        Modelname --> scratch/resume/finetune/test
        """
        gptconf = Config(**self.modelargs)
        self.model = TSTransformer(gptconf)
        self.optimizer = self.model.configure_optimizers(self.args.weight_decay, 
                                        self.args.learning_rate, 
                                        (self.args.beta1, self.args.beta2), 
                                        self.device)
        self.configure_learning_rate()

        if modelname==None:
            self.dataname = self.args.data_name
            self.modelname = (f'{self.args.subcommand}_{self.args.data_name}_{round(self.args.context)}_{self.args.training_batch_size}_'
                f'{self.args.n_embd}_{self.args.n_head}_{self.args.n_layer}_{self.args.loss_function}')
            print(f'Creating model\nThe model is to be saved as:\n{self.modelname}\n')
            print(f'Dataset Use --> training from scratch on a new model')

        else:
            self.dataname = self.args.data_name
            extname = 'r' if self.args.init_type=='resume' else f'ft{self.dataname}'
            self.modelname = f'{self.args.model_name}_{extname}'
            print(f'Loading statedict of existing model:\n{self.modelpath}\n')
            print(f'Dataset Use --> {self.args.init_type} on an existing model\noriginally trained with dataset {self.dataname}')

            exsdataset = torch.load(f'{self.modelpath}/{modelname}', map_location=self.device)
            gptconf = Config(**exsdataset["modelargs"])
            self.model = TSTransformer(gptconf)
            self.optimizer.load_state_dict(exsdataset['optimizer'])
            self.scheduler.load_state_dict(exsdataset['scheduler'])
            self.model.load_state_dict(exsdataset['model'])
            self.modelargs = exsdataset['modelargs']
            self.iter = exsdataset['iternum']
            self.settime(exsdataset['traintime'])
            self.training_loss_list = exsdataset['trloss']
            self.validation_loss_list = exsdataset['valloss']
            self.best_validation_loss = exsdataset['bvalloss']
            print(f'\nModel initialized from checkpoint')

        self.model.to(self.device)
        print(f'Using {self.optimizer.__class__.__name__} optimizer with {self.scheduler.__class__.__name__} scheduling')

    def configure_dataset(self, eval=False):
        """
        Function to configure training model, optimizer and other relevant parameters, simple abstraction.
        If model name is present, the specified model is either finetuned with another specified model,
        or the state dictionary is loaded and training is resumed with the original dataset.
        Eval --> training/testing
        """

        if eval==False:
            train_dataset = TensorDataset(self.gendict['control'],self.gendict['position']) 
            split_ratio = 0.8
            train_size = int(split_ratio * len(train_dataset))
            valid_size = len(train_dataset) - train_size
            train_ds, val_ds = random_split(train_dataset, [train_size, valid_size])

            self.training_dataset = DataLoader(train_ds, 
                                    batch_size=self.args.training_batch_size, 
                                    shuffle=True,
                                    pin_memory=True, num_workers=10)
            self.validation_dataset = DataLoader(val_ds, 
                                    batch_size=self.args.validation_batch_size, 
                                    shuffle=True,
                                    pin_memory=True, num_workers=10)
        elif eval==True:
            test_dataset = TensorDataset(self.gendict['control'],self.gendict['position'])
            self.test_dataset = DataLoader(test_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           pin_memory=True, num_workers=10) 
            
    def load(self, data, eval=False):
        """
        Loads control action used in the dataset from the respective .pt file, control action may belong
        to imposed typology or osc typology, in either case the program functions the same. The loaded
        action may then be augmented with the action derrivative or the link mass vector depending on 
        the proposed system identification methodology.
        *
        Derivatives are either computed through finite difference or autodiff on loaded dataset
        Mass Vectors are obtained from loaded dataset
        Control Actions are obtained from loaded dataset
        End Effector Trajectories are obtained from loaded dataset
        """
        datapath = self.traindatapath if eval==False else self.testdatapath
        actdict = torch.load(Path(f'{datapath}/{data}'))
        
        self.control = torch.movedim(actdict['control_action'][1:,:,:7],-2,-3).to('cpu').detach()
        self.position = torch.movedim(actdict['position'],-2,-3).to('cpu').detach()
        if self.args.include_mass_vectors:
            self.mass = torch.movedim(actdict['mass_vector'],-2,-3).to('cpu').detach()
        if self.args.include_control_diffs:
            self.diff = torch.movedim(actdict['control_diff'],-2,-3).to('cpu').detach()
        
        self.gendict = {
            "control" : self.control,
            "position" : self.position,
            "diff" : self.diff,
            "mass" : self.mass
        }
    
    def loadtoydataset(self):
        """
        Loads the toy linear, nonlinear datasets originally from
        ''
        used in testing the immediate capabilities of models
        """ 
        train_dataset = LinearDynamicalDataset(nx=7, nu=7, ny=14, seq_len=800)

        self.training_dataset = DataLoader(train_dataset, batch_size=8, num_workers=10)

        validation_dataset = LinearDynamicalDataset(nx=7, nu=7, ny=14, seq_len=800)
        self.validation_dataset = DataLoader(validation_dataset, batch_size=8, num_workers=10)


    def setcheckpoint(self, iter, curtime):
        """
        Defines the checkpoint dictionary pertaining to every iteration
        """
        self.checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'modelargs': self.modelargs,
            'iternum': iter,
            'traintime': curtime-self.starttime,
            'trloss': self.training_loss_list,
            'valloss': self.validation_loss_list,
            'bvalloss': self.best_validation_loss,
            'args': self.args
            }
        self.iter = iter
    
    def normalizestd(self, x):
        """
        Normalizes batch tensors by zero mean of a simulation to be fed into the training module
        """
        xmean = x.mean(dim=1, keepdim=True)
        xstd = x.std(dim=1, keepdim=True)
        nx = (x - xmean) / xstd

        return nx, xmean, xstd
    
    def denormalizestd(self, nx, xmean, xstd):
        """
        Denormalizes batch tensors by zero mean of a simulation for assessment in validation and testing phases
        """
        return  nx*xstd + xmean
    
    def normalizelin(self, x):
        """
        Normalizes batch tensors by linear scaling of a simulation to be fed into the training module
        """
        xmax,i = x.max(dim=1, keepdim=True)
        xmin,j = x.min(dim=1, keepdim=True)
        nx = (x - xmin)/(xmax - xmin)

        return nx, xmax, xmin
    
    def denormalizelin(self, nx, xmax, xmin):
        """
        Denormalizes batch tensors by linear scaling of a simulation for assessment in validation and testing phases
        """
        return  nx*(xmax - xmin) + xmin
    
    def normalizecfd(self, x):
        """
        Normalizes batch tensors by cfd scaling of a simulation for assessment in validation and testing phases
        """
        pass

    def denormalizecfd(self, x):
        """
        Denormalizes batch tensors by cfd scaling of a simulation for assessment in validation and testing phases
        """
        pass
        
    def seperate_context(self,x):
        """
        Seperates the u,y batch pairs into given (context) and estimated parts according to a defined
        seq length, currently this seperation is only available for the transformer identification
        procedures
        """
        xctx = x[:, :self.sqlctx, :]
        xnew = x[:, self.sqlctx:, :]

        return xctx, xnew

    def cosine_annealing(self):
        """
        Cosine annealed learning rate, may be deprecated
        """
        self.scheduler = CosineAnnealingWarmRestarts(optimizer=self.optimizer, 
                                                     T_0=self.WARMUP_ITERATIONS,
                                                     eta_min=self.args.learning_rate/100,
                                                     last_epoch=-1)
    
    def step_decay(self):
        """
        Defines learning rate --> Step Decay
        """
        self.scheduler = StepLR(optimizer=self.optimizer,
                                gamma=self.GAMMA,
                                step_size=self.STEPSIZE,
                                last_epoch=-1)

    def exponential_decay(self):
        """
        Defines learning rate --> Exponential Decay
        """
        self.scheduler = ExponentialLR(optimizer=self.optimizer, 
                                       gamma=self.GAMMA,
                                       last_epoch=-1)
    
    def constant_lr(self):
        """
        Defines learning rate --> Constant
        """
        self.scheduler = ConstantLR(optimizer=self.optimizer,
                                    factor=1.0,
                                    total_iters=self.TOTAL_ITERATIONS,
                                    last_epoch=-1)

    def configure_learning_rate(self):
        """
        Defines learning rate --> Warmup Cosine Annealing, Step Decay
                                  Exponential Decay, Constant
        """
        if self.args.cosine_annealing:
            self.cosine_annealing()
        elif self.args.exponential_decay:
            self.exponential_decay()
        elif self.args.step_decay:
            self.step_decay()
        else:
            self.constant_lr()




