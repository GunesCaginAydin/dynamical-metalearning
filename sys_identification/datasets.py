import torch
import math
import numpy as np
import copy
from matplotlib import pyplot as plt
from functools import partial
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingWarmRestarts, ExponentialLR, StepLR, LinearLR, SequentialLR
from scipy import interpolate

import datetime
import os
import json
<<<<<<< HEAD
=======
import csv
>>>>>>> 295079b1ef7ee43dfe067a603b62467db507a76a
from pathlib import Path
try:
    from architectures.transformer.transformer_sim import Configtf, TSTransformer
    from architectures.diffuser.diffuser_sim import Configdf, TSDiffuser
    from architectures.receding_horizon.receding_horizon import configrhdf, configrhtf, RHDiffusionUNet, RHDiffusionTrf
    from architectures.receding_horizon.rechor_utils import EMAModel
except:
    from Sys_Identification.architectures.transformer.transformer_sim import Configtf, TSTransformer # type: ignore
    from Sys_Identification.architectures.diffuser.diffuser_sim import Configdf, TSDiffuser # type: ignore
    from Sys_Identification.architectures.receding_horizon.receding_horizon import configrhdf, configrhtf, RHDiffusionUNet, RHDiffusionTrf # type: ignore
    from Sys_Identification.architectures.receding_horizon.rechor_utils import EMAModel # type: ignore

class cfg():
    JOINTS=7
    COORDINATES=14

    ITER=1000
    CTX=200
    
    SHOT=0
    

class dataset(cfg):
    """
    Dataset object that utilizes any custom dataset with the prescribed data format to be imposed
    in the training regime. The data format is:

        control: input trajectory: MS-CH-VS-FS-FC | 7-9 D
        position: output trajectory: | 13-14-16 D 
        target: target trajectory if controller implemented 7 D
        gains (kp,kd,ki): controller gains if controller implemented 1-6-9 D
        rigidbodyprops: rigid body properties of each asset
        
    Once the dataset object is instantiated, the getdata() method can be used to obtain the metadata
    of a certain dataset whereas the getgen() method can be used to access the specific generation files
    inside the specified dataset. When using getgen() it is important to also call delgen() to remove and 
    clean generation data in each iteration over a dataset including multitudes of generation data.

    Get/Set methods for dataset properties such as:

        metadata: generation time, environment count, etc. | .json
        generation: input/output tensors | .pt
        dataset: training/validation/test tensors
        model: transformer/diffuser/receding_horizon
        losslist: training/validation/best_loss list
        lrlist: learning rate propogation
        time: current time of training

    These properties pertain to every loop and should be called and cast back to the dataset object when
    the loop is over / at the end of each iteration

    Properties of transformer:

        n_layer (int) : number of MLP layers
        n_head (int) : number of transformer heads
        n_embd (int) : number of transformer embeddings
        n_y (int) : estimation dimension
        n_u (int) : action dimension
        seq_len_ctx (int) : context length
        seq_len_new (int) : non-context length
        bias (bool) : if True include bias in neural computations
        dropout (float) : dropout amount
        trftype (int) : 1 for torque as action, 2 for tracking trajectory as action

    Properties of diffuser:

        timesteps (int) : number of denoising timesteps
        hidden (int) : number of Unet layers
        multiplier (int) : number of Unet blocks
        attention (bool) : if True implement attention mechanisms into Unet
        ucond (bool) : if True inpaint action in all diffusion timesteps
        ycond (bool) : if True inpaint trajectory in all diffusion timesteps
        epsilon (bool) : estimation type - epsilon or xstart
        loss_weight (float) : loss weighting in trajectory observations until context length
        n_y (int) : estimation dimension
        n_u (int) : action dimension
        seq_len_ctx (int) : context length
        seq_len_new (int) : non-context length
        diftype (int) : 1 for torque as action, 2 for tracking trajectory as action

    Properties of receding horizon Unet:

        n_y (int) : estimation dimension
        n_u (int) : action dimension
        seq_len_ctx (int) : context length
        seq_len_new (int) : non-context length
        prediction_type (str) : estimation type - epsilon or xstart
        timesteps (int) : number of denioising timesteps
        hidden (int) : number of Unet layers
        multiplier (int) : number of Uner blocks
        lobscond (bool) : local conditioning of actions
        gobscond (bool) : global conditioning of actions
        rechortype (int) : 1 for at, 2 for a, 3 for t, 4 for inpainting, 5 for tracking t, 6 for controller gains
        controlgain_horizon (int) : horizon of control gain input for controller implemented

    Properties of receding horizon Transformer_

        n_y (int) : estimation dimension
        n_u (int) : action dimension
        seq_len_ctx (int) : context length
        seq_len_new (int) : non-context length
        prediction_type (str) : estimation type - epsilon or xstart
        timesteps (int) : number of denioising timesteps
        heads (int) : number of attention heads
        layers (int) : number of MLP layers
        embeddings (int) : number of transformer embeddings
        causality (bool) : if True mask attention layers for causal inference
        timecond (bool) : time conditioning
        obscond (bool) : conditioning of actions
        rechortype (int) : 1 for at, 2 for a, 3 for t, 4 for inpainting, 5 for tracking t, 6 for controller gains
        controlgain_horizon (int) : horizon of control gain input for controller implemented

    Parameters
    ---

    args (dict): arguments parsed by the user
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
        self.trainpath = ''
        self.testpath = ''
        self.metapath = ''

        self.modelargs = {}
        self.datadict = {}
        self.checkpoint = {}
        self.modelname = ''
        self.dataname = ''

        self.trainlist = []
        self.testlist = []
        self.modellist = []
        self.learningrate_list = []

        self.learning_rate = args.learning_rate 
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.ema = None
        self.ema_model = None

        self.training_dataset = torch.empty(0)
        self.current_training_loss = np.inf
        self.training_loss_list = [np.inf]

        self.validation_dataset = torch.empty(0)
        self.current_validation_loss = np.inf
        self.validation_loss_list = [np.inf]
        self.best_validation_loss = [np.inf, np.inf]

        self.test_dataset = torch.empty(0)


        self.control = torch.empty(0)
        self.position = torch.empty(0)
        self.masses = torch.empty(0)
        self.coms = torch.empty(0)
        self.inertias = torch.empty(0)
        self.target = torch.empty(0)
        self.kp = torch.empty(0)
        self.kv = torch.empty(0)
        self.ki = torch.empty(0)       
        self.gendict = {
            "control" : self.control,
            "position" : self.position,
            "masses" : self.masses,
            "coms" : self.coms,
            "inertias" : self.inertias,
            "target" : self.target,
            "kp" : self.kp,
            "kv" : self.kv,
            "ki" : self.ki  
        }

        self.ny = args.out_dimension
        self.nu = args.in_dimension
        self.sql = args.total_sim_iterations
        self.sqlctx = int(self.args.context/100 * self.sql)
        self.sqlpar = self.sql - self.sqlctx

        if self.args.subcommand=='transformer': # transformer init
            self.modelargstf = {
                "n_layer" : None, 
                "n_head" : None, 
                "n_embd" : None, 
                "n_y" : None, 
                "n_u" : None,
                "seq_len_ctx" : None, 
                "seq_len_new" : None,
                "bias" : None, 
                "dropout" : None,
                "trftype" : None
            }
        if self.args.subcommand=='diffuser': # diffuser init
            self.modelargsdf = {
                "n_y" : None, 
                "n_u" : None,
                "seq_len_ctx" : None, 
                "seq_len_new" : None,
                "timesteps" : None,
                "hidden" : None,
                "multiplier" : None,
                "attention" : None,
                "ucond" : None,
                "ycond" : None,
                "epsilon" : None,
                "loss_weight" : None,
                "diftype" : None
            }
        if self.args.subcommand=='rechorUnet': # receding horizon Unet init
            self.modelargsrhdf = {
                "n_y" : None, 
                "n_u" : None,
                "seq_len_ctx" : None, 
                "seq_len_new" : None,
                "prediction_type" : None,
                "timesteps" : None,
                "hidden" : None,
                "multiplier" : None,
                "lobscond" : None,
                "gobscond" : None,
                "rechortype" : None,
                "controlgain_horizon" : None
            }
        if self.args.subcommand=='rechorTrf': # receding horizon Transformer init
            self.modelargsrhtf = {
                "n_y" : None, 
                "n_u" : None,
                "seq_len_ctx" : None, 
                "seq_len_new" : None,
                "timesteps" : None,
                "prediction_type" : None,
                "heads" : None,
                "layers" : None,
                "embeddings" : None,
                "causality" : None,
                "timecond" : None,
                "obscond" : None,
                "rechortype" : None,
                "controlgain_horizon" : None
            }
        self.iter = 0

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
            self.test_dataset,
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
    
    
    def setema(self,
               ema_,
               ema_model_):
        self.ema = ema_
        self.ema_model = ema_model_

    def getema(self):
        return self.ema, self.ema_model
    

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
        if training_loss is not None:
            self.training_loss_list.append(training_loss)
            self.current_training_loss = training_loss
        
        if validation_loss is not None:
            self.validation_loss_list.append(validation_loss)
            self.current_validation_loss = validation_loss

        if best_validation_loss is not None:
            self.best_validation_loss[0] = self.best_validation_loss[1]
            self.best_validation_loss[1] = best_validation_loss


    def getlrlist(self):
        return self.learningrate_list
    
    def setlrlist(self,
                  lr=None):
        self.learningrate_list.append(lr)


    def reset(self):
        torch.cuda.empty_cache()
        self.gendict = {}
        self.training_dataset = torch.empty(0)
        self.validation_dataset = torch.empty(0)
        

    def resolve_datasets(self, eval=False, modelfolder=False, modeldir=False, metadir=False, traindir=False, testdir=False):
        """
        Function to resolve model characteristics and parameters as well as script particularities
        during training and testing. For training on Colab, set modeldir, metadir and traindir to
        the respective drive locations.

        Parameters
        ---
            eval (bool): if False training, else testing
            modelfolder (string) : folder to use train/eval data
            modeldir (string): required for colab training, default False
            metadir (string): required for colab training, default False
            traindir (string): required for colab training, default False
            testdir (string): required for colab testing, default False
        """
        if not self.args.seed:
            self.seed = torch.randint(low=0,high=99999,size=(1,))
        else:
            self.seed = self.args.seed
        torch.manual_seed(seed=self.seed)

        parentpath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

        traindatadir = f'data_generation/data_tensors/train/{self.args.data_name}' if not traindir else traindir
        self.trainpath = os.path.join(parentpath,traindatadir) # TRAINING/TESTING
        self.trainlist = os.listdir(self.trainpath)
        print(f'Training data is acquired from:\n{self.trainpath}\n')

        metadir = f'data_generation/data_objects/{self.args.data_name}.json' if not metadir else metadir
        self.metapath = os.path.join(parentpath,metadir) # TRAINING/TESTING
        with open(self.metapath, 'r') as f:
            metadata  = json.load(f)
        totalsims = len(metadata["dataname"])
        totalenvs = sum(metadata["genenvs"])
        self.trainlist = metadata["genname"] if self.trainlist==[] else self.trainlist
        timetaken = datetime.timedelta(seconds=round(sum(metadata["gentime"])))
        print(f'Metadata is acquired from:\n{self.metapath}\n')
        self.totaliters = int(totalenvs/self.args.training_batch_size)+1
        
        if not eval: # TRAINING 
            modeldir = f'sys_identification/models' if not modeldir else modeldir
            modelpath = os.path.join(parentpath,modeldir)
            Path(modelpath).mkdir(exist_ok=True)
            modelspecific = f'{modeldir}/{self.args.data_name}' if not modelfolder else f'{modeldir}/{modelfolder}'
            self.modelpath = os.path.join(parentpath,modelspecific)
            Path(self.modelpath).mkdir(exist_ok=True)
            print(f'Trained model is to be saved to:\n{self.modelpath}\n')
            print(f'Estimated iterations: {self.totaliters}')
        elif eval: # TESTING
            modeldir = f'sys_identification/models' if not modeldir else modeldir
            modelpath = os.path.join(parentpath,modeldir)
            modelspecific = f'{modeldir}/{self.args.data_name}' if not modelfolder else f'{modeldir}/{modelfolder}'
            self.modelpath = os.path.join(parentpath,modelspecific)
            print(f'Test models are acquired from to:\n{self.modelpath}\n')
            self.modellist = os.listdir(self.modelpath)

            testdatadir = f'data_generation/data_tensors/test/{self.args.test_name}' if not testdir else testdir
            self.testpath = os.path.join(parentpath,testdatadir)
            for dirpath, dirnames, filenames in os.walk(top=self.testpath):
                for filename in filenames:
<<<<<<< HEAD
                    if filename.endswith('.pt'):
=======
                    if filename.endswith('.pt') or filename.endswith('.csv'):
>>>>>>> 295079b1ef7ee43dfe067a603b62467db507a76a
                        self.testlist.append(filename)
            print(f'Test data is acquired from:\n{self.testpath}\n')

            print(f'Over {len(self.modellist)} different models trained on {self.args.data_name}')
            print(f'{len(self.testlist)} different datasets are to be tested.\nAvailable models and test datasets are:\n')
            print(f'Models-->\n{self.modellist}\n\nTest Datasets-->\n{self.testlist}\n')

        self.datadict = {
            "trainlist" : self.trainlist,
            "testlist" : self.testlist,
            "modellist" : self.modellist,
            "tsims" : totalsims,
            "tenvs" : totalenvs,
            "time" : timetaken,
            "mpath" : self.modelpath,
            "dpath" : self.trainpath,
            "epath" : self.testpath
            }   
        
        print(f'Will use {self.device} for the training/testing\n')

    def download_datasets(self, *args, **kwargs):
        """
        Mounts a certain dataset from Google Drive, used in Google Colab instead of resolve_datasets.
            datasets.download_datasets()
            for model in models:
                initialize_model()
                ...

            traindir: content/drive/MyDrive/Thesis/Datasets/Train/Data_Tensors
            metadir: content/drive/MyDrive/Thesis/Datasets/Train/Data_Objects
            modeldir: content/drive/MyDrive/Thesis/Models
            testdir: content/drive/MyDrive/Thesis/Test'
        """
        self.resolve_datasets(*args,
                              **kwargs,
                              traindir=f'content/drive/MyDrive/Thesis/Datasets/Data_Tensors/Train/{self.args.data_name}',
                              metadir=f'content/drive/MyDrive/Thesis/Datasets/Data_Objects/{self.args.data_name}.json',
                              modeldir=f'content/drive/MyDrive/Thesis/Models',
                              testdir=f'content/drive/MyDrive/Thesis/Datasets/Data_Tensors/Test/{self.args.test_name}')

    def initialize_model(self, modelname=None, modeltype=None, modelfolder=None, shot=None,
                               horizon = None, ctxlen=None, seqlen=None, timesteps=None):
        """
        Defines the model that is to be loaded, used and saved at the end of training procedure. Models
        may be created from scratch, or an existing model may be imported to resume its training or 
        finetune on another dataset that the model was originally trained on.
        
        Parameters
        ---
            modelname (string): if given obtain model from checkpoint (finetuning/testing)
            modeltype (string): transformer | diffuser | rechorUnet | rechorTrf if modeltype is not included in modelname
            modelfolder (string) : if given obtain model from modelfolder 
            shot (int) : few shot accuracy testing - DEPRECATED
            ctxlen (int) : context length of training / inference
            seqlen (int) : non-context length of training / inference
            horizon (int) : total length of horizon
            timesteps (int) : denoising inference timesteps - only for diffusion pipeline

        """
        if modelname is None: # NO MODEL GIVEN, CREATE MODEL --- TRAINING
            if self.args.subcommand=='transformer': 
                self.modelargstf = {
                    "n_layer" : self.args.n_layer, 
                    "n_head" : self.args.n_head, 
                    "n_embd" : self.args.n_embd, 
                    "n_y" : self.ny, 
                    "n_u" : self.nu,     
                    "seq_len_ctx" : self.sqlctx, 
                    "seq_len_new" : self.sqlpar,
                    "bias" : self.args.bias, 
                    "dropout" : self.args.dropout,
                    "trftype" : self.args.trftype
                }
                self.modelargs = self.modelargstf
                gptconf = Configtf(**self.modelargs)
                self.model = TSTransformer(gptconf)
                self.optimizer = self.model.configure_optimizers(self.args.weight_decay, 
                                            self.args.learning_rate, 
                                            (self.args.beta1, self.args.beta2), 
                                            self.device)
                self.configure_learning_rate()  
            elif self.args.subcommand=='diffuser':
                self.modelargsdf = {
                    "n_y" : self.ny, 
                    "n_u" : self.nu,
                    "seq_len_ctx" : self.sqlctx, 
                    "seq_len_new" : self.sqlpar,
                    "timesteps" : self.args.timesteps,
                    "hidden" : self.args.hidden,
                    "multiplier" : self.args.multiplier,
                    "attention" : self.args.attention,
                    "ucond" : self.args.input_condition,
                    "ycond" : self.args.output_condition,
                    "epsilon" : self.args.predict_epsilon,
                    "loss_weight" : self.args.loss_weight,
                    "diftype" : self.args.diftype
                }
                self.modelargs = self.modelargsdf
                dfconf = Configdf(**self.modelargsdf)
                self.model = TSDiffuser(dfconf)
                self.optimizer = self.model.model.configure_optimizers(self.args.weight_decay, 
                                            self.args.learning_rate, 
                                            (self.args.beta1, self.args.beta2), 
                                            self.device)
                self.configure_learning_rate()
            elif self.args.subcommand=='rechorUnet':
                self.modelargsrhdf = {
                    "n_y" : self.ny, 
                    "n_u" : self.nu,
                    "seq_len_ctx" : self.sqlctx, 
                    "seq_len_new" : self.sqlpar,
                    "timesteps" : self.args.timesteps,
                    "prediction_type" : self.args.prediction_type,
                    "hidden" : self.args.hidden,
                    "multiplier" : self.args.multiplier,
                    "lobscond" : bool(self.args.local_obscond),
                    "gobscond" : bool(self.args.global_obscond),
                    "rechortype" : int(self.args.rechortype),
                    "controlgain_horizon" : self.args.controlgain_horizon
                }
                self.modelargs = self.modelargsrhdf
                rhdfconf = configrhdf(**self.modelargsrhdf)
                self.model = RHDiffusionUNet(rhdfconf)
                self.optimizer = self.model.model.configure_optimizers(self.args.weight_decay, 
                                            self.args.learning_rate, 
                                            (self.args.beta1, self.args.beta2), 
                                            self.device)
                self.configure_learning_rate()
                if self.args.include_ema:
                    self.ema_model = copy.deepcopy(self.model)
                    self.ema = EMAModel(update_after_step=0,
                                        inv_gamma=1.0,
                                        power=0.75,
                                        min_value=0.0, max_value=0.9999,
                                        model=self.ema_model).to(self.device)
            elif self.args.subcommand=='rechorTrf':
                self.modelargsrhtf = {
                    "n_y" : self.ny, 
                    "n_u" : self.nu,
                    "seq_len_ctx" : self.sqlctx, 
                    "seq_len_new" : self.sqlpar,
                    "timesteps" : self.args.timesteps,
                    "prediction_type" : self.args.prediction_type,
                    "heads" : self.args.heads,
                    "layers" : self.args.layers,
                    "embeddings" : self.args.embeddings,
                    "causality" : bool(self.args.causality),
                    "timecond" : bool(self.args.timecond),
                    "obscond" : bool(self.args.obscond),
                    "rechortype" : int(self.args.rechortype),
                    "controlgain_horizon" : self.args.controlgain_horizon
                }
                self.modelargs = self.modelargsrhtf
                rhtfconf = configrhtf(**self.modelargsrhtf)
                self.model = RHDiffusionTrf(rhtfconf)
                self.optimizer = self.model.model.configure_optimizers(self.args.weight_decay, 
                                            self.args.learning_rate, 
                                            (self.args.beta1, self.args.beta2), 
                                            self.device)
                self.configure_learning_rate()
                if self.args.include_ema:
                    self.ema_model = copy.deepcopy(self.model)
                    self.ema = EMAModel(update_after_step=0,
                                        inv_gamma=1.0,
                                        power=0.75,
                                        min_value=0.0, max_value=0.9999,
                                        model=self.ema_model).to(self.device)

            self.dataname = self.args.data_name
            if self.args.save_name is None: # if no finalname is given, create by default
                if self.args.subcommand=='transformer': # TRANSFORMER MODEL NAME
                    self.modelname = (f'{self.args.subcommand}_{self.args.data_name}_{round(self.args.context)}_{self.args.training_batch_size}_'
                    f'{self.args.n_embd}_{self.args.n_head}_{self.args.n_layer}_{self.args.loss_function}')
                elif self.args.subcommand=='diffuser': # DIFFUSER MODEL NAME
                    att = 1 if self.args.attention else 0
                    uc = 1 if self.args.input_condition else 0
                    yc = 1 if self.args.output_condition else 0
                    eps = 1 if self.args.predict_epsilon else 0
                    self.modelname = (f'{self.args.subcommand}_{self.args.data_name}_{round(self.args.context)}_{self.args.training_batch_size}_'
                    f'{self.args.hidden}_{self.args.multiplier}_{self.args.timesteps}_{att}_{uc}_{yc}_{eps}_{self.args.loss_weight}_{self.args.loss_function}')
                elif self.args.subcommand=='rechorUnet': # RECEDING HORIZON UNET MODEL NAME
                    lc = 1 if self.args.local_obscond else 0
                    gc = 1 if self.args.global_obscond else 0
                    self.modelname = (f'{self.args.subcommand}_{self.args.data_name}_{round(self.args.context)}_{self.args.training_batch_size}_'
                    f'{self.args.hidden}_{self.args.multiplier}_{self.args.timesteps}_{lc}_{gc}_{self.args.rechortype}_{self.args.loss_function}')
                elif self.args.subcommand=='rechorTrf': # RECEDING HORIZON TRANSFORMER MODEL NAME
                    tc = 1 if self.args.timecond else 0
                    oc = 1 if self.args.obscond else 0
                    att = 1 if self.args.causality else 0
                    self.modelname = (f'{self.args.subcommand}_{self.args.data_name}_{round(self.args.context)}_{self.args.training_batch_size}_'
                    f'{self.args.embeddings}_{self.args.heads}_{self.args.layers}_{self.args.timesteps}_{tc}_{oc}_{att}_{self.args.rechortype}_{self.args.loss_function}')
            else:
                self.modelname = self.args.save_name # if finalname is given, use that
            print(f'Creating model\nThe model is to be saved as:\n{self.modelname}\n')
            print(f'Dataset Use --> training from scratch on a new model')

        else: # MODEL IS ALREADY CREATED --- FINETUNING | TESTING
            self.dataname = self.args.data_name
            extname = 'r' if self.args.init_type=='resume' else f'ft{self.dataname}'
            if self.args.save_name is None:
                self.modelname = f'{modelname}_{extname}'
            else:
                self.modelname = self.args.save_name

            if self.modelname in os.listdir(self.modelpath):
                self.modelname = f'{self.modelname}_repeat'

            print(f'Loading statedict of existing model:\n{self.modelpath}\n')
            print(f'Dataset Use --> {self.args.init_type} on an existing model\n\n')

            exsdataset = torch.load(f'{self.modelpath}/{modelname if modelfolder is None else modelfolder}', map_location=self.device, weights_only=False)
            if shot: # inference time context changes for fewshot tests
                exsdataset['modelargs']['seq_len_ctx'],exsdataset['modelargs']['seq_len_new'] = 1000*shot + self.sqlctx, 1000*(shot+1) - (1000*shot + self.sqlctx)

            if ctxlen is not None and seqlen is not None: # inference time context changes
                exsdataset['modelargs']['seq_len_ctx'],exsdataset['modelargs']['seq_len_new'] = ctxlen,seqlen

            if timesteps is not None: # inference time denoising timesteps changes
                exsdataset['modelargs']['timesteps'] = timesteps

            if self.args.subcommand=='transformer' or str.split(modelname,'_')[0]=='transformer' or modeltype=='transformer': # MODELNAME==TRANSFORMER
                gptconf = Configtf(**exsdataset["modelargs"])
                self.model = TSTransformer(gptconf).to(self.device)
                self.modelargs = exsdataset['modelargs']
                self.optimizer = self.model.configure_optimizers(self.args.weight_decay, 
                                            self.args.learning_rate, 
                                            (self.args.beta1, self.args.beta2), 
                                            self.device)
                self.configure_learning_rate()
            elif self.args.subcommand=='diffuser' or str.split(modelname,'_')[0]=='diffuser' or modeltype=='diffuser': # MODELNAME==DIFFUSER
                dfconf = Configdf(**exsdataset["modelargs"])
                self.model = TSDiffuser(dfconf).to(self.device)
                self.modelargs = exsdataset['modelargs']
                self.optimizer = self.model.model.configure_optimizers(self.args.weight_decay, 
                                            self.args.learning_rate, 
                                            (self.args.beta1, self.args.beta2), 
                                            self.device)
                self.configure_learning_rate()
            elif self.args.subcommand=='rechorUnet' or str.split(modelname,'_')[0]=='rechorUnet' or modeltype=='rechorUnet': # MODELNAME==RECHORUNET
                rhdfconf = configrhdf(**exsdataset["modelargs"])
                self.model = RHDiffusionUNet(rhdfconf).to(self.device)
                self.modelargs = exsdataset['modelargs']
                self.optimizer = self.model.model.configure_optimizers(self.args.weight_decay, 
                                            self.args.learning_rate, 
                                            (self.args.beta1, self.args.beta2), 
                                            self.device)
                self.configure_learning_rate()
                if self.args.include_ema:
                    self.ema_model = copy.deepcopy(self.model)
                    self.ema = EMAModel(update_after_step=0,
                                        inv_gamma=1.0,
                                        power=0.75,
                                        min_value=0.0, max_value=0.9999,
                                        model=self.ema_model).to(self.device)
            elif self.args.subcommand=='rechorTrf' or str.split(modelname,'_')[0]=='rechorTrf' or modeltype=='rechorTrf': # MODELNAME==RECHORTRF
                rhtfconf = configrhtf(**exsdataset["modelargs"])
                self.model = RHDiffusionTrf(rhtfconf).to(self.device)
                self.modelargs = exsdataset['modelargs']
                self.optimizer = self.model.model.configure_optimizers(self.args.weight_decay, 
                                            self.args.learning_rate, 
                                            (self.args.beta1, self.args.beta2), 
                                            self.device)
                self.configure_learning_rate()
                if self.args.include_ema:
                    self.ema_model = copy.deepcopy(self.model)
                    self.ema = EMAModel(update_after_step=0,
                                        inv_gamma=1.0,
                                        power=0.75,
                                        min_value=0.0, max_value=0.9999,
                                        model=self.ema_model).to(self.device)
            
            self.model.load_state_dict(exsdataset['model'])
            self.iter = exsdataset['iternum']
            self.settime(exsdataset['traintime'])
            self.training_loss_list = exsdataset['trloss']
            self.validation_loss_list = exsdataset['valloss']
            self.best_validation_loss = exsdataset['bvalloss']
            print(f'\nModel initialized from checkpoint')

        self.model.to(self.device)
        print(f'Using {self.optimizer.__class__.__name__} optimizer with {self.scheduler.__class__.__name__} scheduling')

    def configure_datasets(self, eval=False):
        """
        Function to configure training model, optimizer and other relevant parameters, simple abstraction.
        If model name is present, the specified model is either finetuned with another specified model,
        or the state dictionary is loaded and training is resumed with the original dataset.

        if controller:
            dataset -> control, position, target, kp, kv, ki
        else:
            dateset -> control, position
        
        Parameters
        ---
            eval (bool): if false training/validation datasets are created with a predetermined batchsize, if true
            testing datasets are created with batchsize=1
        """

        if eval==False: # TRAINING
            if self.args.controller:
                train_dataset = TensorDataset(self.gendict['control'],self.gendict['position'],self.gendict['target'],
                                              self.gendict['kp'],self.gendict['kv'],self.gendict['ki'])
            else:
                train_dataset = TensorDataset(self.gendict['control'],self.gendict['position'])

            split_ratio = 0.8
            train_size = int(split_ratio * len(train_dataset))
            valid_size = len(train_dataset) - train_size
            train_ds, val_ds = random_split(train_dataset, [train_size, valid_size])

            self.training_dataset = DataLoader(train_ds, 
                                    batch_size=self.args.training_batch_size, 
                                    shuffle=False,
                                    pin_memory=True, num_workers=10)
            self.validation_dataset = DataLoader(val_ds, 
                                    batch_size=self.args.validation_batch_size, 
                                    shuffle=False,
                                    pin_memory=True, num_workers=10)
        elif eval==True: # TESTING
            if self.args.controller:
                test_dataset = TensorDataset(self.gendict['control'],self.gendict['position'],self.gendict['target'],
                                              self.gendict['kp'],self.gendict['kv'],self.gendict['ki'])
            else:
                test_dataset = TensorDataset(self.gendict['control'],self.gendict['position'])
            self.test_dataset = DataLoader(test_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           pin_memory=True, num_workers=10) 
            
<<<<<<< HEAD
    def load(self, data, eval=False, itr=1000):
=======
    def load(self, data, eval=False, itr=1000, real=False):
>>>>>>> 295079b1ef7ee43dfe067a603b62467db507a76a
        """
        Loads control action used in the dataset from the respective .pt file, control action may belong
        to imposed typology or osc typology, in either case the program functions the same. The loaded
        action may then be augmented with the action derrivative or the link mass vector depending on 
        the proposed system identification methodology.

        Derivatives are either computed through finite difference or autodiff on loaded dataset
        Mass Vectors are obtained from loaded dataset
        Control Actions are obtained from loaded dataset
        End Effector Trajectories are obtained from loaded dataset

        Parameters
        ---
            data (string): dataset to be imported
            eval (bool): if true import testing dataset
            itr (int): number of iterations to store the data until
        """
        datapath = self.trainpath if eval==False else self.testpath
<<<<<<< HEAD
        actdict = torch.load(Path(f'{datapath}/{data}'), weights_only=False)
        
        self.control = torch.movedim(actdict['control_action'][1:itr+1,:,:7],-2,-3).to('cpu').detach()
        self.position = torch.movedim(actdict['position'][:itr,:,:],-2,-3).to('cpu').detach()
=======

        if not real:
            actdict = torch.load(Path(f'{datapath}/{data}'), weights_only=False)
            
            self.control = torch.movedim(actdict['control_action'][1:itr+1,:,:7],-2,-3).to('cpu').detach()
            self.position = torch.movedim(actdict['position'][:itr,:,:],-2,-3).to('cpu').detach()
        else:
            rowl = torch.empty(size=(0,21))
            with open(Path(f'{datapath}/{data}')) as f:
                data = csv.reader(f, delimiter=',')
                for i,row in enumerate(data):
                    if i==0:
                        continue
                    row = torch.tensor([float(x) for x in row]).unsqueeze(dim=0)
                    #if i%(int(1000/60))==1: ###
                    #    rowl = torch.cat((rowl,row),dim=0)
                    if i<=(itr+1):
                        rowl = torch.cat((rowl,row),dim=0)

            self.control = rowl[:,:7][1:itr+1,:].unsqueeze(0)
            self.position = rowl[:,7:][1:itr+1,:].unsqueeze(0)
>>>>>>> 295079b1ef7ee43dfe067a603b62467db507a76a

        if self.args.controller:
            self.masses = actdict['masses'].to('cpu').detach()
            self.coms = actdict['coms'].to('cpu').detach()
            self.inertias = actdict['inertias'].to('cpu').detach()

            self.target = torch.movedim(actdict['target'][:itr,:,:7],-2,-3).to('cpu').detach()

            self.kp = actdict['kp'].to('cpu').detach()
            self.kv = actdict['kd'].to('cpu').detach()
            self.ki = actdict['ki'].to('cpu').detach()

        self.gendict = {
            "control" : self.control,
            "position" : self.position,
            "masses" : self.masses,
            "coms" : self.coms,
            "inertias" : self.inertias,
            "target" : self.target,
            "kp" : self.kp,
            "kv" : self.kv,
            "ki" : self.ki
        }

<<<<<<< HEAD
=======

>>>>>>> 295079b1ef7ee43dfe067a603b62467db507a76a
    def setcheckpoint(self, iter, curtime):
        """
        Defines the checkpoint dictionary pertaining to every iteration.

        Parameters
        ---
        iter (int): current iteration of training
        curtime (float): current time of training
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
    
    def normalizestd(self, x, ndim=1):
        """
        Normalizes batch tensors by zero mean of a simulation to be fed into the training module. Can
        work with constant dimensions...

        Parameters
        ---
            x (torch.Tensor): tensor to be normalized
            ndim (int): normalization dimension

        Returns
        ---
            nx (torch.Tensor): normalized tensor
            xmean (torch.Tensor): mean along ndim
            xstd (torch.Tensor): std alond ndim
        """
        xmean = x.mean(dim=ndim, keepdim=True)
        xstd = x.std(dim=ndim, keepdim=True)
        nx = (x - xmean) / xstd

        if torch.any(torch.isnan(nx)):
            nx = (x - xmean) / (xstd + 1e-12)
        
        if nx.dim()==3:
            B,T,C = nx.shape
            for i in range(C):
                max,_ = torch.max(nx[...,i],dim=1)
                min,_ = torch.min(nx[...,i],dim=1)
                if torch.all(max==min) or torch.all(torch.isinf(nx[...,i])):
                    nx[...,i] = nx[...,i]

        return nx, (xmean, xstd)
    
    def denormalizestd(self, nx, aux):
        """
        Denormalizes batch tensors by zero mean of a simulation for assessment in validation and testing phases.
    
        Parameters
        ---
            nx (torch.Tensor): normalized tensor
            xmean (torch.Tensor): mean along ndim
            xstd (torch.Tensor): std alond ndim

        Returns
        ---
            x (torch.Tensor): denormalized tensor according to normalization parameters
        """
        return  nx*aux[1] + aux[0]
    
    def normalizelin(self, x, ndim=1):
        """
        Normalizes batch tensors by linear scaling of a simulation to be fed into the training module.
        
        Parameters
        ---
            x (torch.Tensor): tensor to be normalized
            ndim (int): normalization dimension

        Returns
        ---
            nx (torch.Tensor): normalized tensor   
            xmax (torch.Tensor): max along ndim   
            xmin (torch.Tensor): min alond ndim
        """
        xmax,i = x.max(dim=ndim, keepdim=True)
        xmin,j = x.min(dim=ndim, keepdim=True)
        nx = (x - xmin)/(xmax - xmin)

        nx = 2*nx - 1

        return nx, (xmax, xmin)
    
    def denormalizelin(self, nx, aux):
        """
        Denormalizes batch tensors by linear scaling of a simulation for assessment in validation and testing phases.

        Parameters
        ---
            nx (torch.Tensor): normalized tensor
            xmean (torch.Tensor): mean along ndim
            xstd (torch.Tensor): std alond ndim

        Returns
        ---
            x (torch.Tensor): denormalized tensor according to normalization parameters
        """
        nx = torch.clip(nx,-1,1)

        nx = (nx+1)/2

        return  nx*(aux[0] - aux[1]) + aux[1]
    
    def normalizecdf(self, x, ndim=1):
        """
        Normalize a batch of tensors by cdf.
        """
        x = x.to('cpu').numpy()

        shape = x.shape
        x = x.reshape(-1, shape[2])
        out = np.zeros_like(x)
        aux = []
        for i in range(shape[2]):
            out[:,i], auxi = self.normalizecdf1d(x[:,i])
            aux.append(auxi)
        return torch.Tensor(out.reshape(shape),device='cpu').to(self.device), aux

    def denormalizecdf(self, nx, aux):
        """
        Denormalize a batch of tensors by cdf.
        """
        nx = nx.to('cpu').numpy()

        shape = nx.shape
        nx = nx.reshape(-1, shape[2])
        out = np.zeros_like(nx)
        for i in range(shape[2]):
            out[:,i] = self.denormalizecdf1d(nx[:,i], aux[i])
        return torch.Tensor(out.reshape(shape),device='cpu').to(self.device)
    
    def normalizecdf1d(self, x):
        """
        Normalizes batch tensors by cfd scaling of a simulation for assessment in validation and testing phases.

        Parameters
        ---
            x (torch.Tensor): tensor to be normalized
            ndim (int): normalization dimension

        Returns
        ---
            y (torch.Tensor): normalized tensor   
            xmin (torch.Tensor): lower quantile along ndim   
            xmax (torch.Tensor): upper quantile along ndim
            ymin (torch.Tensor): lower cumulative prob
            ymax (torch.Tensor): upper cumulative prob
        """
        quantiles, cumprob = self.empirical_cdf(x)
        fn = interpolate.interp1d(quantiles, cumprob)
        inv = interpolate.interp1d(cumprob, quantiles)

        xmin, xmax = quantiles.min(), quantiles.max()
        ymin, ymax = cumprob.min(), cumprob.max()

        x = np.clip(x, xmin, xmax)
        y = fn(x)
        y = 2 * y - 1
        return y, (xmin, xmax, ymin, ymax, inv)

    def denormalizecdf1d(self, x, aux, eps=1e-5):
        """
        Denormalizes batch tensors by cfd scaling of a simulation for assessment in validation and testing phases.

        Parameters
        ---
            x (torch.Tensor): normalized tensor   
            xmin (torch.Tensor): lower quantile along ndim   
            xmax (torch.Tensor): upper quantile along ndim
            ymin (torch.Tensor): lower cumulative prob
            ymax (torch.Tensor): upper cumulative prob
            eps (float): error interval

        Returns
        ---
            y: denormalized tensor
        """
        x = (x + 1) / 2.
        x = np.clip(x, aux[2], aux[3])

        y = aux[4](x)
        return y

    def empirical_cdf(self,sample):
        """
        Empirical estimation of cumulative distribution function to be utilized in cdf normalization/denormalization.

        Parameters
        ---
            sample (torch.Tensor): sample to obtain empirical cumulative probability distribution on
        
        Returns
        ---
            quantiles (torch.Tensor): quantiles on the direction proposed
            cumprob (torch.Tensor): cumprob on the direction proposed
        """
        ## https://stackoverflow.com/a/33346366

        # find the unique values and their corresponding counts
        quantiles, counts = np.unique(sample, return_counts=True)

        # take the cumulative sum of the counts and divide by the sample size to
        # get the cumulative probabilities between 0 and 1
        cumprob = np.cumsum(counts).astype(np.double) / sample.size

        return quantiles, cumprob
    
    def normalize(self, *args, **kwargs):
        """
        Parses normalization functions according to arguments, argument list follows from the 
        referenced normalization functions.
        """
        if self.args.std_normalizer:
            return self.normalizestd(*args, **kwargs)
        elif self.args.lin_normalizer:
            return self.normalizelin(*args, **kwargs)
        elif self.args.cdf_normalizer:
            return self.normalizecdf(*args, **kwargs)

    def denormalize(self, *args, **kwargs):
        """
        Parses denormalization functions according to arguments, argument list follows from the
        referenced denormalization functions.
        """
        if self.args.std_normalizer:
            return self.denormalizestd(*args, **kwargs)
        elif self.args.lin_normalizer:
            return self.denormalizelin(*args, **kwargs)
        elif self.args.cdf_normalizer:
            return self.denormalizecdf(*args, **kwargs)

    def seperate_context(self, x, ctx=None, shot=None):
        """
        Seperates the u,y batch pairs into given (context) and estimated parts according to a defined
        seq length, currently this seperation is only available for the transformer identification
        procedures

        Parameters
        ---
            x (torch.Tensor): tensor to be seperated into context | new
            ctx (int): context window
            shot (int): number of shots for long horizon models

        Returns
        ---
            xctx (torch.Tensor): x until ctx
            xnew (torch.Tensor): x after ctx

        """
        if ctx is None:
            ctx = self.sqlctx

        if shot is not None:
            ctx = shot*1000 + ctx

        xctx = x[:, :ctx, :]
        xnew = x[:, ctx:, :]

        return xctx, xnew

    def cosine_annealing(self, eta_min):
        """
        Cosine annealed learning rate -> Cosine Annealing with Warm Restarts

        Parameters
        ---
            etamin (float): min learning rate to converge at the end of every cosine cycle
        """
        self.scheduler = CosineAnnealingWarmRestarts(optimizer=self.optimizer, 
                                                     T_0=self.totaliters,
                                                     eta_min=eta_min,
                                                     last_epoch=-1)
    
    def cosine_annealing_with_warmup(self, eta_min, warmupiters):
        """
        Cosine annealed learning rate with warmup iterations at every epoch. -> Cosine Annealing with Warmup Iterations

        Parameters
        ---
            etamin (float): min learning rate to converge at the end of every cosine cycle
            warmupiters (int): warmup iterations at the beginning of every epoch for ramping up to the max lr
        """
        scheduler1 = LinearLR(optimizer=self.optimizer,
                              start_factor=0.1,
                              end_factor=1.0,
                              total_iters=warmupiters,
                              last_epoch=-1)
        scheduler2 = CosineAnnealingWarmRestarts(optimizer=self.optimizer, 
                                                     T_0=self.totaliters,
                                                     eta_min=eta_min,
                                                     last_epoch=-1)
        self.scheduler = SequentialLR(optimizer=self.optimizer,
                                      schedulers=(scheduler1,scheduler2),
                                      milestones=[warmupiters],
                                      last_epoch=-1)
    
    def step_decay(self, gamma, stepsize):
        """
        Defines learning rate --> Step Decay

        Parameters
        ---
            gamma (float): decrease at each iteration of scheduler
            stepsize (float): to decrease at
        """
        self.scheduler = StepLR(optimizer=self.optimizer,
                                gamma=gamma,
                                step_size=stepsize,
                                last_epoch=-1)

    def exponential_decay(self, gamma):
        """
        Defines learning rate --> Exponential Decay

        Parameters
        ---
            gamma (float): decrease at each iteration of scheduler
        """
        self.scheduler = ExponentialLR(optimizer=self.optimizer, 
                                       gamma=gamma,
                                       last_epoch=-1)
    
    def constant_lr(self, factor):
        """
        Defines learning rate --> Constant

        Parameters
        ---
            factor (float): constant factor of initial learning rate
        """
        self.scheduler = ConstantLR(optimizer=self.optimizer,
                                    factor=factor,
                                    total_iters=self.totaliters,
                                    last_epoch=-1)

    def configure_learning_rate(self, 
                                eta_min=None,
                                warmup=None,
                                stepsize=None, 
                                gamma=None, 
                                factor=None):
        """
        Defines learning rate --> 
            Warmup Cosine Annealing, 
            Step Decay
            Exponential Decay, 
            Constant
        
        Parameters
        ---
            eta_min (float): min learning rate to converge at the end of every cosine cycle
            warmup (float): warmup iterations for cosine cycles
            stepsize (int): to decrease at
            gamma (float): decrease at each iteration of scheduler
            factor (float): constant factor of initial learning rate
        """
        if self.args.cosine_annealing:
            if eta_min is None:
                eta_min = self.learning_rate/10
            self.cosine_annealing(eta_min)
        
        elif self.args.cosine_annealing_with_warmup:
            if eta_min is None:
                eta_min = self.learning_rate/10
            if warmup is None:
                warmup = 1000
            self.cosine_annealing_with_warmup(eta_min, warmup)

        elif self.args.exponential_decay:
            if gamma is None:
                gamma = float(0.9995)
            self.exponential_decay(gamma)

        elif self.args.step_decay:
            if gamma is None:
                gamma = float(0.75)
            if stepsize is None:
                stepsize = int(self.totaliters/10)
            self.step_decay(gamma,stepsize)

        elif self.args.constant_decay:
            if factor is None:
                factor = float(1.0)
            self.constant_lr(factor)




