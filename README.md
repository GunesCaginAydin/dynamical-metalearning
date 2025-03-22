# CLASSIFY 

This work is the continuation of previous works on system and class modeling with in-context learning through direct implementation of modern neural network architectures.

[1]	From system models to class models: An in-context learning paradigm


[2]	RoboMorph: In-Context Meta-Learning for Robot Dynamics Modeling

# Repository organization
***
└── datageneration
    ├── {data_tensors} : .pt
    │   └── train
    │       ├── MG1
    │       ├── MG2
    │       ├── ...
    │       └── FineTune3
    │   └── test
    │       ├── T1
    │       ├── T2
    │       ├── ...
    |       ├── Oneshot
    |       ├── Fewshot
    │       └── TOSC
    │
    ├── {data_objects} : .json
    │   ├── MG1
    │   ├── MG2
    │   ├── ...
    │   └── FineTune3
    |
    ├── {plots} : .png  
    │   ├── aux
    │   ├── MG1
    │   ├── ...
    │   └── FineTune3
    |
    ├── {logs} : .txt
    │   ├── aux
    │   ├── MG1
    │   ├── ...
    │   └── FineTune3
    |
    ├── {datageneration_modules} : .py
    │   ├── randomenvs.py
    │   ├── controllers.py
    │   ├── genutil.py
    │   └── genfranka.py
    |
    └── {gen.sh}
***

## [data_generation](data_generation)

x

### Simulation Naming and Data Generation Module

x

### Datasets

x

## In Context Learning Architectures for Class Modeling

x

### Training
 
x

### Testing

x

### FineTuning

x

#### Example
I have a certain model `ckpt_partition_20_batch16_embd192_heads8_lay12_MSE_ds1.pt`  
located in `Transformer_for_isaac/out_ds1`  
with a txt file located in `data_generation/training_ds1_list.txt`

I want to train-fine tuning on a certain dsX.

In order to correctly update the same model, it is necessary to rename such as:
`ckpt_partition_20_batch16_embd192_heads8_lay12_MSE.pt`  
is located in `Transformer_for_isaac/out_dsX`    
with the same txt file located in `data_generation/training_dataset_list.txt`  

then Launch (ATTENTION to the parameter: --model-dir "out_dsX")

```
source simulate_and_train.sh
```
In `simulate_and_train.sh` is possible to save some intermediates versions, depending on the dimension of simulation blocks.
Use wandb as register to keep trace.  
https://wandb.ai/home

Once the simulations-trainings will have been finished, the same file needs to be renamed as: 


### [Auxiliar](Auxiliar)

# Additional information

# Installation and requirements

### Isaac Gym rlgpu dataset environment

Download the Isaac Gym Preview 4 release from the website (https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation. We highly recommend using a conda environment to simplify set up. 

### CLASSIFY training environment

With the setup of NVIDIA Isaac Gym, a conda environment called rlgpu should be added automatically during set-up steps.

## Hardware requirements
While all the scripts can run on CPU, execution may be frustratingly slow. For faster training, a GPU is highly recommended.
To run the paper's examples, we used a Desktop Computer equipped with an NVIDIA RTX 4090 GPU.


# Citing
