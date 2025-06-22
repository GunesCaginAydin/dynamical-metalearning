# CLASSIFY 

This work is the continuation of previous works on system and class modeling with in-context learning through direct implementation of modern neural network architectures. 

[1]	From system models to class models: An in-context learning paradigm


[2]	RoboMorph: In-Context Meta-Learning for Robot Dynamics Modeling

Our main contribution can be grouped under 3 main branches:

1)	improved dynamical system identification with metalearning transformer models
	
2) 	implementation of diffusion models as a competitor neural architecture
	
3)	implementation of isaacgym based controllers for controller modeling and sim2real tasks
	
# Repository organization

Our approach is comprised of 2 interworking modules: data_generation, sys_identification. The former utilizes isaacgym environments for synthetic data generation while the latter
performs training, finetuning and testing of the data.

```
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
```

```
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
```
We decided to tackle data generation with varying datasets that have different randomization amounts and schemes. A rundown of these datasets are as below.

Training Datasets
[1]

[2]

[3]

[4]

[5]

[6]

[7]

Testing Datasets
[1]

[2]

[3]

[4]

[5]


## Data Generation

### Datasets

## System Identification

### sim2sim

### sim2real

## In Context Learning Architectures for Class Modeling

### Training
 
### Testing

### FineTuning

#### Example

For data generation, it is important to modify gen.sh for specific needs. A most simplistic use case could be to generate data for franka emika panda
subjected to direct feedforward joint torques, in which case the generation script is callable as:

./gen.sh -> python genfranka.py xxx

For training, it is important to modify gen.sh for specific needs. A most simplistic use case could be to generate data for franka emika panda
subjected to direct feedforward joint torques, in which case the generation script is callable as:

./gen.sh -> python genfranka.py xxx

For testing, it is important to modify gen.sh for specific needs. A most simplistic use case could be to generate data for franka emika panda
subjected to direct feedforward joint torques, in which case the generation script is callable as:

./gen.sh -> python genfranka.py xxx

# Installation and requirements

### IsaacGym

Download the Isaac Gym Preview 4 release from the website (https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation. We highly recommend using a conda environment to simplify set up. 

### Conda Environment

Please replicate the conda environment for reproduciblity

## Hardware requirements

This projects requires a modern GPU to be evaluated. For our findings we used a combination of Nvidia RTX4070 and Nvidia A100 GPUs.

# Citing

If you find this work useful, please consider citing it.