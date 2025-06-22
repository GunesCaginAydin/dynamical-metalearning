# IN CONTEXT META LEARNING FOR SYSTEM IDENTIFICATION

[[Paper]]() 
[[Data]]()
[[Colab (data generation)]]()
[[Colab (sim2sim)]]()
[[Colab (sim2real)]]()

[Gunes Cagin Aydin](https://github.com/Gunesnes)<sup>1</sup>,
[Loris Roveda](https://www.supsi.ch/loris-roveda)<sup>2</sup>,
[Asad Ali Shadid](https://www.supsi.ch/en/asad-ali-shahid)<sup>2</sup>,
[Angelo Morencelli](https://www.supsi.ch/en/angelo-moroncelli-)<sup>1</sup>,

<sup>1</sup>Politecnico di Milano,
<sup>2</sup>SUPSI/IDSIA,

(Insert Media)

This work is the continuation of previous works on system and class modeling with in-context learning through direct implementation of modern neural network architectures. 

[1]	From system models to class models: An in-context learning paradigm

[2]	RoboMorph: In-Context Meta-Learning for Robot Dynamics Modeling

Our main contribution can be grouped under 3 main branches:

1)	improved dynamical system identification with metalearning transformer models
	
2) 	implementation of diffusion models as a competitor neural architecture
	
3)	implementation of isaacgym based controllers for controller modeling and sim2real tasks

## Logs and Plots

Our findings and results are listed on our host website. The naming conventions follow from the repository organization explained in the next section-

[[Datasets]]()
[[Models]]()
[[Logs]]()
[[Plots]]()

## Repository Organization

Our approach is comprised of 2 interworking modules: data_generation, sys_identification. The former utilizes isaacgym environments for synthetic data generation while the latter performs training, finetuning and testing of the data on Franka Emika Panda and Kuka Allegro robotic manipulators. Below are the module hierarchies...

```
└── data_generation
    ├── {data_tensors} : .pt
    │   └── train
    │       ├── MG1
    │       ├── MG2
    │       ├── ...
    │       └── MGC
    │   └── test
    │       ├── T1
    │       ├── T2
    │       ├── ...
    │       └── TC
    │
    ├── {data_objects} : .json
    │   ├── MG1
    │   ├── MG2
    │   ├── ...
    │   └── MGC
    |
    ├── {plots} : .png  
    │   ├── aux
    │   ├── MG1
    │   ├── MG2
    │   ├── ...
    │   └── MGC
    |
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
└── sys_identification
    ├── {models} : .pt
    │   ├── MG1
    │   ├── MG2
    │   ├── ...
    │   └── MGC
    │
    ├── {logs} : .txt
    │   ├── MG1
    │   ├── MG2
    │   ├── ...
    │   └── MGC
    |
    ├── {plots} : .png  
    │   ├── aux
    │   ├── MG1
    │   ├── ...
    │   └── MGC
    |
    ├── {architectures} : .py
    │   └── transformer
    │       └── transformer_sim.py
    │   └── diffuser
    │       ├── diffuser_utils.py
    │       ├── diffuser_models.py
    │       └── diffuser_sim.py
    │   └── recedinghorizon
    │       ├── rechor_utils.py
    │       ├── rechor_models.py
    │       └── rechor_sim.py
    |
    ├── {sysidentification_modules} : .py
    │   ├── losses.py
    │   ├── metrics.py
    │   ├── utils.py
    │   ├── dataset.py
    │   ├── train.py
    │   └── test.py
    │
    ├── {test.sh}
    └── {train.sh}
```

## Data Generation

We decided to tackle data generation with varying datasets that have different randomization amounts and input/output generators. Most important points of variance in our investigation is on:

* rigid body properties, dof states, dof properties

* isaacgym internal controller dof properties

* osc / pid / joint_pid controller gains

* input trajectories

### Training Datasets
1) MG1: base dataset MS/CH

2) MG2: base + 10% randomization on states and props
 
3) MG3: base + 6D orientation

4) MG4: base + frequency ranzomization

5) MG5: base + out-of-distribution frequency randomization
 
6) MG6: base + isaacgym internally measured torques

7) MGOSC: base dataset VS/FS/FC

8) MGC: base dataset osc/pid/joint_pid

### Testing Datasets
1) T1: MS/CH

2) T2: 25-50% randomization

3) T3: out-of-distribution frequencies

4) T4: IMP/TRAPZ

5) TOSC: VS/FS/FC

6) TC: osc/pid/joint_pid

### Creating a Dataset

It is possible to create a new dataset from scratch using the isaacgym pipeline. A simplistic dataset with link and position randomizaiton
could be as below. Check gen.sh for more information and detailed examples.

```console
$ pip install -e .
```

## System Identification

System identification is practices on 2 fundamentally different problems: sim2sim and sim2real. For sim2sim, we train our models using
feedforward torques from isaacgym generated without any prior knowledge of the internal representation of isaacgym controllers. For sim2real
we train our models using feedback torques, tracking trajectories and controller gains with novel implementations of osc, pid and joint pid
controllers.

### sim2sim

For data generation, it is important to modify gen.sh for specific needs. A most simplistic use case could be to generate data for franka emika panda
subjected to direct feedforward joint torques, in which case the generation script is callable as:

```console
$ pip install -e .
```

### sim2real
For data generation, it is important to modify gen.sh for specific needs. A most simplistic use case could be to generate data for franka emika panda
subjected to direct feedforward joint torques, in which case the generation script is callable as:

```console
$ pip install -e .
```

### Training / Testing / Finetuning

For data generation, it is important to modify gen.sh for specific needs. A most simplistic use case could be to generate data for franka emika panda
subjected to direct feedforward joint torques, in which case the generation script is callable as:

```console
$ pip install -e .
```

```console
$ pip install -e .
```

```console
$ pip install -e .
```

# INSTALLATION AND REQUIREMENTS

## Environments

We used IsaacGym 4 (deprecated now) for data generation and varying modules for training/testing our models on machines with Nvidia RTX4070 and Nvidia A100 GPUs with Ubuntu 20.04. 

### IsaacGym

Download the Isaac Gym Preview 4 release from the website (https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation.

```console
$ pip install -e .
```

### Conda Environment

Set the conda environment from .yaml.

```console
$ conda env create -f dep.yaml
```

## Hardware requirements

This projects requires a modern GPU. We used a combination of Nvidia RTX4070 and Nvidia A100 GPUs. However, most of the work is still possible to do on older gen Nvida 3000 GPUs.

## Citing

If you find this work useful, please consider citing it.

```
@article{forgione2023from,
  author={Forgione, Marco and Pura, Filippo and Piga, Dario},
  journal={IEEE Control Systems Letters}, 
  title={From System Models to Class Models:
   An In-Context Learning Paradigm}, 
  year={2023},
  volume={7},
  number={},
  pages={3513-3518},
  doi={10.1109/LCSYS.2023.3335036}
}
```
## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Acknowledgements

* Our [`transformer_sim`]() architecture is adapted from [In-context learning for model-free system identification](https://github.com/forgi86/sysid-transformers).
* Our [`diffuser_sim`]() architecture is adapted from [Planning with Diffusion for Flexible Behavior Synthesis](https://github.com/jannerm/diffuser).
* Our [`rechor_sim`]() architecture is adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).
* Our [`data_generation`]() implementation is adapted from [RoboMorph: In-Context Meta-Learning for Robot Dynamics Modeling](https://github.com/izzab1926/RoboMorph).


