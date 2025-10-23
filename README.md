# Advancing Identification Methodologies Through In-Context Dynamical Metalearning

🚀 [Project Page](https://sites.google.com/view/dynamical-incontextlearning/ana-sayfa) 

[Gunes Cagin Aydin](https://github.com/Gunesnes)<sup>1</sup>,
[Elia Tosin](https://www.supsi.ch/elia-tosin)<sup>2</sup>,
[Asad Ali Shahid](https://www.supsi.ch/en/asad-ali-shahid)<sup>2</sup>,
[Angelo Moroncelli](https://www.supsi.ch/en/angelo-moroncelli-)<sup>2</sup>,
[Loris Roveda](https://www.supsi.ch/loris-roveda)<sup>1,2</sup>,
[Francesco Braghin](https://www.mecc.polimi.it/it/personale/francesco.braghin)<sup>1</sup>,

<sup>1</sup>Politecnico di Milano,
<sup>2</sup>SUPSI/IDSIA,

This work is the continuation of previous works on dynamical identification through in-context learning, principally found on: 

[1]	From system models to class models: An in-context learning paradigm

[2]	RoboMorph: In-Context Meta-Learning for Robot Dynamics Modeling

Our main contribution can be grouped under 3 main categories:

1)	improved dynamical identification with better domain exploration, transfer learning between domains
  
2) 	implementation of non-autoregressive diffusion based neural architectures in dynamical identification tasks
  
3)	benchmarking on real trajectories, conducting sim2real experiments

<p align="center">
  <img src="./assets/meta-learning-scheme.png">
</p>

<p align="center">
  <img height=1000px, src="./assets/identification-pipeline.png">
</p>

## Logs and Plots

Our findings are listed on the [Project Page](https://sites.google.com/view/dynamical-incontextlearning/ana-sayfa) . The naming conventions follow from the repository organization explained in the next section.

[Datasets](https://drive.google.com/drive/folders/1WRHcEYSfIIVRhBrlzR18CGVkvezJIilo?usp=sharing)
[Models](https://drive.google.com/drive/folders/1WRHcEYSfIIVRhBrlzR18CGVkvezJIilo?usp=sharing)
[Logs](https://drive.google.com/drive/folders/1WRHcEYSfIIVRhBrlzR18CGVkvezJIilo?usp=sharing)
[Plots](https://drive.google.com/drive/folders/1WRHcEYSfIIVRhBrlzR18CGVkvezJIilo?usp=sharing)

## Repository Organization

Our approach is comprised of 2 interworking modules: data_generation, sys_identification. The former utilizes isaacgym environments for synthetic data generation while the latter performs training, finetuning and testing of the data on Franka Emika Panda and Kuka Allegro robotic manipulators. Below are the module hierarchies.

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

We decided to tackle data generation with varying datasets that have different randomization amounts and input/output generators. Most important points of variance in our investigations are on:

* rigid properties, dof states, dof properties

* isaacgym internal controller dof properties

* osc / pid / joint_pid / cic controller gains

* input trajectories

### Training Datasets
1) MG1: base dataset MS/CH | feedforward

2) MG2: base + 10% randomization on states and props | feedforward
 
3) MG3: base + 6D orientation | feedforward

4) MG4: base + frequency ranzomization | feedforward

5) MG5: base + out-of-distribution frequency randomization | feedforward
 
6) MG6: base + isaacgym internally measured torques | feedforward

7) MGOSC: base dataset VS/FS/FC | feedforward

8) MGC: base dataset osc/pid/joint_pid/cic | feedback

### Testing Datasets
1) T1: MS/CH | in-distribution 

2) T2: MS/CH out-of-distribution | rigid properties

3) T3: MS/CH out-of-distribution | frequencies

4) T4: IMP/TRAPZ | out-of-distribution

5) TOSC: VS/FS/FC | out-of-distribution

6) TC: osc/pid/joint_pid | out-of-distribution

7) TREAL: VS/FS/FC trajectories collected from Franka Emika Panda | real 

### Creating a Dataset

It is possible to create a new dataset from scratch using the data_generation module. A simplistic dataset with link and position randomization
could be as below:

```console
$ python genfranka.py -ne 8 -ni 1000 -f 0.15 -nctrl -tjt "MS" -td 'train' -nd 'MGC' -hdo '4D' -tr 'franka' -v -dg -df
```

where we specify simulation parameters, control inputs, directories and naming conventions alltogether. Check gen.sh for more information and detailed examples.

## System Identification

System identification (or as colloquially called dynamical identification throughout this work) is practiced in 2 different problems: dynamic identification of feedforward and feedback controller dynamics. In both cases, the resultant input/output mapping defined through an emerging black-box model of the dynamic behavior constitutes to a forward dynamical problem.

For feedforward controller dynamics we principally consider joint torques as inputs and cartesian and joint variables as outputs. For feedback controller dynamics, we choose to diversify the available inputs and consider either one of joint torques, joint or cartesian reference trajectories to the controller or controller as inputs while considering cartesian and joint variables as outputs.

The identified dynamics are consecutively tested in simulation and benchmarked against real trajectories collected from Franka Emika Panda through a set of comparitive metrics in horizon estimation tasks.

### Training / Testing / Finetuning

Training models is possible on all datasets adhering to the (env X horizon X input dim) dimensionality constraints. Hyperparameters as well as neural architecture specific parameters, such as MLP layers, heads and embeddings for transformers and diffusion timesteps, convolutional layers, and block for diffusers, can be fixed before training.

```console
$ python train.py -in 7 -out 14 -cos -std --data-name 'MG1' -lr '6e-4' -trb 32 -vlb 32 -evitr 100 -ctx 20 transformer -ttrf 1 -nl 12 -nh 12 -ne 384

```

Testing, or inference, and finetuning is subsequently conducted on trained models where the inference horizon and context can be updated if the proposed neural architecture is adequate for any such modification.

```console
$ python test.py -cos -std --data-name 'MG1' --test-name 'T1' --total-sim-iterations 500

```

Check train.sh and test.sh for more information and detailed examples.

# INSTALLATION AND REQUIREMENTS

## Environments

We used IsaacGym 4 (deprecated now) for data generation and trained/tested the subsequent models on machines with Nvidia RTX4070 and Nvidia A100 GPUs with Ubuntu 20.04. 

### IsaacGym

Download the Isaac Gym Preview 4 release from the website (https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation. 


### Conda Environment

Before using the data_generation and sys_identification modules it is required to first set the conda environment from .yaml file.

```console
$ conda env create -f dep.yaml
```

## Hardware requirements

This projects requires a modern GPU. We used, at any given time, either one of Nvidia RTX4070 and Nvidia A100 GPUs. However, we presume that the work is emulatable still in older generation Nvidia GPUs. We have not conducted any tests using CPUs. 

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


