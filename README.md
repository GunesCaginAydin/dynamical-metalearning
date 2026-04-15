# Diffusion Sequence Models for Generative In-Context Meta-Learning of Robot Dynamics

🚀 [Project Page](https://robo-meta.github.io/) 

[Angelo Moroncelli](https://www.supsi.ch/en/angelo-moroncelli)<sup>2</sup>,
[Matteo Rufolo](https://www.supsi.ch/en/matteo-rufolo)<sup>2</sup>,
[Gunes Cagin Aydin](https://gunescaginaydin.github.io/)<sup>1</sup>,
[Asad Ali Shahid](https://www.supsi.ch/en/asad-ali-shahid)<sup>1,2</sup>,
[Loris Roveda](https://www.supsi.ch/loris-roveda)<sup>1,2</sup>,

<sup>1</sup>Politecnico di Milano,
<sup>2</sup>SUPSI/IDSIA,

## Datasets and Checkpoints

The naming conventions follow from the repository organization explained in the next section.

[Datasets](https://drive.google.com/drive/folders/1Vujrxv03aJg231vqYL_-AeBe-MDSIK74?usp=sharing)
[Model Checkpoints](https://drive.google.com/drive/folders/1tAl_eww7rZ7e8eJVZ-mDRn2VOlWPlcu9?usp=sharing)

## Repository Organization

Our approach comprises 2 interleaved modules: data_generation, sys_identification. The former utilizes isaacgym environments for synthetic data generation while the latter is used to train and test on the synthetic datasets. Below are the module hierarchies.

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
    |
    ├── {architectures} : .py
    │   └── transformer -> RoboMorph
    │       └── transformer_sim.py
    │   └── diffuser -> Diffuser
    │       ├── diffuser_utils.py
    │       ├── diffuser_models.py
    │       └── diffuser_sim.py
    │   └── recedinghorizon -> CDCNN and CDT
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

All datasets are randomized to some degree on initial conditions, dynamical parameters and torque signal parameters.

### Training Datasets - formerly MG
1) D1: fch=0.3,fms=0.15

2) D2: fch=[0.2,0.4],fms=[0.05,0.15]
 
3) D3: fch=[0.2,0.6],fms=[0.05,0.25]

4) D4: fch=[0.1,0.7],fms=[0.01,0.30]

### Testing Datasets - formerly T
1) Dtest: fch=[0.1,1.0],fms=[0.01,0.50]

### Creating a Dataset

It is possible to create a new dataset from scratch using the data_generation module. A simplistic dataset with link and position randomization
could be as follows:

```console
$ cd data_generation
$ python genfranka.py -ne 8 -ni 1000 -f 0.15 -nctrl -tjt "MS" -td 'train' -nd 'MGC' -hdo '4D' -tr 'franka' -v -dg -df
```
```
or alternatively
```console
$ gen.sh
```

Check gen.sh for more information and detailed examples.

### In-Context Meta-Learning of Forward Dynamics

Black-box meta-models of forward dynamics follow from the meta-learning paradigm suggested in (Forgione et al., 2023). Possible architectures are RoboMorph (base transformer, non-generative), Diffuser (generative diffusion-based, inpainted), CDCNN and CDT (generative diffusion-based, conditioned).

### Training and Testing on a Dataset

Training models is possible on all datasets adhering to the (env X horizon X input dim) dimensionality format. Architecture hyperparameters as well as training and testing parameters can be adjusted.

```console
cd sys_identification
$ python train.py -in 7 -out 14 -cos -std --data-name 'MG1' -lr '6e-4' -trb 32 -vlb 32 -evitr 100 -ctx 20 transformer -ttrf 1 -nl 12 -nh 12 -ne 384

```
or alternatively
```console
$ train.sh
```

```console
cd sys_identification
$ python test.py -cos -std --data-name 'MG1' --test-name 'T1' --total-sim-iterations 500

```
```
or alternatively
```console
$ test.sh
```

Check train.sh and test.sh for more information and detailed examples.

# INSTALLATION AND REQUIREMENTS

## Environments

We used IsaacGym 4 (deprecated now) for data generation and trained/tested the subsequent models on machines with Nvidia A100 GPUs on Ubuntu 20.04. 

### IsaacGym

Download the Isaac Gym Preview 4 release from the website (https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation. 

### Conda Environment

Before using the data_generation and sys_identification modules it is advised to first set the conda environment from the .yaml file.

```console
$ conda env create -f dep.yaml
```

## Hardware requirements

This projects requires a modern GPU. Even though we exclusively used an Nvidia A100 GPU, it should still be possible to emulate the results on older generation GPUs. We have not conducted any tests using CPUs. 

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Acknowledgements

* Our [`RoboMorph`](https://github.com/GunesCaginAydin/dynamical-metalearning/tree/main/sys_identification/architectures/transformer) architecture is adapted from [In-context learning for model-free system identification](https://github.com/forgi86/sysid-transformers).
* Our [`Diffuser`](https://github.com/GunesCaginAydin/dynamical-metalearning/tree/main/sys_identification/architectures/diffuser) architecture is adapted from [Planning with Diffusion for Flexible Behavior Synthesis](https://github.com/jannerm/diffuser).
* Our [`CDCNN and CDT`](https://github.com/GunesCaginAydin/dynamical-metalearning/tree/main/sys_identification/architectures/receding_horizon) architectures are adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).


