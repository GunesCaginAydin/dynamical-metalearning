from isaacgym import gymapi
from isaacgym import torch_utils
import argparse
import torch
import matplotlib.gridspec as grid
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json

class parser():
    """
    Custom parser, a merged script of older data generation programs that
    are obtained through the parser - refer
    to README.md.

    Parameters
    ---
        description (string): id of parser - DEPRECATED
        params (list): list of extra parameters - DEPRECATED
    """
    def __init__(self,
                 description="FrankaGeneration", 
                 params=[]):
        self.description = description
        self.params = params
        self.parser = argparse.ArgumentParser(description=self.description)

    def __str__(self):
        return f'Parser Object instantiated'
    
    def getargs(self):
        return self.parser._get_args()
    
    def setargs(self,_arg):
        self.params.append(_arg)
        self.parse_arguments()
        
    def parse_device_str(self,device_str):
        device = 'cpu'
        device_id = 0

        if device_str == 'cpu' or device_str == 'cuda':
            device = device_str
            device_id = 0
        else:
            device_args = device_str.split(':')
            assert len(device_args) == 2 and device_args[0] == 'cuda', f'Invalid device string "{device_str}"'
            device, device_id_s = device_args
            try:
                device_id = int(device_id_s)
            except ValueError:
                raise ValueError(f'Invalid device string "{device_str}". Cannot parse "{device_id}"" as a valid device id')
        return device, device_id

    def parse_arguments(self):
        """
        Some arguments COULD BE DEPRECATED
        """
        self.parser.add_argument('--sim_device', type=str, default="cuda:0", 
                            help='Physics Device in PyTorch-like syntax')
        self.parser.add_argument('--pipeline', type=str, default="gpu", 
                            help='Tensor API pipeline (cpu/gpu)')
        self.parser.add_argument('--graphics_device_id', type=int, default=0, 
                            help='Graphics Device ID')
        physics_group = self.parser.add_mutually_exclusive_group()
        physics_group.add_argument('--flex', action='store_true', 
                            help='Use FleX for physics')
        physics_group.add_argument('--physx', action='store_true', 
                            help='Use PhysX for physics')
        self.parser.add_argument('--num_threads', type=int, default=0, 
                            help='Number of cores used by PhysX')
        self.parser.add_argument('--subscenes', type=int, default=0, 
                            help='Number of PhysX subscenes to simulate in parallel')
        self.parser.add_argument('--slices', type=int, default=0,
                            help='Number of client threads that process env slices')

        self.parser.add_argument("-ne", "--num-envs", type=int, default=4,
                            help="number of environments")
        
        self.parser.add_argument("-tr", "--type-of-robot", type=str, default="franka", choices=["franka",
                                                                                                "kuka"],
                            help="choose between types of assets")

        self.parser.add_argument("-dg", "--disable-gravity", action="store_true",
                            help="disable gravity in simulation, controller compensation")
        self.parser.add_argument("-df","--disable-friction", action="store_true",
                            help="disable friction in simulation, controller compensation")
        self.parser.add_argument("-nctrl", "--no-controller", action="store_true",
                            help="apply torques to joints directly, relies on isaacgym internal controller")
        self.parser.add_argument("-tjt", "--type-of-joint-torque-profile", choices=["MS",
                                                                                    "CH",
                                                                                    "IMP",
                                                                                    "TRAPZ"],
                            help="type of joint torque profile in the case of no controller")
        self.parser.add_argument("-ctrl","--controller",action="store_true",
                            help="implement controller")
        self.parser.add_argument("-tctrl","--type-of-controller",choices=["osc",
                                                                          "pd",
                                                                          "joint_pd",
                                                                          "pid",
                                                                          "joint_pid",
                                                                          "cic"],
                            help="type of controller that acts upon cartesian or joint space")
        self.parser.add_argument("-ttraj","--type-of-trajectory",choices=["MS",
                                                                          "CH",
                                                                          "IMP",
                                                                          "TRAPZ",
                                                                          "joint_MS",
                                                                          "joint_CH",
                                                                          "joint_IMP",
                                                                          "joint_TRAPZ",
                                                                          "VS",
                                                                          "FS",
                                                                          "FC",
                                                                          "pick",
                                                                          "push"],
                            help="type of trajectory that the controller is to reference")
        self.parser.add_argument("-rcg","--random-controller-gains", action="store_true",
                            help="randomize controller gains for metalearning gain mapping")
        self.parser.add_argument("-dfrp","--damping-from-rigidprops", action="store_true", default=False,
                            help="obtain derivative controller gain from rigid props and nominal damping ratios")        

        self.parser.add_argument("-v", "--visualize", action="store_true",
                            help="run the simulation in visualization mode, None = headless")
        self.parser.add_argument("-is", "--include-saturation", action="store_true",
                            help="include saturated environments in the data struct")
        self.parser.add_argument("-fq", "--fix-quarternions", action="store_true",
                            help="compensate errors in quarternions instead of rejecting")
        self.parser.add_argument("-hdo", "--orientation-dimension", choices=["3D","4D","6D"], default="4D",
                            help="implement 6d continuous rotation matrices instead of 4d discontinuous q's")
        self.parser.add_argument("-mf", "--measure-force", action="store_true",
                            help="measure forces - ground truth check")
        self.parser.add_argument("-mgf", "--measure-gravity-friction", action="store_true",
                            help="measure gravity and friction in buffers - ground truth check")      
        self.parser.add_argument("-dp", "--no-plot", action="store_true",
                            help="plot the input/output tensors")
        self.parser.add_argument("-ds", "--no-save", action="store_true",
                            help="save input/output tensors")
        self.parser.add_argument("-td", "--type-of-dataset", type=str, default="train",
                            help="type of dataset to be created: train|test")
        self.parser.add_argument("-nd", "--name-of-dataset", type=str, default="MG",
                            help="name of dataset to be created, check documentation for full ref.")        

        self.parser.add_argument("-ri", "--random-initial-positions", action="store_true",
                            help="randomize the initial positions")
        self.parser.add_argument("-rv", "--random-initial-velocities", action="store_true",
                            help="randomize the initial velocities")
        self.parser.add_argument("-rm", "--random-masses", type=float, default=0,
                            help="randomize the link masses by the specified percentage around nominal")
        self.parser.add_argument("-rcom", "--random-coms", type=float, default=0,
                            help="randomize the link coms by the specified percentage around nominal")
        self.parser.add_argument("-rinr", "--random-inertias", type=float, default=0,
                            help="randomize the link coms by the specified percentage around nominal")
        self.parser.add_argument("-rstf", "--random-stiffness", type=float, default=0,
                            help="randomize the dof stiffness by the specified percentage around nominal")
        self.parser.add_argument("-rdam", "--random-damping", type=float, default=0,
                            help="randomize the dof damping by the specified percentage around nominal")
        self.parser.add_argument("-rcf", "--random-coulomb-friction", type=float, default=0,
                            help="randomize the dof coulomb friction by the specified percentage around nominal")
        self.parser.add_argument("-rad", "--random-angular-damping", type=float, default=0,
                            help="randomize the asset angular damping by the specified percentage around nominal")

        self.parser.add_argument("-f", "--frequency", type=float, default=0.1,
                            help="master frequency of actions")
        self.parser.add_argument("-ni", "--num-iters", type=int, default=1000,
                            help="number of iterations in a single simulation")
        self.parser.add_argument("-nr", "--num-runs", type=int, default=1,
                            help="number of consecutive runs in a single execution")
        self.parser.add_argument("-urdf", "--urdf-values", default=False, action='store_true',
                            help="number of consecutive runs in a single execution")
        self.parser.add_argument("-s", "--seed", default=False,
                            help="seed for reproducibility")

        for argument in self.params:
            if ("name" in argument) and ("type" in argument or "action" in argument):
                help_str = ""
                if "help" in argument:
                    help_str = argument["help"]

                if "type" in argument:
                    if "default" in argument:
                        self.parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                    else:
                        self.parser.add_argument(argument["name"], type=argument["type"], help=help_str)
                elif "action" in argument:
                    self.parser.add_argument(argument["name"], action=argument["action"], help=help_str)

            else:
                print("\nERROR: command line argument name, type/action must be defined, argument not added to parser")
                print("supported keys: name, type, default, action, help\n")

        args = self.parser.parse_args()

        args.sim_device_type, args.compute_device_id = self.parse_device_str(args.sim_device)
        pipeline = args.pipeline.lower()

        assert (pipeline == 'cpu' or pipeline in ('gpu', 'cuda')), f"Invalid pipeline '{args.pipeline}'. Should be either cpu or gpu."
        args.use_gpu_pipeline = (pipeline in ('gpu', 'cuda'))

        if args.sim_device_type != 'cuda' and args.flex:
            print("Can't use Flex with CPU. Changing sim device to 'cuda:0'")
            args.sim_device = 'cuda:0'
            args.sim_device_type, args.compute_device_id = self.parse_device_str(args.sim_device)

        if (args.sim_device_type != 'cuda' and pipeline == 'gpu'):
            print("Can't use GPU pipeline with CPU Physics. Changing pipeline to 'CPU'.")
            args.pipeline = 'CPU'
            args.use_gpu_pipeline = False

        args.physics_engine = gymapi.SIM_PHYSX
        args.use_gpu = (args.sim_device_type == 'cuda')

        if args.flex:
            args.physics_engine = gymapi.SIM_FLEX

        if args.slices is None:
            args.slices = args.subscenes

        return args

class reader():
    """
    Reader object ot acquire metadata from the data.txt file, the acquisition object should
    comply with the below rules:
    * TOTAL_COORDS: int
    * TOTAL_JOINTS: int
    * TOTAL_LINKS: int
    * SOLVER_TIME: int
    * SUBSTEPS: int
    * SOLVER_TYPE: int
    * NUM_POS_ITER: int
    * NUM_VEL_ITER: int
    * FIX_BASE_LINK: bool
    * FLIP_VISUAL_ATTACHMENTS: bool
    * ARMATURE: float
    * POS_END: np.array(9,2) float
    * VEL_END: np.array(9,1) float
    * ACC_END: np.array(9,1) float
    * JER_END: np.array(9,1) float
    * TOR_END: np.array(9,1) float
    * TAC_END: np.array(9,1) float
    * MASS_NOM: np.array(11,1) float
    * COM_NOM: np.array(11,3) float
    * INERTIA_NOM: np.array(11,6) float
    * STIFFNESS_NOM: np.array(9,1) float
    * DAMPING_NOM: np.array(9,1) float
    * COULOMB_NOM; np.array(9,1) float
    * ANGDAMPING_NOM: float
    * FRICTION_EXT: np.array(7,3)
    * GRAVITY: float

    Parameters
    ---
        path (str) : path to generation data

    Metadata is seperately registered to the urdf, for ease of control flow, metadata is taken from
    data.json.
    """
    def __init__(self,
                 path='data.json'):
        self.path = path
        print(f'\nGeneration MetaData from: {self.path}')
    
    def __str__(self):
        return f'Reader Object instantiated'

    def read_data(self):
        """
        Read metadata from .json file

        Parameters
        ---
            None

        Returns
        ---
            data (dict) : generation data dictionary
        """
        with open(self.path, 'r') as json_file:
            data = json.load(json_file)

        print('Data stored in dict')
        return data
    
class savedata():
    """
    Data saver for creating buffer data objects(.pt) for later reference, input trajectory and
    output pose is recorded in .pt format and accessed in the future stages of the protocol.
    data_save example:

        SEED_ENVS_STEPS_FREQS_G_F_RI_RV_RM_RCOM_RINR_RS_RD_RF_RAD_RCG_QF_ST_QR_TINP_TCTRL_FOR

    * SEED: generated seed of the simulation - int
    * ENVS: number of *valid* environments in the simulation - int
    * STEPS: maximum number of steps in the simulation - int
    * FREQS: master frequency - float
    * G: gravity compensation [0,1]
    * F: friction compensation [0,1]
    * RI: random initial joint positions [0,1]
    * RV: random initial joint velocities [0,1]
    * RM: random link masses [0,25]
    * RCOM: random link coms [0,25]
    * RINR: random inertias [0,25]
    * RS: random joint stiffness [0,25]
    * RD: random joint damping [0,25]
    * RF: random Coulomb friction [0.25]
    * RAD: random angular asset damping [0,25]
    * RCG: random controller gains [0,1]
    * QF: manual quarternion fix [IQ, NQ]
    * ST: saturation inclusion [IS, NS] 
    * QR: type of orientation resolution [3D,4D,6D]
    * TINP: type of input for joint torque profile no controller [MS,CH,IMP,TRAPZ]
    * TTRAJ: type of input for tracking trajectory with controller [MS,CH,IMP,TRAPZ,joint_MS,joint_CH,joint_IMP,joint_TRAPZ,VS,FS,FC,pick,push]
    * FOR: use of dataset [train,test]

    Parameters
    ---
        args (dict) : user arguments parsed in generation
        control_trajectory (torch.Tensor) : control input
        pose (torch.Tensor) : controlled output
        seed (int) : seed for randomization determinancy
        valid_envs (int) : valid environments in each generation after exclusions
        target (torch.Tensor) : control target for controller trajectory inputs ONLY
        rigid body props (torch.Tensor) : rigid props of each asset
        gentime (float) : generation time
        collision (Bool) : collision data, True if every environment collides - DEPRECATED
        path (string) = path to save data to 
        controlgains (torch.Tensor) : controller gains if controller implemented
    """
    def __init__(self,
                 args,
                 control_trajectory,
                 pose,
                 seed,
                 valid_envs,
                 target,
                 rigidbodyprops,
                 gentime,
                 collision=False,
                 path='.',
                 controlgains=None
                ):
        
        self.args = args
        self.pid = (args.type_of_controller=='pid' or args.type_of_controller=='joint_pid' or args.type_of_controller=='cic')
        self.ct = control_trajectory
        self.ps = pose
        self.tr = target
        self.rbp = rigidbodyprops
        if controlgains is not None:
            self.kp = controlgains['kp'] # 1D, 6D, 9D
            self.kd = controlgains['kv'] # 1D, 6D, 9D
            self.ki = controlgains['ki'] # 1D, 6D, 9D
        self.seed = seed
        self.valid_envs = valid_envs
        self.collision = collision
        self.name_tensor = ''
        self.useable = True
        self.path = path
        self.generation_time = gentime
        if self.args.controller:
            print(f"Saving tensor input/output data\n"
                    f"Control Dimension:{self.ct.shape}\n"
                    f"Pose Dimension:{self.ps.shape}\n"
                    f"Target Dimension:{self.tr.shape}\n"
                    f"Control Gain Dimension:{self.kp.shape}")
            self.tensors_from_isaacGym = {
                    'control_action': self.ct.to('cpu'), 
                    'position': self.ps.to('cpu'),
                    'target': self.tr.to('cpu'),
                    'masses': torch.tensor([[link.mass for link in env] for env in self.rbp],device='cpu'),
                    'coms' : torch.tensor([[[link.com.x,link.com.y,link.com.z] for link in env] for env in rigidbodyprops],device='cpu'),
                    'inertias' : torch.tensor([[[link.inertia.x.x, link.inertia.y.y, link.inertia.z.z] for link in env] for env in rigidbodyprops],device='cpu'),
                    'kp' : self.kp.to('cpu'), # 1D, 6D, 9D
                    'kd' : self.kd.to('cpu'), # 1D, 6D, 9D 
                    'ki' : self.ki.to('cpu') if self.pid else torch.zeros_like(self.kp,device='cpu') # 1D, 6D, 9D, None
                    }
        elif self.args.no_controller:
            print(f"Saving tensor input/output data\n"
                    f"Control Dimension:{self.ct.shape}\n"
                    f"Pose Dimension:{self.ps.shape}\n")
            self.tensors_from_isaacGym = {
                    'control_action': self.ct.to('cpu'),
                    'position': self.ps.to('cpu'), 
                    'masses': torch.tensor([[link.mass for link in env] for env in self.rbp],device='cpu'),
                    'coms' : torch.tensor([[[link.com.x,link.com.y,link.com.z] for link in env] for env in rigidbodyprops],device='cpu'),
                    'inertias' : torch.tensor([[[link.inertia.x.x, link.inertia.y.y, link.inertia.z.z] for link in env] for env in rigidbodyprops],device='cpu'),
                    }
        try:
            with open(f'{self.path}/data_objects/{self.args.name_of_dataset}.json', 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        except:
            self.metadata = {"dataname" : [],
                             "genname" : [],
                             "genenvs" : [], 
                             "gentime" : []}
            with open(f'{self.path}/data_objects/{self.args.name_of_dataset}.json', 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, sort_keys=True, indent=4)
            
    def __str__(self):
        return f'Data Saver Object instantiated'
    
    def getdata(self):
        return {
            'control' : self.ct,
            'pose' : self.ps,
            'rigidbodyprops' : self.rbp,
            'target' : self.tr,
            'kp' : self.kp,
            'kv' : self.kd,
            'ki' : self.ki
        }
        
    def setdata(self,control_trajectory_,pose_,target_,rbp_,kp_,kv_,ki_):
        self.ct = control_trajectory_
        self.ps = pose_
        self.rbp = rbp_
        self.tr = target_
        self.kp = kp_
        self.kd = kv_
        self.ki = ki_

    def save_tensors(self):
        """
        Saves input/output tensors in .pt format in the specified directory.
        """
        self.tr = self.tr.movedim(1,2).movedim(0,1) if self.args.controller else ''
        self.collision = False
        G = 'G' if not self.args.disable_gravity else 'NG' 
        F = 'F' if not self.args.disable_friction else 'NF' 
        RI = 'P' if self.args.random_initial_positions else 'NP'
        RV = 'V' if self.args.random_initial_velocities else 'NV'
        RCG = 'K' if self.args.random_controller_gains else 'NK'
        RM = str(int(self.args.random_masses))
        RCOM = str(int(self.args.random_coms))
        RINR = str(int(self.args.random_inertias))
        RS = str(int(self.args.random_stiffness))
        RD = str(int(self.args.random_damping))
        RF = str(int(self.args.random_coulomb_friction))
        RAD = str(int(self.args.random_angular_damping))
        QF = 'Q' if self.args.fix_quarternions else 'NQ' 
        IS = 'S' if self.args.include_saturation else 'NS' 
        QR = str(self.args.orientation_dimension) 
        TINP = self.args.type_of_joint_torque_profile if self.args.type_of_joint_torque_profile is not None else 'NT'
        TCTRL = self.args.type_of_controller if self.args.type_of_controller is not None else 'NC'
        TTRAJ = self.args.type_of_trajectory if self.args.type_of_trajectory is not None else 'NT'
        FOR = self.args.type_of_dataset

        if not self.args.no_save and not self.collision:
            self.name_tensor = (str(self.seed) + '_' + str(self.valid_envs) + '_' + str(self.args.num_iters) + '_' + 
                        str(self.args.frequency).replace('.','') + '_' + G + '_' + F + '_'  +
                        RI + '_' + RV + '_' + RM + '_' + RCOM + '_' + RINR + '_'  +
                        RS + '_' + RD + '_' + RF + '_' + RAD + '_' + RCG + '_' + 
                        QF + '_' + IS + '_' + QR + '_' +  TINP + '_' + TCTRL + '_' + TTRAJ +'_' + FOR) 

            print("\nSimulation to be saved as:\n",self.name_tensor) 
            torch.save(self.tensors_from_isaacGym,f'{self.path}/data_tensors/{self.args.type_of_dataset}/{self.args.name_of_dataset}/{self.name_tensor}.pt')
            print("\nDataset saved with the above namespace")

    def save_metadata(self):
        """
        Saves metadata of the simulation in the specified .json file, metadata is accessed in
        assessment of the created dataset. 

        Parameters
        ---
            None

        Returns
        ---
            None

        """
        metadatanew = {
            "dataname" : self.args.name_of_dataset,
            "genname" : self.name_tensor,
            "genenvs" : self.valid_envs,
            "gentime" : self.generation_time
        }
        for k,v in self.metadata.items():
            v.append(metadatanew[k])
        with open(f'{self.path}/data_objects/{self.args.name_of_dataset}.json', 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, sort_keys=True, indent=4)
        
class postprocessor():
    """
    Custom postprocessor, used to generate plots of link torques and link position of selected
    environments in a given simulation. - refer to README.md

    Parameters
    ---
        joints (int) : number of joints used in generation
        coords (int) : number of coordinates used in generation
        args (dict) : arguments passed in during generation
        control_trajectory (torch.Tensor) : control input
        pose (torch.Tensor) : controlled output
        seed (int) : seed for randomness determinancy
        valid_envs (int) : valid environments after exlusions
        target (torch.Tensor) : control target for controller implementations ONLY
        rigidbodyprops (torch.Tensor) : rigid props of each asset
        path (string) : path to postprocess objects to
        controlgains (torch.Tensor) : controller gains if controller implemented
    """
    label_coordinates4D = ['x','y','z','$X$','$Y$','$Z$','$W$',
                            '$q_0$','$q_1$','$q_2$','$q_3$','$q_4$','$q_5$','$q_6$']
    label_coordinates6D = ['x','y','z','$e_{11}$','$e_{12}$','$e_{13}$','$e_{22}$','$e_{23}$','$e_{33}$',
                            '$q_0$','$q_1$','$q_2$','$q_3$','$q_4$','$q_5$','$q_6$']
    def __init__(self,
                 joints,
                 coords,
                 args,
                 control_trajectory,
                 pose,
                 seed,
                 valid_envs,
                 target,
                 rigidbodyprops,
                 path='.',
                 controlgains=None
                 ):
        self.joints = joints
        self.coords = coords
        self.args = args
        self.cartesian_control = args.type_of_controller=='pd' or args.type_of_controller=='pid' or args.type_of_controller=='cic' or self.args.type_of_controller=='osc'
        self.joint_control = not self.cartesian_control
        self.integral_terms = args.type_of_controller=='pid' or args.type_of_controller=='joint_pid' or args.type_of_controller=='cic'
        self.ct = control_trajectory
        self.ps = pose
        if controlgains is not None:
            self.kp = controlgains['kp']
            self.kd = controlgains['kv']
            self.ki = controlgains['ki'] if self.integral_terms else torch.zeros_like(self.kp)

            self.kp = self.kp.to('cpu').numpy()
            self.kd = self.kd.to('cpu').numpy()
            self.ki = self.ki.to('cpu').numpy()

        self.seed = seed
        self.valid_envs = valid_envs
        self.tr = target
        self.rbp = rigidbodyprops
        self.path = f'{path}/plots/{self.args.name_of_dataset}'
        print("\n Ready to Post-Process")

    def __str__(self):
        return f'Post Processor Object instantiated'
    
    def getdata(self):
        return self.args

    def setdata(self,args_):
        self.args = args_

    def plot_linkmassdist(self):
        """
        Plots link mass distribution as a function of iteration steps.
        """
        mass = torch.tensor([[link.mass for link in env] for env in self.rbp],device='cpu')
        fig, axs = plt.subplots(mass.shape[2], figsize=(20,20))  
        fig.suptitle(f'Masses of Each Link, {self.joints+2} Links') 
        for i in range(mass.shape[2]):  
            temporary=mass[:,:,i].to("cpu").numpy()    
            axs[i].grid()
            axs[i].set(xlabel='Iteration Steps')
            axs[i].set(ylabel='Mass [kg]', title='Link'+str(i)+'')
            axs[i].plot(temporary)
        fig.tight_layout(pad=3) 
        fig.subplots_adjust(top = .96)
        plt.draw()
        fig.savefig(f'{self.path}/linkmass_dist.png',bbox_inches='tight')        
        plt.show()  

    def plot_control(self):
        """
        Plots the generated control trajectory for a given number of randomized envs for all
        joints and all envs.
        """
        fig, axs = plt.subplots(self.joints, figsize=(20,20))
        fig.suptitle(f'Control Action Among Dofs, {self.joints} Joints')
        for i in range(self.joints):
            temporary=self.ct[1:,:,i].to("cpu").numpy()   
            if i <=self.joints-1:
                axs[i].set(ylabel='Control Action [Nm, deg, deg/s]', title='Joint'+str(i)+'')
                axs[i].plot(temporary)
            axs[i].grid()
            axs[i].set(xlabel='Iteration Steps')
        fig.tight_layout(pad=3)
        fig.subplots_adjust(top = .96)
        plt.draw()
        fig.savefig(f'{self.path}/buffer_control.png',bbox_inches='tight')
        plt.show()
        
    def plot_trajectory(self):
        """
        Plots the end effector pose and joint variables in 13/14/16 dims for randomized envs for all envs.
        """
        if self.args.orientation_dimension=='4D':
            fig, axs = plt.subplots(int(self.coords/2),2,figsize=(20,20)) 
            fig.suptitle('Output: Full Pose and Joint Positions')
            k = 0
            for j in range(2):
                for i in range(int(self.coords/2)): 
                    if k <=2:
                        axs[i,j].plot(self.ps[:,:,k].to("cpu").numpy())
                        axs[i,j].set(ylabel='m', title=self.label_coordinates4D[k])
                    elif k>2 and k<=6:
                        axs[i,j].plot(self.ps[:,:,k].to("cpu").numpy())
                        axs[i,j].set(ylabel='[-]', title=self.label_coordinates4D[k])
                    else:
                        axs[i,j].plot(np.rad2deg(self.ps[:,:,k].to("cpu").numpy()))
                        axs[i,j].set(ylabel='deg', title=self.label_coordinates4D[k])
                    if not self.tr.shape[0]==0:
                        if k<=6 and self.cartesian_control:
                            axs[i,j].plot(self.tr[:,:,k].to("cpu").numpy(),'r-',label='target')
                        elif k>6 and self.joint_control:
                            axs[i,j].plot(self.tr[:,:,k-7].to("cpu").numpy()*180/np.pi,'r-',label='target')
                    axs[i,j].grid()
                    k = k+1
                    axs[i,0].set(xlabel='Iteration Steps')
                    axs[i,1].set(xlabel='Iteration Steps')
            fig.subplots_adjust(top = .96)
            fig.tight_layout(pad=3)
            fig.savefig(f'{self.path}/buffer_pose.png',bbox_inches='tight')
            plt.show()

        elif self.args.orientation_dimension=='6D' or self.args.orientation_dimension=='3D':
            fig = plt.figure(figsize=(20,20))
            gs = grid.GridSpec(9,2, figure=fig)
            fig.suptitle('Output: Full Pose and Joint Positions')
            k = 0
            for i1 in range(9):
                axs = fig.add_subplot(gs[i1,0])
                if k <=2:
                    axs.plot(self.ps[:,:,k].to("cpu").numpy())
                    axs.set(ylabel='m', title=self.label_coordinates6D[k])
                    axs.grid()

                elif k>2 and k<=8:
                    axs.plot(self.ps[:,:,k].to("cpu").numpy())
                    axs.set(ylabel='[-]', title=self.label_coordinates6D[k])
                    axs.grid()

                if not self.tr.shape[0]==0:
                    if k<=8 and self.cartesian_control:
                        axs[i,j].plot(self.tr[:,:,k].to("cpu").numpy(),'r-',label='target')

                k=k+1
                axs.set(xlabel='Iteration Steps')
            
            for i2 in range(7):
                axs = fig.add_subplot(gs[i2,1])
                axs.plot(np.rad2deg(self.ps[:,:,k].to("cpu").numpy()))
                axs.set(ylabel='deg', title=self.label_coordinates6D[k])
                axs.grid()

                if not self.tr.shape[0]==0:
                    if k>8 and self.joint_control:
                        axs[i,j].plot(self.tr[:,:,k-9].to("cpu").numpy()*180/np.pi,'r-',label='target')

                k=k+1
                axs.set(xlabel='Iteration Steps')

            fig.subplots_adjust(top = .96)
            fig.tight_layout(pad=3)
            plt.draw()
            fig.savefig(f'{self.path}/buffer_pose.png',bbox_inches='tight')
            plt.show()

    def plot_trajectory_wrt_control_gains(self):
        """
        Plots the end effector pose and joint variables in 13/14/16 dims for randomized envs for all envs.
        
        THIS SOLUTION IS MESSY, BUT IT WORKS...
        """
        if self.args.orientation_dimension=='4D':
            fig, axs = plt.subplots(int(self.coords/2),2,figsize=(20,20)) 
            fig.suptitle('Output: Full Pose and Joint Positions')
            k = 0
            for j in range(2):
                for i in range(int(self.coords/2)): 
                    if k <=2:
                        for env in range(self.valid_envs):
                            axs[i,j].plot(self.ps[:,env,k].to("cpu").numpy(),label=f'{np.array2string(self.kp[env,...], formatter={"float": lambda x: f"{x:.1f}"}, separator=", ")}'
                                                                                   f'{np.array2string(self.kd[env,...], formatter={"float": lambda x: f"{x:.1f}"}, separator=", ")}'
                                                                                   f'{np.array2string(self.ki[env,...], formatter={"float": lambda x: f"{x:.1f}"}, separator=", ")}' if k==0 else None)
                        axs[i,j].set(ylabel='m', title=self.label_coordinates4D[k])
                        axs[i,j].grid()
                    elif k>2 and k<=6:
                        for env in range(self.valid_envs):
                            axs[i,j].plot(self.ps[:,env,k].to("cpu").numpy())
                        axs[i,j].set(ylabel='[-]', title=self.label_coordinates4D[k])
                        axs[i,j].grid()
                    else:
                        for env in range(self.valid_envs):
                            axs[i,j].plot(np.rad2deg(self.ps[:,env,k].to("cpu").numpy()))
                        axs[i,j].set(ylabel='deg', title=self.label_coordinates4D[k])
                        axs[i,j].grid()
                        
                    k = k+1
                    axs[i,0].set(xlabel='Iteration Steps')
                    axs[i,1].set(xlabel='Iteration Steps')

            fig.legend()

            fig.subplots_adjust(top = .96)
            fig.tight_layout(pad=3)
            plt.draw()
            fig.savefig(f'{self.path}/buffer_pose.png',bbox_inches='tight')
            plt.show()

        elif self.args.orientation_dimension=='6D' or self.args.orientation_dimension=='3D':
            fig = plt.figure(figsize=(20,20))
            gs = grid.GridSpec(9,2, figure=fig)
            fig.suptitle('Output: Full Pose and Joint Positions')
            k = 0
            for i1 in range(9):
                axs = fig.add_subplot(gs[i1,0])
                if k <=2:
                    for env in range(self.valid_envs):
                        axs.plot(self.ps[:,env,k].to("cpu").numpy(),label=f'{np.array2string(self.kp[env,...], formatter={"float": lambda x: f"{x:.1f}"}, separator=", ")}'
                                                                                   f'{np.array2string(self.kd[env,...], formatter={"float": lambda x: f"{x:.1f}"}, separator=", ")}'
                                                                                   f'{np.array2string(self.ki[env,...], formatter={"float": lambda x: f"{x:.1f}"}, separator=", ")}' if k==0 else None)
                    axs.set(ylabel='m', title=self.label_coordinates6D[k])
                    axs.grid()

                elif k>2 and k<=8:
                    for env in range(self.valid_envs):
                        axs.plot(self.ps[:,env,k].to("cpu").numpy())
                    axs.set(ylabel='[-]', title=self.label_coordinates6D[k])
                    axs.grid()

                k=k+1
                axs.set(xlabel='Iteration Steps')
            
            for i2 in range(7):
                for env in range(self.valid_envs):
                    axs = fig.add_subplot(gs[i2,1])
                    axs.plot(np.rad2deg(self.ps[:,env,k].to("cpu").numpy()))
                axs.set(ylabel='deg', title=self.label_coordinates6D[k])
                axs.grid()
                    
                k=k+1
                axs.set(xlabel='Iteration Steps')

            fig.legend()

            fig.subplots_adjust(top = .96)
            fig.tight_layout(pad=3)
            plt.draw()
            fig.savefig(f'{self.path}/buffer_pose_wrt_controlgains.png',bbox_inches='tight')
            plt.show()

    def plot_saturation_histogram(self,pose,limits):
        """
        Plots the distribution of max positional reach of the end effector in each environment,
        used for probabilistic assessment of max pose distribution.
        
        Parameters
        ---
            pose (torch.Tensor) : controlled output tensor
            limits (torch.Tensor) : limits of positio/velocity
        """
        if self.args.orientation_dimension=='4D':
            pose = pose[:,:,7:]
        else:
            pose = pose[:,:,9:]
        diffl = pose - limits[0].repeat(pose.size()[0],1,1)
        diffu = pose - limits[1].repeat(pose.size()[0],1,1)
        diff = torch.abs(torch.maximum(diffl,diffu)).mean(dim=0).to("cpu").numpy()
        fig, axs = plt.subplots(self.joints, figsize=(20,20))
        fig.suptitle(f'Joint Saturation Probability, {self.joints} Total Joints')
        for i in range(self.joints):
            axs[i].set(ylabel=f'Saturation', title='Joint'+str(i)+'')
            axs[i].hist(diff[:,i:7], bins=40)
            axs[i].grid()
            axs[i].set(xlabel='Iteration Steps')
        fig.tight_layout(pad=3)
        fig.subplots_adjust(top = .96)
        plt.draw()
        fig.savefig(f'{self.path}/saturation_dist.png',bbox_inches='tight')
        plt.show()
        
    def plot_collision_histogram(self,pose):
        """
        Plots the distribution of max positional reach of the end effector in each environment,
        used for probabilistic assessment of max collision.

        Parameters
        ---
            pose (torch.Tensor) : controlled output tensor
        """
        diff = torch.abs(pose[:,:,2]).mean(dim=0).to("cpu").numpy()
        fig, axs = plt.subplots(figsize=(20,20))
        fig.suptitle(f'Joint Saturation Probability, {self.joints} Total Joints')
        axs.set(ylabel=f'Saturation', title='Collision Plot')
        axs.hist(diff[:], bins=10)
        axs.grid()
        axs.set(xlabel='Iteration Steps')
        fig.tight_layout(pad=3)
        fig.subplots_adjust(top = .96)
        plt.draw()
        fig.savefig(f'{self.path}/collision_dist.png',bbox_inches='tight')
        plt.show()
        
    def plot_secondary_var(self,var,varname):
        """
        Plots a secondary variable other than control and trajectory, possible candidates are:
        friction, gravity, etc. var size = (num envs, num joints, num iters).

        Parameters
        ---
            var (torch.Tensor) : variable to be plotted
            varname (string) : name of the variable
        """
        b = [-10, -10, -5, -10, -5, -5, -2.5, -2,5, -2.5]
        t = [10, 10, 5, 10, 5, 5, 2.5, 2,5, 2.5]
        varp = var.to("cpu").numpy()
        fig, axs = plt.subplots(self.joints, figsize=(20,20))
        fig.suptitle(f'Joint {varname} Variation, {self.joints} Total Joints')
        for i in range(self.joints):
            for j in range(self.valid_envs):
                axs[i].set(ylabel=f'{varname}', title='Joint'+str(i)+'')
                axs[i].plot(varp[j,i,1:])
            axs[i].grid()
            axs[i].set(xlabel='Iteration Steps')
            if varname=="benchmark_control_error":
                axs[i].set_ylim(b[i],t[i])
        fig.tight_layout(pad=3)
        fig.subplots_adjust(top = .96)
        plt.draw()
        fig.savefig(f'{self.path}/{varname}.png',bbox_inches='tight')
        plt.show()
     
def normalize_vector(x):
    """
    l1 vector norm.
    """
    return torch.linalg.vector_norm(x)

def cross_product(x,y):
    """
    alias for torch.linalg.cross.
    """
    return torch.linalg.cross(x,y)

def standardize_quaternion(quaternions):
    """
    preprocess quarternions.
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x): 
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def ortho6d_to_matrix(orn):
    """
    orthogonal 6D orientation matrix TO Euler Angles.

    Parameters
    ---
        orn (torch.Tensor) 6D orientation vector at a given iteration

    Returns
    ---
        matrix (torch.Tensor) orientation matrix of Euler Angles
    """
    x_raw = orn[:,0:3]
    y_raw = orn[:,3:6]

    x = normalize_vector(x_raw)
    z = cross_product(x,y_raw)
    z = normalize_vector(z)
    y = cross_product(z,x)
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2)
    return matrix

def quarternion_to_matrix(orn):
    """
    4D orientation matrix TO Euler Angles.

    Parameters
    ---
        orn (torch.Tensor) 4D orientation vector at a given iteration

    Returns
    ---
        matrix (torch.Tensor) orientation matrix of Euler Angles
    """
    r, i, j, k = torch.unbind(orn, -1)
    two_s = 2.0 / (orn * orn).sum(-1)
    o = torch.stack(
        (
            1 -two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 -two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (j * k - j * r),    
            two_s * (j * k + i * r),
            1 -two_s * (i * i + j * j)
        ),
        -1,
    )
    return o.reshape(orn.shape[:-1] + (3,3))
    
def decide_orientation(orn,dim):
    """
    parse desired orientation scheme to be used in generation from 4D

    Parameters
    ---
        orn (torch.Tensor) : orientation tensor of each iteration
        dim (string) : dimension to convert orientation

    Returns
    ---
        orn_final (torch.Tensor) : converted orientation tensor
    """
    if dim=='4D':
        return orn
    
    elif dim=='3D':
        return torch_utils.get_euler_xyz(orn)
    
    elif dim=='6D':
        mat = quarternion_to_matrix(orn)
        bath_dim = mat.size()[:-2]
        return mat[..., :2, :].clone().reshape(bath_dim + (6,))

def resolve_orientation(orn,dim):
    """
    parse desired orientation scheme to be used in generation from 6D

    Parameters
    ---
        orn (torch.Tensor) : orientation tensor of each iteration
        dim (string) : dimension to convert orientation

    Returns
    ---
        orn_final (torch.Tensor) : converted orientation tensor
    """
    if dim=='6D':
        return orn
    
    elif dim=='3D':
        mat = ortho6d_to_matrix(orn)
        batch = mat.shape[0]
        R = mat
        sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
        singular = sy<1e-6
        singular = singular.float()
            
        x = torch.atan2(R[:,2,1], R[:,2,2])
        y = torch.atan2(-R[:,2,0], sy)
        z = torch.atan2(R[:,1,0],R[:,0,0])
        
        xs = torch.atan2(-R[:,1,2], R[:,1,1])
        ys = torch.atan2(-R[:,2,0], sy)
        zs = R[:,1,0]*0

        out_euler = torch.FloatTensor(batch,3)
        out_euler[:,0] = x*(1-singular)+xs*singular
        out_euler[:,1] = y*(1-singular)+ys*singular
        out_euler[:,2] = z*(1-singular)+zs*singular      
        return out_euler

    elif dim=='4D':
        mat = ortho6d_to_matrix(orn)
        batch_dim = mat.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            mat.reshape(batch_dim + (9,)), dim=-1
        )
        q_abs = _sqrt_positive_part(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            )
        )
        quat_by_rijk = torch.stack(
            [
                torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
            ],
            dim=-2,
        )

        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        out = quat_candidates[
            F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
        ].reshape(batch_dim + (4,))
        return standardize_quaternion(out)
