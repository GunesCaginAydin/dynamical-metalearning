import torch
from matplotlib import pyplot as plt
import numpy as np
import math
from isaacgym import gymutil
from isaacgym.torch_utils import quat_conjugate,quat_mul
from genutil import *

class input():
    """
    Blueprint class for all control objects, inherited throughout action classes.

    Parameters
    ---
        num_envs (int) : num of environments in the current simulation
        num_iter (int) : num of iterations to be done inn the current simulation
        mum_joints (int) : num of asset joints
        num_coords (int) : num of coordinates of interes
        frequency (float) : master frequency of control action
        input_type (string) : input type (MS, CH, TRAPZ, IMP)
        mass_vector (torch.Tensor) : mass vector of each link - DEPRECATED
        args (dict) : arguments passed in generation
        poslim (np.array) : position limit min max
        vellim (np.array) : velocity limit max
        acclim (np.array) : acceleration limit max
        cposlim (np.array) : cartesian position limit
        cvellim (np.array) : cartesian velocity limit 
        cacclim (np.array) : cartesian acceleration limit
        cornlim (np.array) : cartesian orientation limit
        wellim (np.array) : cartesian angular velocity limit
        alfalim (np.array) : cartesian angular acceleration limit
    """
    def __init__(self,
                 num_envs, 
                 num_iter, 
                 num_joints,
                 num_coords, 
                 frequency,
                 rigidbodyprops,
                 args,
                 poslim=[], vellim=[], acclim=[],
                 cposlim=[], cvellim=[], cacclim=[],
                 cornlim=[], wellim=[],alfalim=[]
                 ):
        self.args = args

        self.num_envs = num_envs
        self.num_iter = num_iter
        self.num_joints = num_joints
        self.num_coords = num_coords
        self.frequency = frequency

        self.masses = torch.tensor([[link.mass for link in env] for env in rigidbodyprops],device=args.graphics_device_id)
        self.coms = torch.tensor([[[link.com.x,link.com.y,link.com.z] for link in env] for env in rigidbodyprops],device=args.graphics_device_id)
        self.inertias = torch.tensor([[[link.inertia.x.x, link.inertia.y.y, link.inertia.z.z] for link in env] for env in rigidbodyprops],device=args.graphics_device_id)

        self.poslim = poslim
        self.vellim = vellim
        self.acclim = acclim

        self.cposlim = cposlim
        self.cornlim = cornlim
        self.cvellim = cvellim
        self.wellim = wellim
        self.cacclim = cacclim
        self.alfalim = alfalim

        self.kp = torch.zeros(size=(self.num_envs,3), device=self.args.graphics_device_id)
        self.kv = torch.zeros(size=(self.num_envs,3), device=self.args.graphics_device_id)
        self.ki = torch.zeros(size=(self.num_envs,3), device=self.args.graphics_device_id)

        self.kpr = torch.zeros(size=(self.num_envs,3), device=self.args.graphics_device_id)
        self.kvr = torch.zeros(size=(self.num_envs,3), device=self.args.graphics_device_id)
        self.kir = torch.zeros(size=(self.num_envs,3), device=self.args.graphics_device_id)

        print(f"Control Action is acquired with median frequency {self.frequency*13/12}\n")

        time_step = torch.linspace(0, self.num_iter, self.num_iter)
        self.t = time_step.unsqueeze(1) * 1/60
        self.dt = 1/60

        self.buffer_control_action = torch.empty((1,self.num_envs,self.num_joints), dtype=torch.float32).to(device=self.args.graphics_device_id)
        self.control_action =  torch.empty((self.num_envs,self.num_joints,self.num_iter))
        #self.diff_action = torch.empty(self.num_envs,self.num_joints,self.num_iter)

        if self.args.orientation_dimension=='6D':
            self.buffer_position = torch.empty((0,self.num_envs,self.num_coords+2), dtype=torch.float32).to(device=self.args.graphics_device_id)
        elif self.args.orientation_dimension=='3D':
            self.buffer_position = torch.empty((0,self.num_envs,self.num_coords-1), dtype=torch.float32).to(device=self.args.graphics_device_id)
        elif self.args.orientation_dimension=='4D':
            self.buffer_position = torch.empty((0,self.num_envs,self.num_coords), dtype=torch.float32).to(device=self.args.graphics_device_id)
        #self.buffer_velocities = torch.empty((0,self.num_envs,self.num_joints), dtype=torch.float32).to(device=self.args.graphics_device_id) 

        if self.args.controller:
            if self.args.type_of_controller=='pid' or self.args.type_of_controller=='pd' or self.args.type_of_controller=='cic' or self.args.type_of_controller=='osc':
                self.buffer_target = torch.empty((0,self.num_envs,self.buffer_position.shape[2]-7), dtype=torch.float32).to(device=self.args.graphics_device_id)
                self.init_cartesiancontroller_constants()
            elif self.args.type_of_controller=='joint_pid' or self.args.type_of_controller=='joint_pd':
                self.buffer_target  = torch.empty((0,self.num_envs,self.num_joints), dtype=torch.float32).to(device=self.args.graphics_device_id)
        elif self.args.no_controller:
            self.buffer_target = torch.empty((0,), dtype=torch.float32)

        if self.args.measure_force:   
            self.measured_torque = torch.empty((1,self.num_envs,self.num_joints), dtype=torch.float32).to(device=self.args.graphics_device_id)
        else:
            self.measured_torque = torch.empty((0,), dtype=torch.float32)

        if self.args.measure_gravity_friction:
            self.buffer_friction = torch.empty((self.num_envs,self.num_joints,0), dtype=torch.float32).to(device=self.args.graphics_device_id) 
            self.buffer_gravity = torch.empty((self.num_envs,self.num_joints,0), dtype=torch.float32).to(device=self.args.graphics_device_id)
        else:
            self.buffer_friction = torch.empty((0,), dtype=torch.float32)
            self.buffer_gravity = torch.empty((0,), dtype=torch.float32)

    def init_cartesiancontroller_constants(self):
        if self.args.type_of_trajectory=='VS' or self.args.type_of_trajectory=='FS' or self.args.type_of_trajectory=='FC':
            if self.args.random_controller_gains:
                self._radius = torch.rand(1,self.num_envs).uniform_(0.01,0.10).to(device=self.args.graphics_device_id)
                self._period = torch.rand(1,self.num_envs).uniform_(20,100).to(device=self.args.graphics_device_id)
                self._z_speed = torch.rand(1,self.num_envs).uniform_(0.1,0.3).to(device=self.args.graphics_device_id)
                self._sign = torch.sign(torch.rand(1,self.num_envs).uniform_(-1,1)).to(device=self.args.graphics_device_id)
            #_offset =  torch.sign(torch.rand(1).uniform_(-0.3,0.3)).to(device=self.args.graphics_device_id)
            else:
                self._radius = torch.rand(1).uniform_(0.01,0.10).to(device=self.args.graphics_device_id)
                self._period = torch.rand(1).uniform_(20,100).to(device=self.args.graphics_device_id)
                self._z_speed = torch.rand(1).uniform_(0.1,0.3).to(device=self.args.graphics_device_id)
                self._sign = torch.sign(torch.rand(1).uniform_(-1,1)).to(device=self.args.graphics_device_id)                

        if self.args.type_of_trajectory=='MS':
            self._amp_ms = 2* torch.rand(self.num_envs,7,4).uniform_(-0.25, 0.25)
            self._dir_ms = torch.sign(torch.randint(low=-1,high=1,size=(self.num_envs,7)))
            self._freq_ms = 2 * np.pi * torch.rand(self.num_envs,7).uniform_(self.frequency/1.5, self.frequency*1.5)
            self.dirs = torch.randint(low=0,high=3,size=(self.num_envs,3))>=1
            for dir in self.dirs:
                if not torch.any(dir):
                    dir[torch.randint(low=0,high=3,size=(1,))] = True
                if torch.all(dir):
                    dir[torch.randint(low=0,high=3,size=(1,))] = False
        if self.args.type_of_trajectory=='CH':
            self._amp_ch = torch.rand(self.num_envs,7).uniform_(-0.25, 0.25)    
            self._dir_ch = torch.sign(torch.randint(low=-1,high=1,size=(self.num_envs,7)))
            self._phi_ch = torch.rand(self.num_envs,7).uniform_(-np.pi,np.pi)
            self._q0_ch = torch.rand(self.num_envs,7).uniform_(-.1, .1) 
            self._f1_ch = torch.rand(self.num_envs,7).uniform_(self.frequency/1.1,self.frequency*1.5)
            self._f2_ch = torch.rand(self.num_envs,7).uniform_(self.frequency/1.5, self.frequency*2)
            self.dirs = torch.randint(low=0,high=3,size=(self.num_envs,3))>=1
            for dir in self.dirs:
                if not torch.any(dir):
                    dir[torch.randint(low=0,high=3,size=(1,))] = True
                if torch.all(dir):
                    dir[torch.randint(low=0,high=3,size=(1,))] = False

        if self.args.type_of_trajectory=='IMP':
            self._mag_imp = torch.rand(self.num_envs,7).uniform_(-self.frequency*15,self.frequency*15)
            self._tu_imp = torch.randint(low=20,high=100,size=(1,))
            self._tl_imp = torch.randint(low=20,high=100,size=(1,))
            self._dir_imp = torch.sign(torch.randint(low=-1,high=1,size=(self.num_envs,7)))

            self.start = torch.ones(size=(1,))
            self.dt = torch.zeros(size=(1,))

            self.restlow = torch.ones(size=(1,))
            self.restup = torch.zeros(size=(1,))
            self.rise = torch.zeros(size=(1,))
            self.fall = torch.zeros(size=(1,))

        if self.args.type_of_trajectory=='TRAPZ':
            self._mag_tr = torch.rand(self.num_envs,7).uniform_(-self.frequency*15,self.frequency*15)
            self._tu_tr = torch.randint(low=20,high=100,size=(self.num_envs,7))
            self._tl_tr = torch.randint(low=20,high=100,size=(self.num_envs,7))
            self._dir_tr = torch.sign(torch.randint(low=-1,high=1,size=(self.num_envs,7)))
            self._a1_tr = torch.rand(self.num_envs,7).uniform_(0.1*self.cacclim,0.5*self.cacclim) *  self.dt
            self._a2_tr = torch.rand(self.num_envs,7).uniform_(0.1*self.cacclim,0.5*self.cacclim) *  self.dt
            self._a1_tr = torch.rand(self.num_envs,7).uniform_(0.1*self.alfalim,0.5*self.alfalim) *  self.dt
            self._a2_tr = torch.rand(self.num_envs,7).uniform_(0.1*self.alfalim,0.5*self.alfalim) *  self.dt

            self.trajectory_prev = torch.zeros_like(self._mag_tr)
            self.start = torch.ones(size=(1,))
            self.dt = torch.zeros(size=(1,))

            self.restlow = torch.ones(size=(1,))
            self.restup = torch.zeros(size=(1,))
            self.rise = torch.zeros(size=(1,))
            self.fall = torch.zeros(size=(1,))

        if self.args.type_of_trajectory=='push':
            self._v_push = torch.rand(self.num_envs,7).uniform_(0.25*self.cvellim,0.75*self.cvellim)
            self._xs_push = torch.rand(self.num_envs,7).uniform_(0.05*self.lower,0.25*self.lower)
            self._dir_push = torch.sign(torch.randint(low=-1,high=1,size=(self.num_envs,7)))

        if self.args.type_of_trajectory=='pick':
            self._v1_pick = torch.rand(self.num_envs,7).uniform_(0.25*self.cvellim,0.75*self.cvellim)
            self._v2_pick = torch.rand(self.num_envs,7).uniform_(0.25*self.cvellim,0.75*self.cvellim)
            self._loc_pick = torch.rand(self.num_envs,7).uniform_(0.05*self.lower,0.25*self.lower)
            self.picking = torch.zeros(size=(1,))
            self.pulling = torch.zeros(size=(1,))
            self._xs_pick = torch.rand(self.num_envs,7).uniform_(0.25*self.cvellim,0.75*self.cvellim)

    def getdata(self):
        datadict = {"envs":self.num_envs,
                    "iters":self.num_iter, 
                    "joints":self.num_joints,
                    "coords":self.num_coords,
                    "freq":self.frequency}
        return datadict

    def setdata(self,num_envs_, num_iter_, num_joints_, num_coords_, frequency_):
        self.num_envs = num_envs_
        self.num_iter = num_iter_
        self.num_joints = num_joints_
        self.num_coords = num_coords_
        self.frequency = frequency_

    def getcontrol(self):
        return {
            "ac" : self.control_action.contiguous(),
            #"acd" : self.control_diff.contiguous(),
            "bca" : self.buffer_control_action,
            #"bcad" : self.buffer_control_action[:,:,:self.num_joints],
            "bp" : self.buffer_position,
            #"bv" : self.buffer_velocities,
            "bt" : self.buffer_target,
            "bf" : self.buffer_friction,
            "bg" : self.buffer_gravity,
            "mt" : self.measured_torque
        }

    def setcontrol(self,control_action_):
        self.control_action = control_action_
        #self.control_diff = control_diff_

    def getcontrollergains(self):
        return {
            'kp' : self.kp_aug,
            'kv' : self.kv_aug,
            'ki' : self.ki_aug
        }

    def setcontrollergains(self,kp_,kv_,ki_):
        self.kp_aug = kp_
        self.kv_aug = kv_
        self.ki_aug = ki_

    def plot_trajectory(self, trajectory, num_envs, num_dofs):
        """
        Plots a generated trajectory for all dofs and envs
        """
        plt.figure(figsize=(15, 10))
        plt.rcParams.update({'font.size': 9})
        for i in range(num_envs):
            for j in range(num_dofs):
                plt.subplot(num_envs, num_dofs, i * num_dofs + j + 1)
                for k in range(num_dofs):
                    plt.grid(color='k', linestyle='-', linewidth=0.2)
                    plt.plot(self.time_steps(), trajectory[:][i][j].cpu().numpy(), alpha=1, linewidth=0.8)
                plt.title(f"Simulation {i+1}, DOF {j+1}")
                plt.xlabel("Time Steps")
                plt.ylabel("Position")
        plt.tight_layout()
        plt.show()
    
    def translation_error(self,desired,current):
        """
        Translational Error between the desired and current positions for controller implementations
        """
        return desired - current

    def orientation_error(self,desired,current,form=False):
        """
        Orientation Error between the desired and current orientation for controller implementations.
        """
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1) if form is False else q_r

    def sin(self, reduce=None):
        """
        Sinusoidal randomized trajectory.

        Parameters
        ---
            reduce (float) : multiplicative reduction in signal magnitude

        Returns
        ---
            action,diff (torch.Tensor) : control action throughout all the simulation
        """
        def sin_signal(self,reduce=None):
            t = self.t
            freq = 2 * np.pi * torch.rand(1).uniform_(self.frequency/1.5, self.frequency*1.5)
            if self.args.type_of_controller=='joint_pid' or self.args.type_of_controller=='joint_pd':
                start = (self.poslim[j,0]+self.poslim[j,1])/2
                a = torch.rand(4).uniform_(self.poslim[j,0],self.poslim[j,1])*0.1

                _trajectory = start +  (a[0]*torch.sin(freq*t) 
                                + a[1] * torch.cos(freq*1.5*t) + a[2] *torch.sin(freq*2*t) 
                                +a[3] * torch.cos(freq*3*t))/reduce   
            else:
                r1,r2 = -1,1
                a = 2* torch.rand(4).uniform_(-self.frequency*15,self.frequency*15)
                _trajectory =  torch.sign(torch.rand(1).uniform_(r1,r2)) * (a[0]*torch.sin(freq*t) 
                                    + a[1] * torch.cos(freq*1.5*t) + a[2] *torch.sin(freq*2*t) 
                                    +torch.sign(torch.rand(1).uniform_(r1,r2))* a[3] * torch.cos(freq*3*t))/reduce     
                     
            trajectory = _trajectory.view(self.num_iter) 
            return trajectory
        
        for i in range(self.num_envs):
            for j in range(self.num_joints):
                if self.args.type_of_robot=='franka':
                    if j ==8 or j==7:
                        attenuation_factor = np.inf
                    elif j == 1:
                        attenuation_factor = 1.3 if reduce is None else reduce# 1.3 # 2 for real
                    else:
                        attenuation_factor = 1.6 if reduce is None else reduce# 1.6 # 3 for real    
                elif self.args.type_of_robot=='kuka':     
                    if j ==8 or j==7:
                        attenuation_factor = np.inf
                    elif j == 1:
                        attenuation_factor = 1.3 if reduce is None else reduce # 1.3 # 2 for real
                    else:
                        attenuation_factor = 1.6 if reduce is None else reduce # 1.6 # 3 for real          
                single_dof = sin_signal(self,attenuation_factor)
                self.control_action[i][j] = single_dof
                #self.diff_action[i][j][1:] = torch.diff(self.control_action[i][j])      
        return self.control_action.to(device=self.args.graphics_device_id)
        
    def chirp(self, reduce=None):
        """
        Chirp-like randomized trajectory.

        Parameters
        ---
            reduce (float) : multiplicative reduction in signal magnitude

        Returns
        ---
            action,diff (torch.Tensor) : control action throughout all the simulation
        """
        def chirp_signal(self,reduce=None):
            t = self.t
            phi = torch.rand(1).uniform_(-np.pi,np.pi)
            if self.frequency < 0.3:
                f1 = torch.rand(1).uniform_(self.frequency/1.1,self.frequency*1.5)
                f2 = torch.rand(1).uniform_(self.frequency/1.5, self.frequency*2)
            else:
                f1 = torch.rand(1).uniform_(self.frequency/1.3,self.frequency/1.2)
                f2 = torch.rand(1).uniform_(self.frequency/1.1,self.frequency*1.1)

            if self.args.type_of_controller=='joint_pid' or self.args.type_of_controller=='joint_pd':
                start = (self.poslim[j,0]+self.poslim[j,1])/2
                a = 1  #  [ -3,3] 
                _trajectory = start + a * torch.cos (2* np.pi * f1/2 *( 1 + 1/4 * torch.cos(  2 * np.pi * f2/50* t))*t + phi)
            else:
                q0 = torch.rand(1).uniform_(-.5, .5) 
                a = torch.rand(1).uniform_(-4,4)    #  [ -3,3]  
                _trajectory = q0 + torch.sign(torch.rand(1).uniform_(-1,1))* a * torch.cos (2* np.pi * f1 *( 1 + 1/4 * torch.cos(  2 * np.pi * f2* t))*t + phi)
            trajectory = _trajectory.view(self.num_iter)/reduce 
            return trajectory

        for i in range(self.num_envs):
            for j in range(self.num_joints):
                if self.args.type_of_robot=='franka':
                    if j ==8 or j==7:
                        attenuation_factor=np.inf
                    else: 
                        attenuation_factor = 2.0 if reduce is None else reduce# 1.6 safe
                elif self.args.type_of_robot=='kuka':
                    if j ==8 or j==7:
                        attenuation_factor=np.inf
                    else: 
                        attenuation_factor = 5.0 if reduce is None else reduce# 1.6 safe                    
                single_dof = chirp_signal(self,attenuation_factor)
                self.control_action[i][j] = single_dof
                #self.diff_action[i][j][1:] = torch.diff(self.control_action[i][j])
        return self.control_action.to(device=self.args.graphics_device_id)
    
    def impulse(self, reduce=None):
        """
        Impulse randomized trajectory.

        Parameters
        ---
            reduce (float) : multiplicative reduction in signal magnitude

        Returns
        ---
            action,diff (torch.Tensor) : control action throughout all the simulation
        """
        def imp_signal(lower,upper,reduce=None):
            t = self.t
            trajectory_iter = torch.zeros(size=(1,))
            
            start = torch.ones(size=(1,))
            dt = torch.zeros(size=(1,))
            restlow = torch.ones(size=(1,))
            restup = torch.zeros(size=(1,))

            for t1 in t:
                t1 = int(t1)

                if start:
                    _mag = torch.rand(1).uniform_(0.25*lower,0.75*upper) # same _mag for all iterations
                    _tup = torch.randint(low=20,high=100,size=(1,))
                    _tlow = torch.randint(low=20,high=100,size=(1,))

                if restlow:
                    start = torch.zeros(size=(1,))
                    dt = dt+1
                    trajectory_cur = torch.zeros(size=(1,))
                    trajectory_iter = torch.cat((trajectory_iter,trajectory_cur),dim=0)
                    if dt==_tlow:
                        dir = torch.sign(torch.rand(1).uniform_(-1,1))
                        dt = torch.zeros(size=(1,))
                        restlow = torch.zeros(size=(1,))
                        restup = torch.ones(size=(1,))

                elif restup:
                    dt = dt+1
                    trajectory_cur = dir*_mag
                    trajectory_iter = torch.cat((trajectory_iter,trajectory_cur),dim=0)
                    if dt==_tup:
                        dt = torch.zeros(size=(1,))
                        restlow = torch.ones(size=(1,))
                        restup = torch.zeros(size=(1,))
                        start = torch.ones(size=(1,))

            trajectory = torch.Tensor(trajectory_iter)/reduce
            trajectory = trajectory[1:].view(self.num_iter)
            return trajectory

        for i in range(self.num_envs):
            for j in range(self.num_joints):
                if self.args.type_of_robot=='franka':
                    if j ==8 or j==7:
                        attenuation_factor=np.inf
                    else: 
                        attenuation_factor = 1.5 if reduce is None else reduce # 1.6 safe
                if self.args.type_of_robot=='kuka':
                    if j ==8 or j==7:
                        attenuation_factor=np.inf
                    else:
                        attenuation_factor = 5.0 if reduce is None else reduce # 1.6 safe

                single_dof = imp_signal(lower=self.poslim[j,0],upper=self.poslim[j,1],reduce=attenuation_factor)
                self.control_action[i][j] = single_dof
                #self.diff_action[i][j][1:] = torch.diff(self.control_action[i][j])     
        return self.control_action.to(device=self.args.graphics_device_id)

    def trapz(self, reduce=None):
        """
        Trapezoidal randomized trajectory.

        Parameters
        ---
            reduce (float) : multiplicative reduction in signal magnitude

        Returns
        ---
            action,diff (torch.Tensor) : control action throughout all the simulation
        """
        def trapz_signal(lower,upper,acclim, reduce=None):
            t = self.t
            trajectory_iter = torch.zeros(size=(1,))
            
            start = torch.ones(size=(1,))
            dt = torch.zeros(size=(1,))

            restlow = torch.ones(size=(1,))
            restup = torch.zeros(size=(1,))
            rise = torch.zeros(size=(1,))
            fall = torch.zeros(size=(1,))

            dir = torch.sign(torch.rand(1).uniform_(-1,1))

            for t1 in t:
                t1 = int(t1)

                if start:
                    _ang1 = torch.rand(1).uniform_(0.5*acclim,0.75*acclim) *  1/60 # same _ang1 for all iterations
                    _ang2 = torch.rand(1).uniform_(0.5*acclim,0.75*acclim) *  1/60 # same _ang2 for all iterations
                    _mag = torch.rand(1).uniform_(0.3*upper,0.7*upper) # same _mag for all iterations
                    _tlow = torch.randint(low=20,high=50,size=(1,))
                    _tup = torch.randint(low=20,high=150,size=(1,))

                if restlow and not rise:
                    dt = dt+1
                    start = torch.zeros(size=(1,))
                    trajectory_cur = torch.zeros(size=(1,))
                    trajectory_iter = torch.cat((trajectory_iter,trajectory_cur),dim=0)
                    if dt>=_tlow:
                        rise = torch.ones(size=(1,))

                elif rise:
                    dt = torch.zeros(size=(1,))
                    restlow = torch.zeros(size=(1,))
                    trajectory_cur = dir*_ang1*1 + trajectory_iter[t1]
                    trajectory_iter = torch.cat((trajectory_iter,trajectory_cur),dim=0)   
                    if abs(dir*_mag-trajectory_iter[t1])<_ang1:
                        rise = torch.zeros(size=(1,))
                        restup = torch.ones(size=(1,))

                elif restup and not fall:
                    dt = dt+1
                    trajectory_cur = dir*_mag
                    trajectory_iter = torch.cat((trajectory_iter,trajectory_cur),dim=0)
                    if dt>=_tup:
                        fall = torch.ones(size=(1,))

                elif fall:
                    dt = torch.zeros(size=(1,))
                    restup = torch.zeros(size=(1,))
                    trajectory_cur = -dir*_ang2*1 + trajectory_iter[t1] 
                    trajectory_iter = torch.cat((trajectory_iter,trajectory_cur),dim=0)           
                    if  abs(trajectory_iter[t1])<_ang2:
                        fall = torch.zeros(size=(1,))
                        restlow = torch.ones(size=(1,))
                        start = torch.ones(size=(1,))
                        dir = dir*-1

            trajectory = torch.Tensor(trajectory_iter)/reduce
            trajectory = trajectory[1:].view(self.num_iter)
            return trajectory

        for i in range(self.num_envs):
            for j in range(self.num_joints):
                if self.args.type_of_robot=='franka':
                    if j ==8 or j==7:
                        attenuation_factor=np.inf
                    else: 
                        attenuation_factor = 1.5 if reduce is None else reduce # 1.6 safe
                if self.args.type_of_robot=='kuka':
                    if j ==8 or j==7:
                        attenuation_factor=np.inf
                    else: 
                        attenuation_factor = 5.0 if reduce is None else reduce # 1.6 safe
                single_dof = trapz_signal(lower=0,upper=self.vellim[j],acclim=self.acclim[j],reduce=attenuation_factor)
                self.control_action[i][j] = single_dof
                #self.diff_action[i][j][1:] = torch.diff(self.control_action[i][j])     
        return self.control_action.to(device=self.args.graphics_device_id)

    def vertical_spiral(self,posd,posi,itr):
        """
        Vertical Spiral OSC control with vertical spiral target.
        
        Parameters
        ---
            posd (torch.Tensor) : desired position tensor from previous iteration
            posi (torch.Tensor) : initial position
            itr (int) : current iteration

        Returns
        ---
            posd (torch.Tensor) : desired position
        """
        posd[:, 0] = posi[:, 0] + torch.sin(itr / self._period) * self._radius 
        posd[:, 1] = posi[:, 1] + torch.cos(itr / self._period) * self._radius
        posd[:, 2] = posi[:, 2] - 0.1 + self._sign * self._z_speed * itr/self.num_iter
        return posd
    
    def fixed_spiral(self,posd,posi,itr):  
        """
        Fixed Spiral OSC control with fixed spiral target.
        
        Parameters
        ---
            posd (torch.Tensor) : desired position tensor from previous iteration
            posi (torch.Tensor) : initial position
            itr (int) : current iteration

        Returns
        ---
            posd (torch.Tensor) : desired position
        """
        posd[:, 0] = posi[:, 0] + torch.sin(itr / self._period) * 0.05
        posd[:, 1] = posi[:, 1] + torch.cos(itr / self._period) * 0.05
        posd[:, 2] = posi[:, 2] + - 0.1 + 0.2 * itr/self.num_iter           
        return posd
    
    def fixed_circular(self,posd,posi,itr):
        """
        Fixed Circular OSC control with fixed circular target.
        
        Parameters
        ---
            posd (torch.Tensor) : desired position tensor from previous iteration
            posi (torch.Tensor) : initial position
            itr (int) : current iteration

        Returns
        ---
            posd (torch.Tensor) : desired position
        """
        posd[:, 0] = posi[:, 0] 
        posd[:, 1] = posi[:, 1] + torch.sin(itr / self._period) * self._radius 
        posd[:, 2] = posi[:, 2] + torch.cos(itr / self._period) * self._radius
        return posd
    
    def sin_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        """
        Cartesian multisinusoidal for randomized signals in position and orientation.

        Parameters
        ---
            posd (torch.Tensor) : desired position tensor from previous iteration
            posi (torch.Tensor) : initial position
            ornd (torch.Tensor) : desired orientation tensor from previous iteration
            orni (torch.Tensor) : initial orientation
            itr (int) : current iteration
            reduce (float) : signal reduction

        Returns
        ---
            action,diff (torch.Tensor) : control action throughout all the simulation
        """
        t = itr*self.dt
        if self.args.type_of_robot=='franka':
            reduce = 10.0 if reduce is None else reduce # 1.6 safe
        elif self.args.type_of_robot=='kuka':
            reduce = 5.0 if reduce is None else reduce # 1.6 safe                      

        _trajectory = self._dir_ms * (self._amp_ms[...,0]*torch.sin(self._freq_ms*t) 
                            + self._amp_ms[...,1] * torch.cos(self._freq_ms*1.5*t) + self._amp_ms[...,2] *torch.sin(self._freq_ms*2*t) 
                            + self._dir_ms* self._amp_ms[...,3] * torch.cos(self._freq_ms*3*t))/reduce

        trajectory_pos = _trajectory[:,:3]
        trajectory_pos[self.dirs] = trajectory_pos[self.dirs]
        trajectory_orn = _trajectory[:,3:].zero_()

        posd = posi + trajectory_pos.to(device=self.args.graphics_device_id)
        ornd = orni + trajectory_orn.to(device=self.args.graphics_device_id)

        return torch.cat((posd,ornd),dim=1)

    def chirp_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        """
        Cartesian chirp for randomized signals in position and orientation.

        Parameters
        ---
            posd (torch.Tensor) : desired position tensor from previous iteration
            posi (torch.Tensor) : initial position
            ornd (torch.Tensor) : desired orientation tensor from previous iteration
            orni (torch.Tensor) : initial orientation
            itr (int) : current iteration
            reduce (float) : signal reduction

        Returns
        ---
            action,diff (torch.Tensor) : control action throughout all the simulation
        """
        t = itr*self.dt
        if self.args.type_of_robot=='franka':
            reduce = 3.0 if reduce is None else reduce # 1.6 safe
        elif self.args.type_of_robot=='kuka':
            reduce = 5.0 if reduce is None else reduce # 1.6 safe  

        _trajectory = self._q0_ch + self._dir_ch* self._amp_ch * torch.cos (2* np.pi * self._f1_ch *( 1 + 1/4 * torch.cos(  2 * np.pi * self._f2_ch* t))*t + self._phi_ch)/reduce   
    
        trajectory_pos = _trajectory[:,:3]
        trajectory_pos[self.dirs] = trajectory_pos[self.dirs]
        trajectory_orn = _trajectory[:,3:].zero_()

        posd = posi + trajectory_pos.to(device=self.args.graphics_device_id)
        ornd = orni + trajectory_orn.to(device=self.args.graphics_device_id)

        return torch.cat((posd,ornd),dim=1)
    
    def trapz_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        """
        Trapezoidal randomized trajectory in cartesian space for randomized spikes in position and orientation.

        Parameters
        ---
            posd (torch.Tensor) : desired position tensor from previous iteration
            posi (torch.Tensor) : initial position
            ornd (torch.Tensor) : desired orientation tensor from previous iteration
            orni (torch.Tensor) : initial orientation
            itr (int) : current iteration
            reduce (float) : signal reduction

        Returns
        ---
            action,diff (torch.Tensor) : control action throughout all the simulation
        """
        t = itr*self.dt

        if self.args.type_of_robot=='franka':
            reduce = 1.5 if reduce is None else reduce # 1.6 safe
        elif self.args.type_of_robot=='kuka':
            reduce = 5.0 if reduce is None else reduce # 1.6 safe  

        t1 = int(t)

        if self.start:
            pass

        if self.restlow and not self.rise:
            self.start = torch.zeros(size=(1,))
            self.dt = self.dt+1
            trajectory_iter = torch.zeros(size=(1,))
            if self.dt==self._tl_tr:
                self.rise = torch.ones(size=(1,))

        elif self.rise:
            self.dt = torch.zeros(size=(1,))
            self.restlow = torch.zeros(size=(1,))
            trajectory_iter = self._dir_tr*self._a1_tr*1 + self.trajectory_prev
            if abs(self._dir_tr*self._mag_tr-self.trajectory_prev)<self._a1_tr:
                self.rise = torch.zeros(size=(1,))
                self.restup = torch.ones(size=(1,))

        elif self.restup and not self.fall:
            self.dt = self.dt+1
            trajectory_iter = self._dir_tr*self._mag_tr
            if self.dt==self._tu_tr:
                self.fall = torch.ones(size=(1,))

        elif self.fall:
            self.dt = torch.zeros(size=(1,))
            self.restup = torch.zeros(size=(1,))
            trajectory_iter = -self._dir_tr*self._a2_tr*1 + self.trajectory_prev
            if  abs(self.trajectory_prev)<self._a2_tr:
                self.fall = torch.zeros(size=(1,))
                self.restlow = torch.ones(size=(1,))
                self.start = torch.ones(size=(1,))
                self._dir_tr = self._dir_tr*-1

        self.trajectory_prev = trajectory_iter
        _trajectory = torch.Tensor(trajectory_iter)/reduce

        trajectory_pos = _trajectory[:,:3]/reduce/40
        trajectory_orn = _trajectory[:,3:]/reduce/50 

        posd = posi + trajectory_pos.to(device=self.args.graphics_device_id)
        ornd = orni + trajectory_orn.to(device=self.args.graphics_device_id)

        return torch.cat((posd,ornd),dim=1)
    
    def impulse_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        """
        Impulse randomized trajectory in cartesian space for randomized spikes in position and orientation.

        Parameters
        ---
            posd (torch.Tensor) : desired position tensor from previous iteration
            posi (torch.Tensor) : initial position
            ornd (torch.Tensor) : desired orientation tensor from previous iteration
            orni (torch.Tensor) : initial orientation
            itr (int) : current iteration
            reduce (float) : signal reduction

        Returns
        ---
            action,diff (torch.Tensor) : control action throughout all the simulation
        """
        t = itr*self.dt

        if self.args.type_of_robot=='franka':
            reduce = 1.5 if reduce is None else reduce # 1.6 safe
        elif self.args.type_of_robot=='kuka':
            reduce = 5.0 if reduce is None else reduce # 1.6 safe  

        t1 = int(t)
        
        if self.start:
            pass

        if self.restlow:
            self.start = torch.zeros(size=(1,))
            self.dt = self.dt+1
            trajectory_iter = torch.zeros(size=(self.num_envs,7))
            if self.dt==self._tl_imp:
                self._dir_imp = torch.sign(torch.rand(1).uniform_(-1,1))
                self.dt = torch.zeros(size=(1,))
                self.restlow_imp = torch.zeros(size=(1,))
                self.restup = torch.ones(size=(1,))

        elif self.restup:
            self.dt = self.dt+1
            trajectory_iter = self._dir_imp*self._mag_imp
            if self.dt==self._tu_imp:
                self.dt = torch.zeros(size=(1,))
                self.restlow = torch.ones(size=(1,))
                self.restup = torch.zeros(size=(1,))
                self.start = torch.ones(size=(1,))

        _trajectory = trajectory_iter/reduce
        print(_trajectory)
        trajectory_pos = _trajectory[:,:3]/reduce/40
        trajectory_orn = _trajectory[:,3:]/reduce/50 

        posd = posi + trajectory_pos.to(device=self.args.graphics_device_id)
        ornd = orni + trajectory_orn.to(device=self.args.graphics_device_id)

        return torch.cat((posd,ornd),dim=1)

    def pickobject(self, posd, ornd, posi, orni, itr, reduce=None):
        """
        Franka Object Picking task imitation without any compliance task involved.
        """
        t = itr*self.dt
        if self.args.type_of_robot=='franka':
            reduce = 1.5 if reduce is None else reduce # 1.6 safe
        elif self.args.type_of_robot=='kuka':
            reduce = 5.0 if reduce is None else reduce # 1.6 safe  

        if self.picking:
            trajectory = self._xs_pick + self._dir_pick*self._v1_pick*t
        if self.pulling:
            trajectory = trajectory_pos + self._dir_pull*self._v2_pick*t

        trajectory_pos = trajectory[:,:3]   
        trajectory_orn = trajectory[:,3:]

        posd = posi + trajectory_pos.to(device=self.args.graphics_device_id)
        ornd = orni + trajectory_orn.to(device=self.args.graphics_device_id)

        return torch.cat((posd,ornd),dim=1)

    def pushobject(self, posd, ornd, posi, orni, itr, reduce=None):
        """
        Franka Object Push task imitation without any compliance task involved.
        """
        t = itr*self.dt
        if self.args.type_of_robot=='franka':
            reduce = 1.5 if reduce is None else reduce # 1.6 safe
        elif self.args.type_of_robot=='kuka':
            reduce = 5.0 if reduce is None else reduce # 1.6 safe  

        trajectory = self._xs_push + self._dir_push*self._v_push*t

        trajectory_pos = trajectory[:,:3]   
        trajectory_orn = trajectory[:,3:]

        posd = posi + trajectory_pos.to(device=self.args.graphics_device_id)
        ornd = orni + trajectory_orn.to(device=self.args.graphics_device_id)

        return torch.cat((posd,ornd),dim=1)
    
class action(input):
    """
    Action generator object for the generation and visualization of various randomized 
    input trajectories. Inherits from input class determining the regimes of application.

    Available trajectories are: 
    * sinusoidal: multi-sinusoidal trajectory randomized magnitudes, directions and freqs
    * chirp: chirp trajectory, randomized freqs
    * impulse: impulse trajectory with multiple rest and rise regimes
    * trapezoidal: trapezoidal velocity profile trajectory with multiple rest and rise regimes
    """

    def __init__(self,
                 num_envs, 
                 num_iter, 
                 num_joints,
                 num_coords, 
                 frequency, 
                 rigidbodyprops,
                 args,
                 poslim=[], vellim=[], acclim=[],
                 cposlim=[], cvellim=[], cacclim=[],
                 cornlim=[], wellim=[],alfalim=[]
                 ):
        super().__init__(num_envs, num_iter, num_joints, num_coords, frequency, rigidbodyprops, args,
                         poslim, vellim, acclim, cposlim, cvellim, cacclim, cornlim, wellim, alfalim)

        if self.args.type_of_joint_torque_profile == 'MS':
            self.control_action = self.sin()
        elif self.args.type_of_joint_torque_profile == 'CH':
            self.control_action = self.chirp()
        elif self.args.type_of_joint_torque_profile == "IMP":
            self.control_action = self.impulse()
        elif self.args.type_of_joint_torque_profile == "TRAPZ":
            self.control_action = self.trapz()
    
    def plot(self, trajectory, num_envs, num_dofs):
        return super().plot_trajectory(trajectory, num_envs, num_dofs)
    
    def getcontrol(self):
        return super().getcontrol()
    
    def setcontrol(self, control_action_):
        return super().setcontrol(control_action_) 

    def sin(self, reduce=None):
        return super().sin(reduce)
    
    def chirp(self, reduce=None):
        return super().chirp(reduce)
    
    def impulse(self, reduce=None):
        return super().impulse(reduce)
    
    def trapz(self, reduce=None):
        return super().trapz(reduce)
    
class osc(input):
    """
    Operational Space Control control input generator, generated action is determined by the arguments,
    inherits from input class determining the regime of application.

    Possible cartesian OSC Tasks are:

    * vertical spiral (VS)
    * fixed spiral (FS)
    * fixed circular (FC)
    * multi-sinusoidal (MS)
    * chirp (CH)
    * impulse (IMP)
    * trapezoidal (TRAPZ)
    * object pick (PICK)
    * object push (PUSH)
    """
    def __init__(self,
                 num_envs, 
                 num_iter, 
                 num_joints,
                 num_coords, 
                 frequency, 
                 rigidbodyprops,
                 args,                 
                 poslim=[], vellim=[], acclim=[],
                 cposlim=[], cvellim=[], cacclim=[],
                 cornlim=[], wellim=[],alfalim=[]):
        super().__init__(num_envs, num_iter, num_joints, num_coords, frequency, rigidbodyprops, args,
                         poslim, vellim, acclim, cposlim, cvellim, cacclim, cornlim, wellim, alfalim)

        if self.args.random_controller_gains:
            kp_lower_bound, kp_higher_bound = (2,5)
            kv_lower_bound, kv_higher_bound = (1.5,2*(5**0.5))
            print('\n OSC K randomization:\nKp -->\n'+str(kp_lower_bound)+'|'+str(kp_higher_bound) 
            +'\nKv -->\n' +str(kv_lower_bound)+'|'+str(kv_higher_bound))
            self.kp = torch.FloatTensor(self.args.num_envs,1).uniform_(kp_lower_bound,kp_higher_bound).to(device=self.args.graphics_device_id)
            self.kv = torch.FloatTensor(self.args.num_envs,1).uniform_(kv_lower_bound,kv_higher_bound).to(device=self.args.graphics_device_id)
        else:
            kp_nom = 3.5
            kv_nom = (1.5+2*(5**0.5))/2
            print('\n OSC K default:\nKp -->\n'+str(kp_nom) 
            +'\nKv -->\n' +str(kv_nom))
            self.kp = kp_nom*torch.ones(self.args.num_envs,1).to(device=self.args.graphics_device_id)
            self.kv = kv_nom*torch.ones(self.args.num_envs,1).to(device=self.args.graphics_device_id)

        self.kp_aug = self.kp # 1D
        self.kv_aug = self.kv # 1D
        self.ki_aug = self.ki # 1D
    
    def vertical_spiral(self, posd, posi, itr):
        return super().vertical_spiral(posd, posi, itr)
    
    def fixed_circular(self, posd, posi, itr):
        return super().fixed_circular(posd, posi, itr)
    
    def fixed_spiral(self, posd, posi, itr):
        return super().fixed_spiral(posd, posi, itr)

    def sin_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().sin_cartesian(posd, ornd, posi, orni, itr, reduce)
    
    def chirp_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().chirp_cartesian(posd, ornd, posi, orni, itr, reduce)
    
    def trapz_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().trapz_cartesian(posd, ornd, posi, orni, itr, reduce)
    
    def impulse_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().impulse_cartesian(posd, ornd, posi, orni, itr, reduce)
    
    def pushobject(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().pushobject(posd, ornd, posi, orni, itr, reduce)
    
    def pickobject(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().pickobject(posd, ornd, posi, orni, itr, reduce)

    def obtain_trajectory(self, pos=None, orn=None, ipos=None, iorn=None, itr=0):
        """
        Obtains tracking trajectory which defines the desired variables of the controller.

        Parameters
        ---
            pos (torch.Tensor) : current position
            orn (torch.Tensor) : current velocity
            ipos (torch.Tensor) : initial position
            iorn (torch.Tensor) : initial orientation
            itr (int) : current iteration

        Returns
        ---
            action (torch.Tensor) : control action for the current iteration, step must be called every iter
        """
        orn = decide_orientation(orn, self.args.orientation_dimension)

        if self.args.type_of_trajectory=='VS':
            trajectory = self.vertical_spiral(posd=pos,posi=ipos,itr=itr)
            trajectory = torch.cat((trajectory, orn),dim=1)
        elif self.args.type_of_trajectory=='FS':
            trajectory = self.fixed_spiral(posd=pos,posi=ipos,itr=itr)
            trajectory = torch.cat((trajectory, orn),dim=1)
        elif self.args.type_of_trajectory=='FC':
            trajectory = self.fixed_circular(posd=pos,posi=ipos,itr=itr)
            trajectory = torch.cat((trajectory, orn),dim=1)
        elif self.args.type_of_trajectory=='MS':
            trajectory = self.sin_cartesian(posd=pos,ornd=orn,posi=ipos,orni=iorn,itr=itr)
        elif self.args.type_of_trajectory=='CH':
            trajectory = self.chirp_cartesian(posd=pos,ornd=orn,posi=ipos,orni=iorn,itr=itr)
        elif self.args.type_of_trajectory=='IMP':
            trajectory = self.impulse_cartesian(posd=pos,ornd=orn,posi=ipos,orni=iorn,itr=itr)
        elif self.args.type_of_trajectory=='TRAPZ':
            trajectory = self.trapz_cartesian(posd=pos,ornd=orn,posi=ipos,orni=iorn,itr=itr)

        self.desired_trajectory = trajectory

        return self.desired_trajectory
    
    def step_osc(self, pos_current, orn_current, vel_current, 
                 eff_jacobian, franka_mass,
                 dpos=None, dorn=None):
        """
        Step OSC control to be called in each iteration controlled with OSC.

        Parameters
        ---
            pos_current (torch.Tensor) : current position
            vel_current (torch.Tensor) : current velocity
            orn_current (torch.Tensor) : current orientation
            eff_jacobian (torch.Tensor) : end-effector jacobian
            franka_mass (torch.Tensor) : mass vector of Franka asset
            dpos (torch.Tensor) : desired position of the current iteration (if not obtain_trajectory)
            dorn (torch.Tensor) : desired orientation of the current iteration (if not obtain_trajectory)

        Returns
        ---
            action (torch.Tensor) : control action for the current iteration, step must be called every iter
        """
        vel_current = vel_current.view(self.num_envs, self.num_joints, 1)

        pos_desired = self.desired_trajectory[:,:3] if dpos is None else dpos
        orn_desired = self.desired_trajectory[:,3:] if dorn is None else dorn
        
        m_inv = torch.inverse(franka_mass)  
        m_eef = torch.inverse(eff_jacobian @ m_inv @ torch.transpose(eff_jacobian, 1, 2)) 
        orn_current /= torch.norm(orn_current, dim=-1).unsqueeze(-1)
        pos_err = self.kp * (pos_desired - pos_current)
        dpose = torch.cat([pos_err, self.orientation_error(orn_desired,orn_current)], -1)
        self.control_action = self.kp.unsqueeze(-1) * (torch.transpose(eff_jacobian, 1, 2) @ m_eef @ (dpose.unsqueeze(-1))) - self.kv.unsqueeze(-1) * franka_mass @  vel_current
        #self.control_diff = self.control_action
        return self.control_action.to(self.args.graphics_device_id) 

class PID(input):
    """
    PID Controller for sim2real applications on Franka Emika Panda. PID Control is defined as:

        T_nom = K_i*dt*deps + K_d*ddeps + K_p*deps -> deps: 6D/7D pose error
    
    where K_d and K_p are the derivative and proportional control gains which act upon the nominal and desired
    differences between the velocity and position. Note that this implementation is inherently different from the
    internal isaacgym functionalities. The corresponding control is generated at each step of the simulation.

    Position error is 3D, orientation error is 3D (even though quarternions are used, this is directly taken
    from isaacgym's own implementation where W is ignored and projection error is estimated).

    Cartesian PID gains are obtained by trial and error. A more robust approach is to use some sort of
    identification/estimation procedure to autotune the PID. 

    Possible cartesian PID Tasks are:

    * vertical spiral (VS)
    * fixed spiral (FS)
    * fixed circular (FC)
    * multi-sinusoidal (MS)
    * chirp (CH)
    * impulse (IMP)
    * trapezoidal (TRAPZ)
    * object pick (PICK)
    * object push (PUSH)
    """
    def __init__(self,
                 num_envs, 
                 num_iter, 
                 num_joints,
                 num_coords, 
                 frequency, 
                 rigidbodyprops,
                 args,
                 poslim=[], vellim=[], acclim=[],
                 cposlim=[], cvellim=[], cacclim=[],
                 cornlim=[], wellim=[],alfalim=[]):
        super().__init__(num_envs, num_iter, num_joints, num_coords, frequency, rigidbodyprops, args,
                         poslim, vellim, acclim, cposlim, cvellim, cacclim, cornlim, wellim, alfalim)

        if self.args.random_controller_gains:
            kp_bounds = ((30,40),(30,40),(30,40)) if self.args.type_of_robot=='franka' else ((35,45),(35,45),(35,45))
            h_bounds = ((0.5,0.7),(0.5,0.7),(0.5,0.7)) # x,y,z
            kv_bounds = ((5.0,10.0),(5.0,10.0),(5.0,10.0)) if self.args.type_of_robot=='franka' else ((0.8,1.6),(0.8,1.6),(0.8,1.6))
            ki_bounds = ((0.2,0.5),(0.2,0.5),(0.2,0.5)) # x,y,z

            kpr_bounds = ((3,6),(3,6),(3,6)) if self.args.type_of_robot=='franka' else ((3.7,4.7),(3.7,4.7),(3.7,4.7))
            hr_bounds = ((0.5,0.7),(0.5,0.7),(0.5,0.7)) # X,Y,Z
            kvr_bounds = ((1.0,1.2),(1.0,1.2),(1.0,1.2)) if self.args.type_of_robot=='franka' else ((0.15,0.27),(0.15,0.27),(0.15,0.27))
            kir_bounds = ((0.2,0.5),(0.2,0.5),(0.2,0.5)) if self.args.type_of_robot=='franka' else ((0.10,0.20),(0.10,0.20),(0.10,0.20))

            print('\n PID gain randomization:\n\nPosition Gains\nKp -->\nx:'+str(kp_bounds[0])+'|y:'+str(kp_bounds[1])+'|z:'+str(kp_bounds[2])
            +'\nKd -->\nx:'+str(kv_bounds[0])+'|y:'+str(kv_bounds[1])+'|z:'+str(kv_bounds[2])
            +'\nKi -->\nx:'+str(ki_bounds[0])+'|y:'+str(ki_bounds[1])+'|z:'+str(ki_bounds[2])
            +'\n\nOrientation Gains\nKpr -->\neX:'+str(kpr_bounds[0])+'|Y:'+str(kpr_bounds[1])+'|Z:'+str(kpr_bounds[2])
            +'\nKvr -->\neX:'+str(kvr_bounds[0])+'|Y:'+str(kvr_bounds[1])+'|Z:'+str(kvr_bounds[2])
            +'\nKir -->\neX:'+str(kir_bounds[0])+'|Y:'+str(kir_bounds[1])+'|Z:'+str(kir_bounds[2]))

            self.kp = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in kp_bounds]).squeeze().T
            if self.args.damping_from_rigidprops:
                self.kv = self.dampingratio2kd(h_bounds,self.kp,self.masses,self.inertias)
            else:
                self.kv = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in kv_bounds]).squeeze().T
            self.ki = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in ki_bounds]).squeeze().T

            self.kpr = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in kpr_bounds]).squeeze().T
            if self.args.damping_from_rigidprops:
                self.kvr = self.dampingratio2kd(hr_bounds,self.kpr,self.masses,self.inertias)       
            else:
                self.kvr = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in kvr_bounds]).squeeze().T  
            self.kir = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in kir_bounds]).squeeze().T

        else:
            kp_nom = 50.0 if self.args.type_of_robot=='franka' else 40.0
            h_nom = ((0.7,0.7),(0.7,0.7),(0.7,0.7)) # x,y,z
            kv_nom = 15.0 if self.args.type_of_robot=='franka' else 1.2
            ki_nom = 0.3 # x,y,z

            kpr_nom = 10.0 if self.args.type_of_robot=='franka' else 4.2
            hr_nom = ((0.7,0.7),(0.7,0.7),(0.7,0.7)) # e11, e22, e33
            kvr_nom = 2.0 if self.args.type_of_robot=='franka' else 0.25
            kir_nom = 1.0 if self.args.type_of_robot=='franka' else 0.15

            print('\n PID gain nominal values:\n\nPosition Gains\nKp -->\nx:'+str(kp_nom)+'|y:'+str(kp_nom)+'|z:'+str(kp_nom)
            +'\nKd -->\nx:'+str(kv_nom)+'|y:'+str(kv_nom)+'|z:'+str(kv_nom)
            +'\nKi -->\nx:'+str(ki_nom)+'|y:'+str(ki_nom)+'|z:'+str(ki_nom)
            +'\n\nOrientation Gains\nKpr -->\neX:'+str(kpr_nom)+'|Y:'+str(kpr_nom)+'|Z:'+str(kpr_nom)
            +'\nKvr -->\neX:'+str(kvr_nom)+'|Y:'+str(kvr_nom)+'|Z:'+str(kvr_nom)
            +'\nKir -->\neX:'+str(kir_nom)+'|Y:'+str(kir_nom)+'|Z:'+str(kir_nom))

            self.kp = kp_nom*torch.ones(self.args.num_envs,3).to(device=self.args.graphics_device_id)
            if self.args.damping_from_rigidprops:
                self.kv = self.dampingratio2kd(h_nom,self.kp,self.masses,self.inertias)
            else:
                self.kv = kv_nom*torch.ones(self.args.num_envs,3).to(device=self.args.graphics_device_id) 
            self.ki = ki_nom*torch.ones(self.args.num_envs,3).to(device=self.args.graphics_device_id)

            self.kpr = kpr_nom*torch.ones(self.args.num_envs,3).to(device=self.args.graphics_device_id)
            if self.args.damping_from_rigidprops:
                self.kvr = self.dampingratio2kd(hr_nom,self.kpr,self.masses,self.inertias)
            else:
                self.kvr = kvr_nom*torch.ones(self.args.num_envs,3).to(device=self.args.graphics_device_id)   
            self.kir = kir_nom*torch.ones(self.args.num_envs,3).to(device=self.args.graphics_device_id)
        
        self.epos_prev = 0 # derivative previous error
        self.i = 0 # integral accumulator
        self.eorn_prev = 0 # derivative previous error
        self.ir = 0 # integral accumulator

        self.Tjoint_previous = torch.tensor(0) # filter previous accusation

        self.kp_aug = torch.cat((self.kp,self.kpr),dim=1) # 6D
        self.kv_aug = torch.cat((self.kv,self.kvr),dim=1) # 6D
        self.ki_aug = torch.cat((self.ki,self.kir),dim=1) # 6D

    def vertical_spiral(self, posd, posi, itr):
        return super().vertical_spiral(posd, posi, itr)
    
    def fixed_circular(self, posd, posi, itr):
        return super().fixed_circular(posd, posi, itr)
    
    def fixed_spiral(self, posd, posi, itr):
        return super().fixed_spiral(posd, posi, itr)
    
    def sin_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().sin_cartesian(posd, ornd, posi, orni, itr, reduce)
    
    def chirp_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().chirp_cartesian(posd, ornd, posi, orni, itr, reduce)
    
    def trapz_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().trapz_cartesian(posd, ornd, posi, orni, itr, reduce)
    
    def impulse_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().impulse_cartesian(posd, ornd, posi, orni, itr, reduce)
    
    def pushobject(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().pushobject(posd, ornd, posi, orni, itr, reduce)
    
    def pickobject(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().pickobject(posd, ornd, posi, orni, itr, reduce)

    def dampingratio2kd(self, h_bounds, kps, ms, Is):
        zetas = torch.stack([torch.FloatTensor(1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in h_bounds]).squeeze().T
        if len(zetas)==4:
            Is = torch.stack([torch.cat((I,torch.ones(size=(I.shape[0],1),device=self.args.graphics_device_id)),dim=1) for I in Is])
            return torch.stack([zetas*2*torch.sqrt(I[-4]*kp) for I,kp in zip(Is,kps)])
        elif len(zetas)==3:
            return torch.stack([zetas*2*torch.sqrt(m[-4]*kp) for m,kp in zip(ms,kps)])

    def obtaintrajectory(self, pos=None, orn=None, ipos=None, iorn=None, itr=0):
        """
        Obtains tracking trajectory which defines the desired variables of the controller.

        Parameters
        ---
            pos (torch.Tensor) : current position
            orn (torch.Tensor) : current velocity
            ipos (torch.Tensor) : initial position
            iorn (torch.Tensor) : initial orientation
            itr (int) : current iteration

        Returns
        ---
            action
        """
        orn = decide_orientation(orn, self.args.orientation_dimension)

        if self.args.type_of_trajectory=='VS':
            trajectory = self.vertical_spiral(posd=pos,posi=ipos,itr=itr)
            trajectory = torch.cat((trajectory, orn),dim=1)
        elif self.args.type_of_trajectory=='FS':
            trajectory = self.fixed_spiral(posd=pos,posi=ipos,itr=itr)
            trajectory = torch.cat((trajectory, orn),dim=1)
        elif self.args.type_of_trajectory=='FC':
            trajectory = self.fixed_circular(posd=pos,posi=ipos,itr=itr)
            trajectory = torch.cat((trajectory, orn),dim=1)
        elif self.args.type_of_trajectory=='MS':
            trajectory = self.sin_cartesian(posd=pos,ornd=orn,posi=ipos,orni=iorn,itr=itr)
        elif self.args.type_of_trajectory=='CH':
            trajectory = self.chirp_cartesian(posd=pos,ornd=orn,posi=ipos,orni=iorn,itr=itr)
        elif self.args.type_of_trajectory=='IMP':
            trajectory = self.impulse_cartesian(posd=pos,ornd=orn,posi=ipos,orni=iorn,itr=itr)
        elif self.args.type_of_trajectory=='TRAPZ':
            trajectory = self.trapz_cartesian(posd=pos,ornd=orn,posi=ipos,orni=iorn,itr=itr)

        self.desired_trajectory = trajectory

        return self.desired_trajectory

    def getcontrol(self):
        return super().getcontrol()
    
    def setcontrol(self, control_action_):
        return super().setcontrol(control_action_)
    
    def autotune(self, endgain, oscperiod):
        """
        Implements Ziegler-Nichols autotuner to set the nominal values of control gains.
        """
        Kp = 0.6*endgain
        Ti = 0.5*oscperiod
        Td = 0.125*oscperiod
        Ki = 1.2*endgain/Ti
        Kv = 0.075*endgain*Td

        return {
            'kp' : Kp,
            'kv' : Kv,
            'ki' : Ki
        }

    def singularity_check(self, jacobians, Sthreshold=1e-4, Cthreshold=1000, dls_damping=None):
        """
        Checks for singularities in the jacobian for infeasible velocity profiles using
        singular value decomposition.
        """
        singularity = False
        if dls_damping is not None:
            pass
        else:
            jacobians_crr = torch.zeros_like(jacobians)
            for env,jacobian in enumerate(jacobians):
                U, S, Vh  = torch.linalg.svd(jacobian, full_matrices=False)
                C = torch.max(S)/torch.min(S)
                if C>=Cthreshold:
                    print('possible singularity')
                for s in S:
                    if s<=Sthreshold:
                        print(s)
                        print('wow')
                        singularity = True
                        s=torch.inf
                jacobians_crr[env,...] = U @ torch.diag(S) @ Vh
        return jacobians_crr if singularity else jacobians

    def invert_jacobian(self, jacobians, dls_damping=0.0):
        """
        Inverts jacobian matrices for cartesian to joint space projection according to damped 
        least squares are pseudoinverse inversion.
        """
        jacobians_inv = torch.zeros_like(torch.transpose(jacobians,1,2))
        for env,jacobian in enumerate(jacobians):
            J = jacobian
            JT = jacobian.T
            JJT = J @ JT

            jacobians_inv[env,...] = JT @ torch.linalg.inv(JJT + dls_damping**2 * torch.eye(J.shape[0],device=J.device))

        return jacobians_inv
        
    def step_PD(self, cpos, corn, jacobian, torque_limits,
                dpos=None, dorn=None, dfriction=None,
                cutoff_coeff=0.0):
        """
        Step PD control to be called in each iteration controlled with PD.

        Parameters
        ---
            pos_current (torch.Tensor) : current position
            orn_current (torch.Tensor) : current orientation
            jacobian (torch.Tensor) : end-effector jacobian
            torque_limits (torch.Tensor/numpy.array) : joint torque limits
            dpos (torch.Tensor) : desired position of the current iteration (if not obtain_trajectory)
            dorn (torch.Tensor) : desired orientation of the current iteration (if not obtain_trajectory)
            dfriction (torch.Tensor) : desired friction per projected joint torques
            cutoff_coeff (float) : derivative controller low pass filter rate

        Returns
        ---
            action (torch.Tensor) : control action for the current iteration, step must be called every iter
        """
        torque_min = -1*torch.tensor(torque_limits,device=self.args.graphics_device_id).expand(self.num_envs,self.num_joints)
        torque_max = 1*torch.tensor(torque_limits,device=self.args.graphics_device_id).expand(self.num_envs,self.num_joints)

        dpos = self.desired_trajectory[:,:3] if dpos is None else dpos
        dorn = self.desired_trajectory[:,3:] if dorn is None else dorn

        epos = dpos - cpos
        derrpos = epos - self.epos_prev
        self.epos_prev = epos
        eorn = self.orientation_error(dorn, corn)
        derrorn = eorn - self.eorn_prev
        self.eorn_prev = eorn

        s = self.kp*epos
        d = self.kv*derrpos/self.dt

        sr = self.kpr*eorn
        dr = self.kvr*derrorn/self.dt

        f = 0 if dfriction is None else dfriction

        Tcartesian = (torch.cat((s,sr),dim=1) + \
                     torch.cat((d,dr),dim=1) + \
                     f).unsqueeze(-1)

        Tjoint = (torch.transpose(jacobian,1,2) @ Tcartesian).squeeze()

        if torch.any(Tjoint<=torque_min) or torch.any(Tjoint>=torque_max): # preclamping
            Tjoint = torch.clamp(Tjoint, min=0.1*torque_min, max=0.1*torque_max) 

        Tjoint_clamped = torch.clamp(Tjoint, min=torque_min, max=torque_max) 

        Tjoint_filtered = cutoff_coeff*self.Tjoint_previous + (1-cutoff_coeff)*Tjoint_clamped
        self.Tjoint_previous = Tjoint_filtered

        return Tjoint_filtered.unsqueeze(-1)
    
    def step_PID(self, cpos, corn, jacobian, torque_limits,
                dpos=None, dorn=None, dfriction=None,
                windup_limit=1000, cutoff_coeff=0.0):
        """
        Step PID control to be called in each iteration controlled with PID.

        Parameters
        ---
            pos_current (torch.Tensor) : current position
            orn_current (torch.Tensor) : current orientation
            jacobian (torch.Tensor) : end-effector jacobian
            torque_limits (torch.Tensor/numpy.array) : joint torque limits
            dpos (torch.Tensor) : desired position of the current iteration (if not obtain_trajectory)
            dorn (torch.Tensor) : desired orientation of the current iteration (if not obtain_trajectory)
            dfriction (torch.Tensor) : desired friction per projected joint torques
            windup_limit (int) : integral windup limit to prevent overflow
            cutoff_coeff (float) : derivative controller low pass filter rate

        Returns
        ---
            action (torch.Tensor) : control action for the current iteration, step must be called every iter
        """
        torque_min = -1*torch.tensor(torque_limits,device=self.args.graphics_device_id).expand(self.num_envs,self.num_joints)
        torque_max = 1*torch.tensor(torque_limits,device=self.args.graphics_device_id).expand(self.num_envs,self.num_joints)

        dpos = self.desired_trajectory[:,:3] if dpos is None else dpos
        dorn = self.desired_trajectory[:,3:] if dorn is None else dorn

        epos = dpos - cpos
        derrpos = epos - self.epos_prev
        self.epos_prev = epos
        eorn = self.orientation_error(dorn, corn)
        derrorn = eorn - self.eorn_prev
        self.eorn_prev = eorn

        s = self.kp*epos
        d = self.kv*derrpos/self.dt
        self.i += self.ki*epos*self.dt
        if torch.any(self.i >=windup_limit):
            self.i = windup_limit*torch.ones_like(self.i)
        if torch.any(self.i<= -windup_limit):
            self.i = -windup_limit*torch.ones_like(self.i)

        sr = self.kpr*eorn
        dr = self.kvr*derrorn/self.dt
        self.ir = self.kir*eorn*self.dt
        if torch.any(self.ir >=windup_limit):
            self.ir = windup_limit*torch.ones_like(self.ir)
        if torch.any(self.ir<= -windup_limit):
            self.ir = -windup_limit*torch.ones_like(self.ir)

        f = 0 if dfriction is None else dfriction

        Tcartesian = (torch.cat((s,sr),dim=1) + \
                     torch.cat((d,dr),dim=1) + \
                     torch.cat((self.i,self.ir),dim=1) + \
                     f).unsqueeze(-1)

        Tjoint = (torch.transpose(jacobian,1,2) @ Tcartesian).squeeze()


        if torch.any(Tjoint<=torque_min) or torch.any(Tjoint>=torque_max): # preclamping
            Tjoint = torch.clamp(Tjoint, min=0.1*torque_min, max=0.1*torque_max) 
        
        Tjoint_clamped = torch.clamp(Tjoint, min=0.8*torque_min, max=0.8*torque_max) 
        
        Tjoint_filtered = cutoff_coeff*self.Tjoint_previous + (1-cutoff_coeff)*Tjoint_clamped
        self.Tjoint_previous = Tjoint_filtered

        return Tjoint_filtered.unsqueeze(-1)

class joint_PID(input):
    """
    Joint PID Controller for sim2real applications on Franka Emika Panda. PID Control is defined as:

        T_nom = K_i*dt*deps + K_d*ddeps + K_p*deps -> deps: 7D joint position error
    
    where K_d and K_p are the derivative and proportional control gains which act upon the nominal and desired
    differences between the velocity and position error. Note that this implementation is inherently different from the
    internal isaacgym functionalities. The corresponding control is generated at each step of the simulation.

    Joint PID gains are obtained by trial and error. A more robust approach is to use some sort of
    identification/estimation procedure to autotune the PID. 

    Possible joint PID Tasks are:

    * multi-sinusoidal (MS)
    * chirp (CH)
    * impulse (IMP)
    * trapezoidal (TRAPZ)
    """
    def __init__(self,
                 num_envs, 
                 num_iter, 
                 num_joints,
                 num_coords, 
                 frequency, 
                 rigidbodyprops,
                 args,
                 poslim=[], vellim=[], acclim=[],
                 cposlim=[], cvellim=[], cacclim=[],
                 cornlim=[], wellim=[],alfalim=[]):
        super().__init__(num_envs, num_iter, num_joints, num_coords, frequency, rigidbodyprops, args,
                         poslim, vellim, acclim, cposlim, cvellim, cacclim, cornlim, wellim, alfalim)

        if self.args.random_controller_gains:
            kp_bounds = ((33.0,42.0),(50.0,57.0),(45.0,54.0),(45.0,54.0),(8.0,12.5),(2.9,3.3),(1.8,2.2),(2.7,3.2),(4.5,5.5))
            h_bounds = ((0.7,1.0),(0.7,1.0),(0.7,1.0),(0.7,1.0),(0.7,1.0),(0.7,1.0),(0.7,1.0),(0.7,1.0),(0.7,1.0))
            kv_bounds = ((3.5,4.5),(6.0,7.0),(3.5,4.5),(4.0,5.0),(1.5,1.9),(1.1,1.3),(0.87,1.03),(1.1,1.3),(1.5,1.9))
            ki_bounds = ((2.2,2.8),(4.0,5.0),(3.5,4.0),(3.0,3.5),(0.4,0.6),(0.23,0.28),(0.08,0.12),(0.17,0.23),(0.45,0.55))

            print('\n PID gain randomization:\nKp -->\nt0:'+str(kp_bounds[0])+'|t1:'+str(kp_bounds[1])+'|t2:'+str(kp_bounds[2])
            +'|t3:'+str(kp_bounds[3])+'|t4:'+str(kp_bounds[4])+'|t5:'+str(kp_bounds[5])+'|t6:'+str(kp_bounds[6])
            +'\nKd -->\nt0:'+str(kv_bounds[0])+'|t1:'+str(kv_bounds[1])+'|t2:'+str(kv_bounds[2])
            +'|t3:'+str(kv_bounds[3])+'|t4:'+str(kv_bounds[4])+'|t5:'+str(kv_bounds[5])+'|t6:'+str(kv_bounds[6])
            +'\nKi -->\nt0:'+str(ki_bounds[0])+'|t1:'+str(ki_bounds[1])+'|t2:'+str(ki_bounds[2])
            +'|t3:'+str(ki_bounds[3])+'|t4:'+str(ki_bounds[4])+'|t5:'+str(ki_bounds[5])+'|t6:'+str(ki_bounds[6]))

            self.kp = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in kp_bounds]).squeeze().T
            if self.args.damping_from_rigidprops:
                self.kv = self.dampingratio2kd(h_bounds,self.kp,self.inertias).T
            else:
                self.kv = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in kv_bounds]).squeeze().T
            self.ki = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in ki_bounds]).squeeze().T

        else:
            kp_nom = torch.tensor((37.0,50.0,45.0,45.0,10.0,3.0,2.0,3.0,5.0),device='cuda').unsqueeze(0) if self.args.type_of_robot=='franka' else torch.tensor((10.0,20.0,15.0,15.0,8.0,3.0,2.0,3.0,5.0),device='cuda').unsqueeze(0) 
            h_nom = torch.tensor(((0.5),(0.5),(0.5),(0.5),(0.5),(0.5),(0.5),(0.5),(0.5)),device='cuda').unsqueeze(0)
            kv_nom = torch.tensor(((4.0),(6.5),(4.0),(4.5),(1.7),(1.2),(1.0),(1.2),(1.7)),device='cuda').unsqueeze(0) if self.args.type_of_robot=='franka' else torch.tensor(((1.0),(2.5),(1.0),(1.5),(0.5),(0.3),(0.2),(0.3),(0.5)),device='cuda').unsqueeze(0)
            ki_nom = torch.tensor(((2.5),(3.1),(3.1),(2.5),(0.5),(0.2),(0.1),(0.2),(0.5)),device='cuda').unsqueeze(0) if self.args.type_of_robot=='franka' else torch.tensor(((2.5),(3.1),(3.1),(2.5),(0.5),(0.2),(0.1),(0.2),(0.5)),device='cuda').unsqueeze(0)

            print('\n PID gain randomization:\nKp -->\nt0:'+str(kp_nom)+'|t1:'+str(kp_nom)+'|t2:'+str(kp_nom)
            +'|t3:'+str(kp_nom)+'|t4:'+str(kp_nom)+'|t5:'+str(kp_nom)+'|t6:'+str(kp_nom)
            +'\nKd -->\nt0:'+str(kv_nom)+'|t1:'+str(kv_nom)+'|t2:'+str(kv_nom)
            +'|t3:'+str(kv_nom)+'|t4:'+str(kv_nom)+'|t5:'+str(kv_nom)+'|t6:'+str(kv_nom)
            +'\nKi -->\nt0:'+str(ki_nom)+'|t1:'+str(ki_nom)+'|t2:'+str(ki_nom)
            +'|t3:'+str(ki_nom)+'|t4:'+str(ki_nom)+'|t5:'+str(ki_nom)+'|t6:'+str(ki_nom))

            self.kp = torch.ones(self.args.num_envs,1).to(device=self.args.graphics_device_id) @ kp_nom
            if self.args.damping_from_rigidprops:
                self.kv = self.dampingratio2kd(h_nom,self.kp,self.inertias).T
            else:
                self.kv = torch.ones(self.args.num_envs,1).to(device=self.args.graphics_device_id) @ kv_nom
            self.ki = torch.ones(self.args.num_envs,1).to(device=self.args.graphics_device_id) @ ki_nom

        self.kp[:,7:] =  self.kp[:,7:].zero_()
        self.kv[:,7:] =  self.kv[:,7:].zero_()
        self.ki[:,7:] =  self.ki[:,7:].zero_()

        if self.args.type_of_robot=='kuka':
            self.kp =  self.kp[:,:7]
            self.kv =  self.kv[:,:7]
            self.ki =  self.ki[:,:7]

        self.epos_prev = 0 # derivative error
        self.i = 0 # integral accumulator
        
        self.Tjoint_previous = torch.tensor(0)

        self.kp_aug = self.kp # 9D
        self.kv_aug = self.kv # 9D
        self.ki_aug = self.ki # 9D

        if self.args.type_of_trajectory=='MS':
            self.vars = self.sin()
        elif self.args.type_of_trajectory=='CH':
            self.vars = self.chirp()
        elif self.args.type_of_trajectory=='IMP':
            self.vars = self.impulse()
        elif self.args.type_of_trajectory=='TRAPZ':
            self.vars = self.trapz()

    def sin(self, reduce=None):
        return super().sin(reduce)
    
    def chirp(self, reduce=None):
        return super().chirp(reduce)
    
    def impulse(self, reduce=None):
        return super().impulse(reduce)
    
    def trapz(self, reduce=None):
        return super().trapz(reduce)
    
    def dampingratio2kd(self, h_bounds, kps, Is):
        zetas = torch.stack([torch.FloatTensor(1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in h_bounds]).squeeze().T
        return torch.stack([zetas*2*torch.sqrt(I[1:10][:,(0 if i==(1,3,5) else 2)]*kp) for i,(I,kp) in enumerate(zip(Is,kps))]).T

    def obtainjointvars(self, itr):
        """
        Obtains tracking trajectory which defines the desired variables of the controller directly
        from the joint variables - for joint_PID controller.

        Parameters
        ---
            itr (int) : current iteration

        Returns
        ---
            action
        """
        vars = self.vars[:,:,itr]
        self.desired_trajectory = vars
    
        return self.desired_trajectory

    def getcontrol(self):
        return super().getcontrol()
    
    def setcontrol(self, control_action_):
        return super().setcontrol(control_action_)
    
    def autotune(self, endgain, oscperiod):
        """
        Implements Ziegler-Nichols autotuner to set the nominal values of control gains.
        """
        Kp = 0.6*endgain
        Ti = 0.5*oscperiod
        Td = 0.125*oscperiod
        Ki = 1.2*endgain/Ti
        Kd = 0.075*endgain*Td

        return {
            'kp' : Kp,
            'kd' : Kd,
            'ki' : Ki
        }

    def step_PD(self, cpos, torque_limits,
                dpos=None, dfriction=None,
                cutoff_coeff=0.0):
        """
        Step PD control to be called in each iteration controlled with PD.

        Parameters
        ---
            pos_current (torch.Tensor) : current position
            torque_limits (torch.Tensor/numpy.array) : joint torque limits
            dpos (torch.Tensor) : desired position of the current iteration (if not obtain_trajectory)
            dfriction (torch.Tensor) : desired friction per projected joint torques
            cutoff_coeff (float) : derivative controller low pass filter rate

        Returns
        ---
            action (torch.Tensor) : control action for the current iteration, step must be called every iter
        """
        torque_min = -1*torch.tensor(torque_limits,device=self.args.graphics_device_id).expand(self.num_envs,self.num_joints)
        torque_max = 1*torch.tensor(torque_limits,device=self.args.graphics_device_id).expand(self.num_envs,self.num_joints)

        dpos = self.desired_trajectory if dpos is None else dpos

        epos = dpos - cpos.squeeze()
        derrpos = epos - self.epos_prev
        self.epos_prev = epos
        
        s = self.kp*epos
        d = self.kv*derrpos/self.dt

        f = 0 if dfriction is None else dfriction

        Tjoint = s + d + f

        if torch.any(Tjoint<=torque_min) or torch.any(Tjoint>=torque_max): # preclamping
            Tjoint = torch.clamp(Tjoint, min=0.1*torque_min, max=0.1*torque_max) 

        Tjoint_clamped = torch.clamp(Tjoint, min=torque_min, max=torque_max) 

        Tjoint_filtered = cutoff_coeff*self.Tjoint_previous + (1-cutoff_coeff)*Tjoint_clamped
        self.Tjoint_previous = Tjoint_filtered

        Tjoint_filtered[:,7:] = Tjoint_filtered[:,7:].zero_()

        return Tjoint_filtered.unsqueeze(-1)
    
    def step_PID(self, cpos, torque_limits,
                dpos=None, dfriction=None,
                windup_limit=1000, cutoff_coeff=0.0):
        """
        Step PID control to be called in each iteration controlled with PID.

        Parameters
        ---
            pos_current (torch.Tensor) : current position
            torque_limits (torch.Tensor/numpy.array) : joint torque limits
            dpos (torch.Tensor) : desired position of the current iteration (if not obtain_trajectory)
            dfriction (torch.Tensor) : desired friction per projected joint torques
            windup_limit (int) : integral windup limit to prevent overflow
            cutoff_coeff (float) : derivative controller low pass filter rate

        Returns
        ---
            action (torch.Tensor) : control action for the current iteration, step must be called every iter
        """
        torque_min = -1*torch.tensor(torque_limits,device=self.args.graphics_device_id).expand(self.num_envs,self.num_joints)
        torque_max = 1*torch.tensor(torque_limits,device=self.args.graphics_device_id).expand(self.num_envs,self.num_joints)

        dpos = self.desired_trajectory if dpos is None else dpos

        epos = dpos - cpos.squeeze()
        derrpos = epos - self.epos_prev
        self.epos_prev = epos
        s = self.kp*epos
        d = self.kv*derrpos/self.dt
        self.i += self.ki*epos*self.dt
        if torch.any(self.i >=windup_limit):
            self.i = windup_limit*torch.ones_like(self.i)
        if torch.any(self.i<= -windup_limit):
            self.i = -windup_limit*torch.ones_like(self.i)

        f = 0 if dfriction is None else dfriction

        Tjoint = s + d + self.i + f 

        if torch.any(Tjoint<=torque_min) or torch.any(Tjoint>=torque_max): # preclamping
            Tjoint = torch.clamp(Tjoint, min=0.1*torque_min, max=0.1*torque_max) 

        Tjoint_clamped = torch.clamp(Tjoint, min=0.8*torque_min, max=0.8*torque_max) 

        Tjoint_filtered = cutoff_coeff*self.Tjoint_previous + (1-cutoff_coeff)*Tjoint_clamped
        self.Tjoint_previous = Tjoint_filtered

        Tjoint_filtered[:,7:] = Tjoint_filtered[:,7:].zero_()

        return Tjoint_filtered.unsqueeze(-1)

class cic(input):
    """
    Cartesian Impedance Control for sim2real applications on Franka Emika Panda. CIC Control is defined as:

        T_nom = K_a*dw + K_d*dv + K_p*dx

    where the specifications of control are similar to PID controller.

    Possible cartesian CIC Tasks are:

    * vertical spiral (VS)
    * fixed spiral (FS)
    * fixed circular (FC)
    * multi-sinusoidal (MS)
    * chirp (CH)
    * impulse (IMP)
    * trapezoidal (TRAPZ)
    * object pick (PICK)
    * object push (PUSH)
    """
    def __init__(self,
                 num_envs, 
                 num_iter, 
                 num_joints,
                 num_coords, 
                 frequency, 
                 rigidbodyprops,
                 args,
                 poslim=[], vellim=[], acclim=[],
                 cposlim=[], cvellim=[], cacclim=[],
                 cornlim=[], wellim=[],alfalim=[]):
        super().__init__(num_envs, num_iter, num_joints, num_coords, frequency, rigidbodyprops, args,
                         poslim, vellim, acclim, cposlim, cvellim, cacclim, cornlim, wellim, alfalim)

        if self.args.random_controller_gains:
            kp_bounds = ((1,5),(1,5),(1,5)) # x,y,z
            kv_bounds = ((0.5,1.0),(0.5,1.0),(0.5,1.0)) # x,y,z
            ka_bounds = ((0.1,0.5),(0.1,0.5),(0.1,0.5)) # x,y,z

            kpr_bounds = ((1,5),(1,5),(1,5)) # X, Y, Z
            kvr_bounds = ((0.5,1.0),(0.5,1.0),(0.5,1.0)) # X, Y, Z
            kar_bounds = ((0.1,0.5),(0.1,0.5),(0.1,0.5)) # X, Y, Z

            print('\n PID gain randomization:\nKp -->\nx:'+str(kp_bounds[0])+'|y:'+str(kp_bounds[1])+'|z:'+str(kp_bounds[2])
            +'\nKd -->\nx:'+str(kv_bounds[0])+'|y:'+str(kv_bounds[1])+'|z:'+str(kv_bounds[2])
            +'\nKa -->\nx:'+str(ka_bounds[0])+'|y:'+str(ka_bounds[1])+'|z:'+str(ka_bounds[2])
            +'\nKpr -->\ne11:'+str(kpr_bounds[0])+'|e22:'+str(kpr_bounds[1])+'|e33:'+str(kpr_bounds[2])
            +'\nKvr -->\ne11:'+str(kvr_bounds[0])+'|e22:'+str(kvr_bounds[1])+'|e33:'+str(kvr_bounds[2])
            +'\nKar -->\ne11:'+str(kar_bounds[0])+'|e22:'+str(kar_bounds[1])+'|e33:'+str(kar_bounds[2]))

            self.kp = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in kp_bounds]).squeeze()
            self.kv = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in kv_bounds]).squeeze()
            self.ka = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in ka_bounds]).squeeze()

            self.kpr = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in kpr_bounds]).squeeze()
            self.kvr = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in kvr_bounds]).squeeze()
            self.kar = torch.stack([torch.FloatTensor(self.args.num_envs,1).uniform_(bounds[0],bounds[1]).to(device=self.args.graphics_device_id) for bounds in kar_bounds]).squeeze()

        else:
            kp_nom = 3 
            kv_nom = (1+2*(5**0.5))/2
            ka_nom = 0.5 

            kpr_nom = 3 
            kvr_nom = (1+2*(5**0.5))/2
            kar_nom = 0.5 

            print('\n PID gain randomization:\nKp -->\nx:'+str(kp_bounds[0])+'|y:'+str(kp_bounds[1])+'|z:'+str(kp_bounds[2])
            +'\nKd -->\nx:'+str(kv_bounds[0])+'|y:'+str(kv_bounds[1])+'|z:'+str(kv_bounds[2])
            +'\nKa -->\nx:'+str(ka_bounds[0])+'|y:'+str(ka_bounds[1])+'|z:'+str(ka_bounds[2])
            +'\nKpr -->\ne11:'+str(kpr_bounds[0])+'|e22:'+str(kpr_bounds[1])+'|e33:'+str(kpr_bounds[2])
            +'\nKvr -->\ne11:'+str(kvr_bounds[0])+'|e22:'+str(kvr_bounds[1])+'|e33:'+str(kvr_bounds[2])
            +'\nKar -->\ne11:'+str(kar_bounds[0])+'|e22:'+str(kar_bounds[1])+'|e33:'+str(kar_bounds[2]))

            self.kp = kp_nom*torch.ones(self.args.num_envs,3).to(device=self.args.graphics_device_id)
            self.kv = kv_nom*torch.ones(self.args.num_envs,3).to(device=self.args.graphics_device_id)
            self.ka = ka_nom*torch.ones(self.args.num_envs,3).to(device=self.args.graphics_device_id)

            self.kpr = kpr_nom*torch.ones(self.args.num_envs,4).to(device=self.args.graphics_device_id)
            self.kvr = kvr_nom*torch.ones(self.args.num_envs,4).to(device=self.args.graphics_device_id)
            self.kar = kar_nom*torch.ones(self.args.num_envs,4).to(device=self.args.graphics_device_id)
        
        self.kp_aug = torch.cat((self.kp,self.kpr),dim=1) # 6D
        self.kv_aug = torch.cat((self.kv,self.kvr),dim=1) # 6D
        self.ki_aug = torch.cat((self.ka,self.kar),dim=1) # 6D

    def sin_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().sin_cartesian(posd, ornd, posi, orni, itr, reduce)
    
    def chirp_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().chirp_cartesian(posd, ornd, posi, orni, itr, reduce)
    
    def trapz_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().trapz_cartesian(posd, ornd, posi, orni, itr, reduce)
    
    def impulse_cartesian(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().impulse_cartesian(posd, ornd, posi, orni, itr, reduce)
    
    def pushobject(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().pushobject(posd, ornd, posi, orni, itr, reduce)
    
    def pickobject(self, posd, ornd, posi, orni, itr, reduce=None):
        return super().pickobject(posd, ornd, posi, orni, itr, reduce)

    def obtaintrajectory(self, pos=None, orn=None, ipos=None, iorn=None, itr=0):
        """
        Obtains tracking trajectory which defines the desired variables of the controller.

        Parameters
        ---
            pos (torch.Tensor) : current position
            orn (torch.Tensor) : current velocity
            ipos (torch.Tensor) : initial position
            iorn (torch.Tensor) : initial orientation
            itr (int) : current iteration

        Returns
        ---
            action
        """
        orn = decide_orientation(orn, self.args.orientation_dimension)

        if self.args.type_of_trajectory=='VS':
            trajectory = self.vertical_spiral(posd=pos,posi=ipos,itr=itr)
            trajectory = torch.cat((trajectory, orn),dim=1)
        elif self.args.type_of_trajectory=='FS':
            trajectory = self.fixed_spiral(posd=pos,posi=ipos,itr=itr)
            trajectory = torch.cat((trajectory, orn),dim=1)
        elif self.args.type_of_trajectory=='FC':
            trajectory = self.fixed_circular(posd=pos,posi=ipos,itr=itr)
            trajectory = torch.cat((trajectory, orn),dim=1)
        elif self.args.type_of_trajectory=='MS':
            trajectory = self.sin_cartesian(posd=pos,ornd=orn,posi=ipos,orni=iorn,itr=itr)
        elif self.args.type_of_trajectory=='CH':
            trajectory = self.chirp_cartesian(posd=pos,ornd=orn,posi=ipos,orni=iorn,itr=itr)
        elif self.args.type_of_trajectory=='IMP':
            trajectory = self.impulse_cartesian(posd=pos,ornd=orn,posi=ipos,orni=iorn,itr=itr)
        elif self.args.type_of_trajectory=='TRAPZ':
            trajectory = self.trapz_cartesian(posd=pos,ornd=orn,posi=ipos,orni=iorn,itr=itr)
        
        self.desired_trajectory = trajectory

        return self.desired_trajectory

    def getcontrol(self):
        return super().getcontrol()
    
    def setcontrol(self, control_action_):
        return super().setcontrol(control_action_)

    def singularity_check(self, jacobian):
        if torch.linalg.det(torch.dot(jacobian*torch.transpose(jacobian))) <= 1e-8:
            U, S, V  = torch.linalg.svd(jacobian)
            for sval in S:
                if sval <= 1e-4:
                    sval = 0.0
                else:
                    sval = 1.0/float(sval)
            jacobian = torch.dot(V, torch.dot(torch.diag(S), U.T)) 

        return jacobian
    
    def step_cic(self, cpos, corn, cvel, cwel, cacc, calfa, jacobian, torque_limits,
                 dpos=None, dorn=None, dvel=None, dwel=None, dacc=None, dalfa=None, dfriction=None):
        """
        Step CIC control to be called in each iteration controlled with PID.

        Parameters
        ---
            pos_current (torch.Tensor) : current position
            orn_current (torch.Tensor) : current orientation
            vel_current (torch.Tensor) : current velocity
            wel_current (torch.Tensor) : current angular velocity
            acc_current (torch.Tensor) : current acceleration
            alfa_current (torch.Tensor) : current angular acceleration
            jacobian (torch.Tensor) : end-effector jacobian
            torque_limits (torch.Tensor/numpy.array) : joint torque limits
            dpos (torch.Tensor) : desired position of the current iteration (if not obtain_trajectory)
            dorn (torch.Tensor) : desired orientation of the current iteration (if not obtain_trajectory)
            dfriction (torch.Tensor) : desired friction per projected joint torques
            windup_limit (int) : integral windup limit to prevent overflow
            cutoff_coeff (float) : derivative controller low pass filter rate

        Returns
        ---
            action (torch.Tensor) : control action for the current iteration, step must be called every iter
        """
        torque_min = -1*torch.tensor(torque_limits,device=self.args.graphics_device_id).expand(self.num_envs,self.num_joints)
        torque_max = 1*torch.tensor(torque_limits,device=self.args.graphics_device_id).expand(self.num_envs,self.num_joints)

        dpos = self.desired_trajectory[:,:3] if dpos is None else dpos
        dorn = self.desired_trajectory[:,3:] if dorn is None else dorn

        epos = dpos - cpos
        self.epos_prev = epos
        eorn = self.orientation_error(dorn, corn)
        self.eorn_prev = eorn

        spos = self.s * epos
        self.i += self.ki*epos*self.dt
        if torch.any(self.i >=windup_limit):
            self.i = windup_limit*torch.ones_like(self.i)
        if torch.any(self.i<= -windup_limit):
            self.i = -windup_limit*torch.ones_like(self.i)

        sorn = self.sr * eorn
        self.ir = self.kir*eorn*self.dt
        if torch.any(self.ir >=windup_limit):
            self.ir = windup_limit*torch.ones_like(self.ir)
        if torch.any(self.ir<= -windup_limit):
            self.ir = -windup_limit*torch.ones_like(self.ir)

        f = 0 if dfriction is None else dfriction

        sterm = torch.cat((spos,sorn),dim=1)
        iterm = torch.cat((self.i,self.ir),dim=1)
        dofvel = dofvel.reshape(self.num_envs,self.num_joints,1)

        Tjoint = (torch.transpose(jacobian,1,2) @ sterm.unsqueeze(-1) - \
                  torch.transpose(jacobian,1,2) @ (self.d.unsqueeze(-1).unsqueeze(-1) * (jacobian @ (dofvel))) + \
                  torch.transpose(jacobian,1,2) @ iterm.unsqueeze(-1) + \
                  f).squeeze()

        if torch.any(Tjoint<=torque_min) or torch.any(Tjoint>=torque_max): # preclamping
            Tjoint = torch.clamp(Tjoint, min=0.1*torque_min, max=0.1*torque_max) 
        
        Tjoint_clamped = torch.clamp(Tjoint, min=0.8*torque_min, max=0.8*torque_max) 
        
        Tjoint_filtered = cutoff_coeff*self.Tjoint_previous + (1-cutoff_coeff)*Tjoint_clamped
        self.Tjoint_previous = Tjoint_filtered

        return Tjoint_filtered.unsqueeze(-1)

class compensate():
    """
    Compensation object to be used in compesnsating for external non-conservative forces, current
    compensation options are: gravity, friction.
    * gravity_compensation: handled through https://github.com/NVlabs/oscar/blob/main/oscar/agents/franka.py
    * friction_compensation: handled through gym.dof_properties or input level manipulation

    Parameters
    ---
        args (dict) : arguments passed in generation
        gravity (float) : master environment gravity
        friction_params (np.array) : friction parameters for velocity dependent damping
        num_joints (int) : number of joints
    """
    def __init__(self,
                 args,
                 gravity,
                 friction_params,
                 num_joints):
        self.args = args
        self.num_joints = num_joints
        self.g = torch.empty(self.args.num_envs, self.num_joints+(1 if self.args.type_of_robot=='franka' else 2), 6, 1, 
                        dtype=torch.float, device=self.args.graphics_device_id)
        self.g[:,:,2,:] = gravity
        self.num_joints = num_joints
        self.args = args
        self.fp = torch.tensor(friction_params).to(device=args.graphics_device_id)
        self.fval = torch.empty(self.args.num_envs, self.num_joints)
        if not self.args.disable_gravity:
            print("Compensating for gravitational losses - implementation on control action")
        if not self.args.disable_friction:
            print("Compensating for frictional losses - implementation on control action")
    
    def getargs(self):
        return self.args

    def setargs(self,args_):
        self.args = args_
    
    def __str__(self):
        return f'Compensator Object instantiated'

    def gravity(self,jacobian,rigidbodyprops):
        """
        Estimated gravitational losses on each joint as a function of jacobian matrix, losses are 
        included in the input trajectory during each step of the simulation.

        Parameters
        ---
            jacobian (torch.Tensor) : Jacobian matrix of each asset 
            rigidbodyprops (torch.Tensor) : rigid props of each asses - extracted mass
        
        Returns
        ---
            gtorque (torch.Tensor) : compensation amount to accound for gravitation effects
        """
        mass = torch.tensor([[link.mass for link in env] for env in rigidbodyprops],device=self.args.graphics_device_id)
        gforce = mass.squeeze(0)[:,1:].unsqueeze(-1).unsqueeze(-1) * self.g
        jlink = jacobian[:, :self.num_joints+(1 if self.args.type_of_robot=='franka' else 2), :, :self.num_joints]
        gtorque_ = (torch.transpose(jlink, 2, 3) @ gforce).squeeze(-1)
        gtorque = torch.sum(gtorque_, dim=1, keepdim=False).unsqueeze(-1)
        return gtorque
    
    def compensate_gravity(self,traj,jacobian,mass):
        updtraj = traj + self.gravity(self,jacobian,mass)
        return updtraj

    def friction(self,vel):
        """
        Estimated frictional losses on each joint as a function of joint velocity, losses are directly
        implemented through isaacgym's internal compensation method. - coulomb friction NOT IDEAL

        Parameters
        ---
            vel (torch.Tensor) : current velocity
        
        Returns
        ---
            ftorque (torch.Tensor) : compensation amount to accound for frictional effects
        """
        vel = vel.contiguous().view(self.args.num_envs,self.num_joints,1).squeeze()
        self.fval = self.fp[:,0]/(1+torch.exp(-self.fp[:,1]*(vel+self.fp[:,2]))) - self.fp[:,0]/(1+torch.exp(-self.fp[:,1]*self.fp[:,2]))
        return self.fval.unsqueeze(-1).float()
    
    def compensate_friction_on_torque(self,traj,vel):
        updtraj = traj + self.friction_on_torque(self,vel)        
        return updtraj


