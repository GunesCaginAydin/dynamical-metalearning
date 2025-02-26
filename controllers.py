import torch
from matplotlib import pyplot as plt
import numpy as np
import math
from isaacgym import gymutil
from isaacgym.torch_utils import quat_conjugate,quat_mul

class input():
    """
    Blueprint class for all control objects, inherited throughout

            POSSIBLE TO UNITE ACTION AND OSC --- WORK IN PROGRESS
    """
    def __init__(self,
                 num_envs, 
                 num_iter, 
                 num_joints,
                 num_coords, 
                 frequency, 
                 input_type,
                 mass_vector,
                 args):
        self.args = args
        self.num_envs = num_envs
        self.num_iter = num_iter
        self.num_joints = num_joints
        self.num_coords = num_coords
        self.frequency = frequency
        self.input_type = input_type
        self.mass_vector = mass_vector
        
        #self.diff_action = torch.empty(self.num_envs,self.num_joints,self.num_iter)
        print("Control Action is acquired")

        time_step = torch.linspace(0, self.num_iter, self.num_iter)
        self.t = time_step.unsqueeze(1) * 1/60

        if self.args.osc_task:
            self.control_action = torch.empty(0)
            self.buffer_control_action = torch.empty((self.num_envs, self.num_joints,1), dtype=torch.float32).to(device=self.args.graphics_device_id)
        elif self.args.control_imposed:
            self.control_action = torch.empty((self.num_envs,self.num_joints,self.num_iter))
            self.buffer_control_action = torch.empty((1,self.num_envs,self.num_joints), dtype=torch.float32).to(device=self.args.graphics_device_id)

        if self.args.measure_force:   
            self.measured_torque = torch.empty((1,self.num_envs,self.num_joints), dtype=torch.float32).to(device=self.args.graphics_device_id)
        else:
            self.measured_torque = torch.empty(0)

        self.buffer_position = torch.empty((0,self.num_envs,self.num_coords), dtype=torch.float32).to(device=self.args.graphics_device_id)
        self.buffer_target  = torch.empty((0,self.num_envs,3), dtype=torch.float32)
        self.buffer_velocities = torch.empty((0,self.num_envs,self.num_joints), dtype=torch.float32).to(device=self.args.graphics_device_id) 

        if self.args.measure_gravity_friction:
            self.buffer_friction = torch.empty((self.num_envs,self.num_joints,0), dtype=torch.float32).to(device=self.args.graphics_device_id) 
            self.buffer_gravity = torch.empty((self.num_envs,self.num_joints,0), dtype=torch.float32).to(device=self.args.graphics_device_id)
        else:
            self.buffer_friction = torch.empty(0)
            self.buffer_gravity = torch.empty(0)

    def getdata(self):
        datadict = {"envs":self.num_envs,
                    "iters":self.num_iter, 
                    "joints":self.num_joints,
                    "type":self.input_type,
                    "freq":self.frequency}
        return datadict

    def setdata(self,num_envs_, num_iter_, num_joints_, frequency_, input_type_):
        self.num_envs = num_envs_
        self.num_iter = num_iter_
        self.num_joints = num_joints_
        self.frequency = frequency_
        self.input_type = input_type_

    def getcontrol(self):
        return {
            "ac" : self.control_action.contiguous(),
            #"acd" : self.control_diff.contiguous(),
            "bca" : self.buffer_control_action,
            #"bcad" : self.buffer_control_action[:,:,:self.num_joints],
            "bp" : self.buffer_position,
            "bv" : self.buffer_velocities,
            "bt" : self.buffer_target,
            "bf" : self.buffer_friction,
            "bg" : self.buffer_gravity,
            "mt" : self.measured_torque
        }

    def setcontrol(self,control_action_):
        self.control_action = control_action_
        #self.control_diff = control_diff_

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
    
class action(input):
    """
    Action generator object for the generation and visualization of various randomized 
    input trajectories. Inherits from input class determining the regimes of application

    Available trajectories are: 
    sinusoidal: multi-sinusoidal trajectory randomized magnitudes, directions and freqs
    chirp: chirp trajectory, randomized freqs
    impulse: impulse trajectory with multiple rest and rise regimes
    trapezoidal: trapezoidal velocity profile trajectory with multiple rest and rise regimes

    Initialization parameters are:
    num_envs: number of active environments in the simulation
    num_iter: maximum number of iterations of the simulation
    num_joints: number of joints of the generic actor in the simulation
    frequency: master frequency of the trajectory - randomized inside the simulation
    input_type: input type of the generation
    """

    def __init__(self,
                 num_envs, 
                 num_iter, 
                 num_joints,
                 num_coords, 
                 frequency, 
                 input_type,
                 mass_vector,
                 args):
        super().__init__(num_envs, num_iter, num_joints, num_coords, frequency, input_type, mass_vector, args)

        if self.args.type_of_input == 'MS':
            self.control_action = self.sin()
        elif self.args.type_of_input == 'CH':
            self.control_action = self.chirp()
        elif self.args.type_of_input == "IMP":
            self.control_action = self.impulse()
        elif self.args.type_of_input == "TRAPZ":
            self.control_action = self.trapz()
    
    def plot(self, trajectory, num_envs, num_dofs):
        return super().plot_trajectory(trajectory, num_envs, num_dofs)
    
    def getcontrol(self):
        return super().getcontrol()
    
    def setcontrol(self, control_action_):
        return super().setcontrol(control_action_)
        
    def sin(self):
        """
        Sinusoidal randomized trajectory
        """
        def sin_signal(self,attenuation_factor):
            r1,r2 = -1,1
            a = 2* torch.rand(4).uniform_(-self.frequency*15,self.frequency*15)
            t = self.t
            freq = 2 * np.pi * torch.rand(1).uniform_(self.frequency/1.5, self.frequency*1.5)
            _trajectory =  torch.sign(torch.rand(1).uniform_(r1,r2)) * (a[0]*torch.sin(freq*t) 
                                + a[1] * torch.cos(freq*1.5*t) + a[2] *torch.sin(freq*2*t) 
                                +torch.sign(torch.rand(1).uniform_(r1,r2))* a[3] * torch.cos(freq*3*t))/attenuation_factor          
            trajectory = _trajectory.view(self.num_iter) 
            return trajectory

        for i in range(self.num_envs):
            for j in range(self.num_joints):
                if j ==8 or j==7:
                    attenuation_factor = np.inf
                elif j == 1:
                    attenuation_factor = 1.3 # 1.3 # 2 for real
                else:
                    attenuation_factor = 1.6 # 1.6 # 3 for real               
                single_dof = sin_signal(self,attenuation_factor)
                self.control_action[i][j] = single_dof
                #self.diff_action[i][j][1:] = torch.diff(self.control_action[i][j])      
        return self.control_action.to(device=self.args.graphics_device_id)
        
    def chirp(self):
        """
        Chirp-like randomized trajectory
        """
        def chirp_signal(self,attenuation_factor,j):
            t = self.t
            phi = torch.rand(1).uniform_(-np.pi,np.pi)
            q0 = torch.rand(1).uniform_(-.5, .5) 
            a = torch.rand(1).uniform_(-4,4)    #  [ -3,3]  

            if self.frequency < 0.3:
                f1 = torch.rand(1).uniform_(self.frequency/1.1,self.frequency*1.5)
                f2 = torch.rand(1).uniform_(self.frequency/1.5, self.frequency*2)
            else:
                f1 = torch.rand(1).uniform_(self.frequency/1.3,self.frequency/1.2)
                f2 = torch.rand(1).uniform_(self.frequency/1.1,self.frequency*1.1)
            
            if j>=7:
                _trajectory = 0*t
            else:
                _trajectory = q0 + torch.sign(torch.rand(1).uniform_(-1,1))* a * torch.cos (2* np.pi * f1 *( 1 + 1/4 * torch.cos(  2 * np.pi * f2* t))*t + phi)
            trajectory = _trajectory.view(self.num_iter)/attenuation_factor 
            return trajectory
        
        for i in range(self.num_envs):
            for j in range(self.num_joints):
                if j ==8 or j==7:
                    attenuation_factor=np.inf
                else: 
                    attenuation_factor = 2 # 1.6 safe
                single_dof = chirp_signal(self,attenuation_factor,j)
                self.control_action[i][j] = single_dof
                #self.diff_action[i][j][1:] = torch.diff(self.control_action[i][j])
        return self.control_action.to(device=self.args.graphics_device_id)
    
    def impulse(self):
        """
        Impulse-like randomized trajectory
        """
        def impulse_signal(self,attenuation_factor,fillval):
            r1,r2,reach = -1,1,1.5
            t = self.time_steps()
            spikes = torch.zeros(fillval)
            null = []
            spikes_extend = 0
            for i in range(len(spikes)):
                spikes[i] = torch.Tensor.random_(0,t[-1])
                spikes
            spike_intervals = spikes_extend(spikes)
            if t in null:
                _trajectory = 0
            else:
                _trajectory = torch.sign(torch.rand(1).uniform_(r1,r2))*reach/attenuation_factor

            trajectory = torch.Tensor(_trajectory)/attenuation_factor
            trajectory = trajectory.view(self.num_iter) 
            return trajectory

        for i in range(self.num_envs):
            for j in range(self.num_joints):
                if j ==8 or j==7:
                    attenuation_factor=np.inf
                else: 
                    attenuation_factor = 2 # 1.6 safe
                single_dof = impulse_signal(self,attenuation_factor,j)
                self.control_action[i][j] = single_dof
                #self.diff_action[i][j][1:] = torch.diff(self.control_action[i][j])     
        return self.control_action.to(device=self.args.graphics_device_id)
    
    def trapz(self):
            """
            Trapezoidal randomized trajectory
            """
            def trapz_signal(self,attenuation_factor,len,ang1,ang2):
                t = self.time_steps()
                _len = 0
                _ang1 = 0
                _ang2 = 0
                trapz = []
                rise = []
                still = []
                drop = []
                for trap in trapz:
                    if t in rise:
                        _trajectory = 0
                    elif t in still:
                        _trajectory = 0
                    elif t in drop:
                        _trajectory = 0

                trajectory = torch.Tensor(_trajectory)/attenuation_factor
                trajectory = trajectory.view(self.num_iter) 
                return trajectory

            for i in range(self.num_envs):
                for j in range(self.num_joints):
                    if j ==8 or j==7:
                        attenuation_factor=np.inf
                    else: 
                        attenuation_factor = 2 # 1.6 safe
                    single_dof = trapz_signal(self,attenuation_factor,j)
                    self.control_action[i][j] = single_dof
                    #self.diff_action[i][j][1:] = torch.diff(self.control_action[i][j])     
            return self.control_action.to(device=self.args.graphics_device_id)
    
class osc(input):
    """
    Operational Space Control control input generator, generated action is determined by the arguments,
    inherits from input class determining the regime of application
    """
    def __init__(self,
                 num_envs, 
                 num_iter, 
                 num_joints,
                 num_coords, 
                 frequency, 
                 input_type,
                 mass_vector,
                 args):
        super().__init__(num_envs, num_iter, num_joints, num_coords, frequency, input_type, mass_vector, args)

        self._radius = torch.rand((1,self.args.num_envs)).uniform_(0.01,0.12).to(device=self.args.graphics_device_id)
        self._period = torch.rand(1).uniform_(20,100).to(device=self.args.graphics_device_id)
        self._z_speed = torch.rand(1).uniform_(0.1,0.4).to(device=self.args.graphics_device_id)
        self._sign = torch.sign(torch.rand(1).uniform_(-1,1)).to(device=self.args.graphics_device_id)
        #_offset =  torch.sign(torch.rand(1).uniform_(-0.3,0.3)).to(device=self.args.graphics_device_id)

        if self.args.random_osc_gains:
            kp_lower_bound, kp_higher_bound = (1,5)
            kv_lower_bound, kv_higher_bound = (1,5**0.5)
            print('\n OSC K randomization:\nKp -->\n'+str(kp_lower_bound)+'|'+str(kp_higher_bound) 
            +'\nKv -->\n' +str(kv_lower_bound)+'|'+str(kv_higher_bound))
            self.kp = torch.FloatTensor(self.args.num_envs,1).uniform_(kp_lower_bound,kp_higher_bound).to(device=self.args.graphics_device_id)
            self.kv = torch.FloatTensor(self.args.num_envs,1).uniform_(kv_lower_bound,kv_higher_bound).to(device=self.args.graphics_device_id)
        else:
            self.kp = torch.FloatTensor(10).repeat(self.args.num_envs,1).to(device=self.args.graphics_device_id)
            self.kv = torch.FloatTensor(2*10**0.5).repeat(self.args.num_envs,1).to(device=self.args.graphics_device_id)
            print('\n OSC K default:\nKp -->\n'+str(self.kp) 
            +'\nKv -->\n' +str(self.kv))

    def getcontrol(self):
        return super().getcontrol()
    
    def setcontrol(self, control_action_):
        return super().setcontrol(control_action_)
     
    def vertical_spiral(self,posd,posi,itr):
        posd[:, 0] = posi[:, 0] + math.sin(itr / self._period) * self._radius 
        posd[:, 1] = posi[:, 1] + math.cos(itr / self._period) * self._radius
        posd[:, 2] = posi[:, 2] - 0.1 + self._sign * self._z_speed * itr/self.num_iter
        return posd
    
    def fixed_spiral(self,posd,posi,itr):  
        self._radius = 0.1    
        posd[:, 0] = posi[:, 0] + math.sin(itr / 80) * self._radius 
        posd[:, 1] = posi[:, 1] + math.cos(itr / 80) * self._radius
        posd[:, 2] = posi[:, 2] + - 0.1 + 0.2 * itr/self.num_iter           
        return posd
    
    def fixed_circular(self,posd,posi,itr):
        self._radius = 0.1
        posd[:, 0] = posi[:, 0] 
        posd[:, 1] = posi[:, 1] + math.sin(itr / 50) * self._radius 
        posd[:, 2] = posi[:, 2] + math.cos(itr / 50) * self._radius
        return posd
    
    def orientation_error(self,desired,current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def step_osc(self, pos_desired, orn_desired, pos_current, vel_current, orn_current, pos_initial, eff_jacobian, franka_mass, itr):
        vel_current = vel_current.view(self.args.num_envs, 9, 1)
        if self.args.type_of_osc == 'VS':
            pos_desired = self.vertical_spiral(pos_desired,pos_initial,itr)
        elif self.args.type_of_osc == 'FS':
            pos_desired = self.fixed_spiral(pos_desired,pos_initial,itr)
        elif self.args.type_of_osc == "FC":
            pos_desired = self.fixed_circular(pos_desired,pos_initial,itr)
        m_inv = torch.inverse(franka_mass)
        m_eef = torch.inverse(eff_jacobian @ m_inv @ torch.transpose(eff_jacobian, 1, 2)) 
        orn_current /= torch.norm(orn_current, dim=-1).unsqueeze(-1)
        pos_err = self.kp * (pos_desired - pos_current)
        dpose = torch.cat([pos_err, self.orientation_error(orn_desired,orn_current)], -1)
        self.control_action = self.kp.unsqueeze(-1) * (torch.transpose(eff_jacobian, 1, 2) @ m_eef @ (dpose.unsqueeze(-1))) - self.kv.unsqueeze(-1) * franka_mass @  vel_current
        #self.control_diff = self.control_action
        return self.control_action.to(self.args.graphics_device_id)

class compensate():
    """
    Compensation object to be used in compesnsating for external non-conservative forces, current
    compensation options are: gravity, friction.
    gravity_compensation: handled through https://github.com/NVlabs/oscar/blob/main/oscar/agents/franka.py
    friction_compensation: handled through gym.dof_properties or input level manipulation
    """
    def __init__(self,
                 args,
                 gravity,
                 friction_params,
                 num_joints):
        self.args = args
        self.num_joints = num_joints
        self.g = torch.empty(self.args.num_envs, self.num_joints+1, 6, 1, 
                        dtype=torch.float, device=self.args.graphics_device_id)
        self.g[:,:,2,:] = gravity
        self.num_joints = num_joints
        self.args = args
        self.fp = torch.tensor(friction_params).to(device=args.graphics_device_id)
        self.fval = torch.empty(self.args.num_envs, self.num_joints)
        if not self.args.disable_gravity:
            print("Compensating for gravitational losses - implementation on input trajectory")
        if not self.args.disable_friction:
            print("Compensating for frictional losses - implementation on isaacgym dof_props")
    
    def getargs(self):
        return self.args

    def setargs(self,args_):
        self.args = args_
    
    def __str__(self):
        return f'Compensator Object instantiated'

    def gravity(self,jacobian,mass):
        """
        Estimated gravitational losses on each joint as a function of jacobian matrix, losses are 
        included in the input trajectory during each step of the simulation.
        """
        gforce = mass.squeeze(0)[:,1:].unsqueeze(-1).unsqueeze(-1) * self.g
        jlink = jacobian[:, :self.num_joints+1, :, :self.num_joints]
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
        """
        vel = vel.contiguous().view(self.args.num_envs,self.num_joints,1).squeeze()
        self.fval = self.fp[:,0]/(1+torch.exp(-self.fp[:,1]*(vel+self.fp[:,2]))) - self.fp[:,0]/(1+torch.exp(-self.fp[:,1]*self.fp[:,2]))
        return self.fval.unsqueeze(-1).float()
    
    def friction_on_torque(self,vel,jacobian,mass):
        """
        Estimated frictional losses on each joint as a function of joint velocity, losses are 
        included in the input trajectory during each step of the simulation. - IDEAL

                                        WORK IN PROGRESS
        """
        print("Compensating for frictional losses - implementation on input trajectory")
        vel = vel.contiguous().view(self.args.num_envs,self.num_joints,1).squeeze()
        self.fval = self.fp[:,0]/(1+torch.exp(-self.fp[:,1]*(vel+self.fp[:,2]))) - self.fp[:,0]/(1+torch.exp(-self.fp[:,1]*self.fp[:,2]))
        return self.fval
    
    def compensate_friction_on_torque(self,traj,vel):
        updtraj = traj + self.friction_on_torque(self,vel)        
        return updtraj


