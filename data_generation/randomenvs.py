import torch
from matplotlib import pyplot as plt
import numpy as np
from isaacgym import gymapi

class randomize():
    """
    Randomization object to be used whenever there is a joint randomization in the simulation,
    current randomization options are: mass, center of mass, stiffness, damping and initial
    position.
    * mass: link rigid_body_properties - link | effort control
    * com: link rigid_body_properties - link | effort control
    * inertia: link rigid_body_properties - link | effort control
    * controller stiffness: dof dof_properties - controller | pos and vel control * ISAACGYM INTERNAL
    * controller damping: dof dof_properties - controller | vel control * ISAACGYM INTERNAL
    * angular damping: asset angular damping - joint | effort control * ISAACGYM INTERNAL
    * linear damping: asset linear damping - joint | effort control *
    * coulomb: dof dof_properties - joint | effort control * ISAACGYM INTERNAL
    * initial_position: dof dof_properties - joint | effort control
    * initial_velocity: dof dof_properties - joint | effort control

    Parameters
    ---
        args (dict) : arguments parsed in generation
    """
    def __init__(self,
                 args):
        self.args = args

    def __str__(self):
        return f'Randomizer Object instantiated'
    
    def getargs(self):
        return self.args
    
    def setargs(self,args_):
        self.args = args_
    
    def decide_minmax(self,l,h):
        mask = l>h
        l[mask],h[mask] = h[mask],l[mask]
        return (l,h)
        
    def mass(self,nom,amount=0):
        """
        Mass randomization for the specified amount in percentage.

        Parameters
        ---
            nom (np.array) : nominal value to randomize around
            amount (float) : amount of percent randomization
        Returns
        ---
            bounds (np.array) : lower bound, upper bound of randomization
        """
        if self.args.random_masses:
            lower_bound,higher_bound = nom*(1-(amount)/100),nom*(1+(amount)/100)
            print('\n Mass randomization:\nLower bound -->\n'+str(lower_bound.transpose().squeeze().round(2)) 
                +'\nHigher bound -->\n' +str(higher_bound.transpose().squeeze().round(2)))    
            return (lower_bound, higher_bound)
        else:
            print('\n No Mass randomization:\nNominal -->\n'+str(nom.transpose().squeeze().round(2))) 
            return nom  
    
    def com(self,nom,amount=0):
        """
        Center of Mass randomization for the specified amount in percentage.

        Parameters
        ---
            nom (np.array) : nominal value to randomize around
            amount (float) : amount of percent randomization
        Returns
        ---
            bounds (np.array) : lower bound, upper bound of randomization
        """
        if self.args.random_coms:
            lower_bound,higher_bound = self.decide_minmax(nom*(1-(amount)/100),nom*(1+(amount)/100))
            print('\n CoM randomization:\nLower bound -->\n'+str(lower_bound.transpose().squeeze().round(4)) 
                +'\nHigher bound -->\n' +str(higher_bound.transpose().squeeze().round(4)))
            return self.decide_minmax(lower_bound,higher_bound)    
        else:
            print('\n No CoM randomization:\nNominal -->\n'+str(nom.transpose().squeeze().round(4))) 
            return nom
        
    def inertia(self,nom,amount=0):
        """
        Inertia randomization for the specified amount in percentage.

        Parameters
        ---
            nom (np.array) : nominal value to randomize around
            amount (float) : amount of percent randomization
        Returns
        ---
            bounds (np.array) : lower bound, upper bound of randomization
        """
        if self.args.random_inertias:
            lower_bound,higher_bound = self.decide_minmax(nom*(1-(amount)/100),nom*(1+(amount)/100))
            print('\n Inertia randomization:\nLower bound -->\n'+str(lower_bound.transpose().squeeze().round(4)) 
                +'\nHigher bound -->\n' +str(higher_bound.transpose().squeeze().round(4)))    
            return self.decide_minmax(lower_bound, higher_bound)
        else:
            print('\n No Inertia randomization:\nNominal -->\n'+str(nom.transpose().squeeze().round(4))) 
            return nom    
    
    def stiffness(self,nom,amount=0):
        """
        Stiffness randomization in DOF control for the specified amount in percentage - used for non effort control.

        Parameters
        ---
            nom (np.array) : nominal value to randomize around
            amount (float) : amount of percent randomization
        Returns
        ---
            bounds (np.array) : lower bound, upper bound of randomization
        """
        if self.args.random_stiffness:
            lower_bound,higher_bound = nom*(1-(amount)/100),nom*(1+(amount)/100)
            print('\n Stiffness randomization:\nLower bound -->\n'+str(lower_bound.transpose().squeeze().round(2)) 
                +'\nHigher bound -->\n' +str(higher_bound.transpose().squeeze().round(2)))    
            return (lower_bound, higher_bound)
        else:
            print('\n No Stiffness randomization:\nNominal -->\n'+str(nom.transpose().squeeze().round(2))) 
            return nom  
    
    def damping(self,nom,amount=0):
        """
        Damping randomization in DOF control for the specified amount in percentage - used for non effort control.

        Parameters
        ---
            nom (np.array) : nominal value to randomize around
            amount (float) : amount of percent randomization
        Returns
        ---
            bounds (np.array) : lower bound, upper bound of randomization
        """
        if self.args.random_damping:
            lower_bound,higher_bound = nom*(1-(amount)/100),nom*(1+(amount)/100)
            print('\n Damping randomization:\nLower bound -->\n'+str(lower_bound.transpose().squeeze().round(2)) 
                +'\nHigher bound -->\n' +str(higher_bound.transpose().squeeze().round(2)))    
            return (lower_bound, higher_bound)
        else:
            print('\n No Damping randomization:\nNominal -->\n'+str(nom.transpose().squeeze().round(2))) 
            return nom
    
    def angdamping(self,nom,amount=0):
        """
        Angular Damping randomization in asset root for the specified amount in percentage - general use.

        Parameters
        ---
            nom (np.array) : nominal value to randomize around
            amount (float) : amount of percent randomization
        Returns
        ---
            bounds (np.array) : lower bound, upper bound of randomization
        """
        if self.args.random_angular_damping:
            lower_bound,higher_bound = nom*(1-(amount)/100),nom*(1+(amount)/100)
            print('\n Angular Damping randomization:\nLower bound -->\n'+str(lower_bound) 
                +'\nHigher bound -->\n' +str(higher_bound))    
            return (lower_bound, higher_bound)
        else:
            print('\n No Angular Damping randomization:\nNominal -->\n'+str(nom)) 
            return nom
        
    def lindamping(self,nom,amount=0):
        """
        Linear Damping randomization in asset root for the specified amount in percentage - general use.

        Parameters
        ---
            nom (np.array) : nominal value to randomize around
            amount (float) : amount of percent randomization
        Returns
        ---
            bounds (np.array) : lower bound, upper bound of randomization
        """
        if self.args.random_angular_damping:
            lower_bound,higher_bound = nom*(1-(amount)/100),nom*(1+(amount)/100)
            print('\n Linear Damping randomization:\nLower bound -->\n'+str(lower_bound) 
                +'\nHigher bound -->\n' +str(higher_bound))    
            return (lower_bound, higher_bound)
        else:
            print('\n No Linear DAmping randomization:\nNominal -->\n'+str(nom)) 
            return nom

    def coulomb(self,nom,amount=0):
        """
        Coulomb Friction randomization in asset root for the specified amount in percentage - general use.

        Parameters
        ---
            nom (np.array) : nominal value to randomize around
            amount (float) : amount of percent randomization
        Returns
        ---
            bounds (np.array) : lower bound, upper bound of randomization
        """
        if self.args.random_coulomb_friction:
            lower_bound,higher_bound = nom*(1-(amount)/100),nom*(1+(amount)/100)
            print('\n Coulomb Friction randomization:\nLower bound -->\n'+str(lower_bound.transpose().squeeze().round(2)) 
                +'\nHigher bound -->\n' +str(higher_bound.transpose().squeeze().round(2)))    
            return (lower_bound, higher_bound)
        else:
            print('\n No Coulomb Friction randomization:\nNominal -->\n'+str(nom.transpose().squeeze().round(2))) 
            return nom
        
    def initpos(self,nom):
        """
        Initial Position randomization from lower bound to upper bound.

        Parameters
        ---
            nom (np.array) : nominal value to randomize around
        Returns
        ---
            bounds (np.array) : lower bound, upper bound of randomization
        """
        if self.args.random_initial_positions:
            print('\n Pos randomization:\nLower bound -->\n'+str(np.round(nom[:,0],2)) 
            +'\nHigher bound -->\n' +str(np.round(nom[:,1],2))) 
            return (nom[:,0],nom[:,1])  
        else:
            print('\n No Pos randomization:\nNominal -->\n'+str(np.average([nom[:,0],nom[:,1]],0).round(2)))
            return np.average([nom[:,0],nom[:,1]],0)
        
    def initvel(self,nom):
        """
        Initial Velocity randomization from 0 to upper bound.
    
        Parameters
        ---
            nom (np.array) : nominal value to randomize around
        Returns
        ---
            bounds (np.array) : lower bound, upper bound of randomization
        """
        if self.args.random_initial_velocities:
            print('\n Vel randomization:\nLower bound -->\n'+str(np.zeros(len(nom))) 
            +'\nHigher bound -->\n' +str(np.round(nom*0.1,2))) 
            return   
        else:
            print('\n No Vel randomization:\nNominal -->\n'+str(np.zeros(len(nom))))
            return np.zeros(len(nom))
        
    def decide_bounds(self,
                      mass_nom,
                      com_nom,
                      inertia_nom,
                      stiffness_nom,
                      damping_nom,
                      pos_end,
                      vel_end,
                      coulomb_nom,
                      angdamp_nom):
        """
        Decides on the randomization amounts from the upper class methods.
        Parameters:
        ---
            nom values (np.array) : all possible nominal values

        Returns:
        ---
            bound dictionary (dict) : randomization amount of all variables
        """

        mass_bounds = self.mass(nom=mass_nom,amount=self.args.random_masses)
        com_bounds = self.com(nom=com_nom,amount=self.args.random_coms)
        inertia_bounds = self.inertia(nom=inertia_nom,amount=self.args.random_inertias)
        stiffness_bounds = self.stiffness(nom=stiffness_nom,amount=self.args.random_stiffness)
        damping_bounds = self.damping(nom=damping_nom,amount=self.args.random_damping)
        coulomb_bounds = self.coulomb(nom=coulomb_nom,amount=self.args.random_coulomb_friction)
        angdamp_bounds = self.angdamping(nom=angdamp_nom,amount=self.args.random_angular_damping)
        pos_bounds = self.initpos(nom=pos_end)
        vel_bounds = self.initvel(nom=vel_end)

        return {
            "mb" : mass_bounds,
            "comb" : com_bounds,
            "ib" : inertia_bounds,
            "sb" : stiffness_bounds,
            "db" : damping_bounds,
            "pb" : pos_bounds,
            "vb" : vel_bounds,
            "cb" : coulomb_bounds,
            "adb" : angdamp_bounds,
        }       

class envinit(randomize):
    """
    Creates the environments and situates the assets according to the args provided by the user, randomization
    is also handled if required by the user.

    Parameters
    ---
        args (dict) : arguments passed in generation
        gym (gym object) : instantiated gym object
        sim (gym object) : current simulation object
        env_lower (np.array) : lower position bounds of the environment
        env_upper (np.array) : lower position bounds of the environment
        numperrow (int) : number of environments per row of domain
        pose (torch.Tensor) : lower position bounds of the environment
        totallinks (int) : number of asset links
        totaljoints (int) : number of asset joints
        fix_base_link (int) : gym internal functionality
        flip_visual_attachments (bool) : gym internal functionality
        armature (float) : controller armature
        disable_gravity (bool) : if True sim ignores gravity
        nominal values ... (np.array) : nominal values of randomizable variables
    """
    def __init__(self,
                 args,
                 gym,
                 sim,
                 env_lower,
                 env_upper,
                 numperrow,
                 pose,
                 totallinks,
                 totaljoints,
                 fix_base_link,
                 flip_visual_attachments,
                 armature,
                 disable_gravity,
                 angdamp_nom,
                 mass_nom,
                 com_nom,
                 inertia_nom,
                 stiffness_nom,
                 damping_nom,
                 coulomb_nom,
                 pos_end,
                 vel_end
                 ):
        
        super().__init__(args)
        self.gym = gym
        self.sim = sim
        self.envl = env_lower
        self.envu = env_upper
        self.npr = numperrow
        self.pose = pose
        self.tlinks = totallinks
        self.tjoints = totaljoints

        self.envs = []
        self.handles = []
        self.handid = []
        self.posl = []
        self.ornl = []
        self.rigid_body_props = []

        #self._stiffeff = 10.0 
        self._stiffeff = 0.0 if self.args.no_controller else 0.0
        self._stiffpos = 800.0 if self.args.no_controller else 0.0
        
        #self._dampeff =  0.0 
        self._dampeff = 10.0 if self.args.no_controller else 0.0
        self._damppos = 40.0 if self.args.no_controller else 0.0

        self.dict = self.decide_bounds(mass_nom,com_nom,inertia_nom,stiffness_nom,damping_nom,pos_end,vel_end,coulomb_nom,angdamp_nom)

        _asset_root = "./" 
        _franka_asset_file = "franka_description/robots/franka_panda.urdf"
        _kuka_asset_file = "kuka_allegro_description/kuka.urdf"
        _asset_file = _franka_asset_file if self.args.type_of_robot=='franka' else _kuka_asset_file
        _asset_options = gymapi.AssetOptions()
        _asset_options.fix_base_link = fix_base_link
        _asset_options.flip_visual_attachments = flip_visual_attachments
        _asset_options.armature = armature
        _asset_options.disable_gravity = disable_gravity    
            
        if self.args.random_angular_damping: 
            _asset_options.angular_damping = np.random.uniform(self.dict["adb"][0],self.dict["adb"][1])
        else:
            _asset_options.angular_damping = angdamp_nom

        print("\n\nLoading asset '%s' from '%s'" % (_asset_file, _asset_root))
        self.asset = self.gym.load_asset(self.sim, _asset_root, _asset_file, _asset_options)
        self.dof_prop = self.gym.get_asset_dof_properties(self.asset)
        self.dof_state = np.ones(totaljoints, gymapi.DofState.dtype)

        self.flower_limits = self.dof_prop['lower']
        self.fupper_limits = self.dof_prop['upper']
        self.fmid = 0.5 * (self.flower_limits + self.fupper_limits)
        if self.args.type_of_robot=='franka':
            self.dof_state["pos"][:7] = np.average([self.flower_limits,self.fupper_limits],0)[:7]
            self.dof_state["vel"][:7].fill(0.0)
        elif self.args.type_of_robot=='kuka':
            self.dof_state["pos"][0] =  0.1
            self.dof_state["pos"][1] =  np.pi/16
            self.dof_state["pos"][2] =  self.fupper_limits[2] * 0.9 
            self.dof_state["pos"][3] =  self.fupper_limits[3] * 0.7
            self.dof_state["pos"][4] =  0
            self.dof_state["pos"][5] =  self.flower_limits[5] * .7 
            self.dof_state["pos"][6] = self.flower_limits[6] * 0.5
            self.dof_state["vel"][:7].fill(0.0)

        self.dof_prop["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
        self.dof_prop["driveMode"][7:].fill(gymapi.DOF_MODE_POS) 

        self.dof_prop["stiffness"][0:7].fill(self._stiffeff)
        self.dof_prop["stiffness"][7:9].fill(self._stiffpos)

        self.dof_prop["damping"][0:7].fill(self._dampeff)
        self.dof_prop["damping"][7:9].fill(self._damppos)
    
        self.dof_prop["friction"] = coulomb_nom
    
    def __str__(self):
        print("Environment Builder Object Instantiated")

    def getsg(self):
        return self.sim,self.gym
    
    def setsg(self,sim_,gym_):
        self.sim = sim_
        self.gym = gym_
        
    def create_envs(self):
        """
        Creates and randomizes envs and assets. EnvDict is utilized in defining the 
        generation process.

        Parameters
        ---
            None

        Returns
        ---
            env_dict (dict) : environment dictionary with extractable info about randomization
        """
        print("Creating %d environments\n" % self.args.num_envs)

        for i in range(self.args.num_envs):
            env = self.gym.create_env(self.sim, self.envl, self.envu, self.npr)
            self.envs.append(env)
            handle = self.gym.create_actor(env, self.asset, self.pose, self.args.type_of_robot, i, 1)
            self.handles.append(handle)
            if self.args.measure_force:
                self.gym.enable_actor_dof_force_sensors(env, handle)
            rigid_body_prop = self.gym.get_actor_rigid_body_properties(env, handle)
            link_mass_tensor = torch.zeros(self.tlinks,dtype=torch.float32,device=self.args.graphics_device_id)
            
            if not self.args.urdf_values:
                if self.args.random_initial_positions:
                    if self.args.type_of_robot=='franka':
                        magnitude = torch.rand(1).uniform_(0.2,0.8).numpy()
                        self.dof_state["pos"][0] =  magnitude * np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                                    torch.rand(1).uniform_(self.flower_limits[0],self.fupper_limits[0]).numpy() 
                        self.dof_state["pos"][1:7] = self.fmid[1:7] + np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                                    0.25 * torch.rand(6).numpy()
                    elif self.args.type_of_robot=='kuka':
                        magnitude = torch.rand(1).uniform_(0.2,0.9).numpy()
                        self.dof_state["pos"][0] =  0.1 + np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                                    torch.rand(1).uniform_(self.flower_limits[0],self.fupper_limits[0]).numpy() 
                        self.dof_state["pos"][1] =  np.pi/16 + np.sign(torch.rand(1).uniform_(-1,1).numpy()) \
                                    * 0.25 * torch.rand(1).numpy()
                        self.dof_state["pos"][2] =  self.fupper_limits[2] * 0.9 + np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                                    0.25 * torch.rand(1).numpy()
                        self.dof_state["pos"][3] =  self.fupper_limits[3] * 0.7 + np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                                    0.25 * torch.rand(1).numpy()
                        self.dof_state["pos"][4] =  0 + np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                                    0.25 * torch.rand(1).numpy()
                        self.dof_state["pos"][5] =  self.flower_limits[5] * .7 + np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                                    0.25 * torch.rand(1).numpy()
                        self.dof_state["pos"][6] = self.flower_limits[6] * 0.5 + np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                                    0.25 * torch.rand(1).numpy()
                            
                if self.args.random_initial_velocities:
                    if self.args.type_of_robot=='franka':
                        magnitude = torch.rand(1).uniform_(0.2,0.8).numpy()
                        self.dof_state["vel"][0] =  magnitude * np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                                    torch.rand(1).uniform_(0,0.5).numpy() 
                        self.dof_state["vel"][1:7] = magnitude * np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                                    torch.rand(6).uniform_(0,2).numpy() 
                    elif self.args.type_of_robot=='kuka':
                        magnitude = torch.rand(1).uniform_(0.2,0.8).numpy()
                        self.dof_state["vel"][0] =  magnitude * np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                                    torch.rand(1).uniform_(0,0.5).numpy() 
                        self.dof_state["vel"][1:7] = magnitude * np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                                    torch.rand(6).uniform_(0,2).numpy() 
                
                if self.args.random_stiffness:
                    if self.args.type_of_robot=='franka':
                        self.dof_prop["stiffness"] = np.random.uniform(self.dict["sb"][0],self.dict["sb"][1],9) 
                    else:
                        self.dof_prop["stiffness"] = np.random.uniform(self.dict["sb"][0],self.dict["sb"][1],7)
                    
                    if self.args.controller:
                            self.dof_prop["stiffness"][0:7].fill(self._stiffeff)
                            self.dof_prop["stiffness"][7:9].fill(self._stiffpos)

                if self.args.random_damping:
                    if self.args.type_of_robot=='franka':
                        self.dof_prop["damping"] = np.random.uniform(self.dict["db"][0],self.dict["db"][1],9) 
                    else:
                        self.dof_prop["damping"] = np.random.uniform(self.dict["db"][0],self.dict["db"][1],7) 
                    
                    if self.args.controller:
                            self.dof_prop["damping"][0:7].fill(self._dampeff)
                            self.dof_prop["damping"][7:9].fill(self._damppos)

                if self.args.random_coulomb_friction:
                    if self.args.type_of_robot=='franka':
                        self.dof_prop["friction"] = np.random.uniform(self.dict["cb"][0],self.dict["cb"][1],9)     
                    else:
                        self.dof_prop["friction"] = np.random.uniform(self.dict["cb"][0],self.dict["cb"][1],7)   

                    if self.args.controller:
                            self.dof_prop["friction"][0:7].fill(0.0)
                            self.dof_prop["friction"][7:9].fill(0.0)  

                for i,link_props in enumerate(rigid_body_prop):
                    if self.args.random_masses:
                        link_props.mass = np.random.uniform(self.dict["mb"][0][i],self.dict["mb"][1][i])
                    else:
                        link_props.mass = self.dict["mb"][i]
                    link_mass_tensor[i] = link_props.mass

                    if self.args.random_coms:
                        link_props.com.x = np.random.uniform(self.dict["comb"][0][i][0],self.dict["comb"][1][i][0])
                        link_props.com.y = np.random.uniform(self.dict["comb"][0][i][1],self.dict["comb"][1][i][1])     
                        link_props.com.z = np.random.uniform(self.dict["comb"][0][i][2],self.dict["comb"][1][i][2])   
                    else:
                        link_props.com.x = self.dict["comb"][i][0]
                        link_props.com.y = self.dict["comb"][i][1]     
                        link_props.com.z = self.dict["comb"][i][2]  

                    if self.args.random_inertias:
                        link_props.inertia.x.x = np.random.uniform(self.dict["ib"][0][i][0],self.dict["ib"][1][i][0])
                        link_props.inertia.x.y = np.random.uniform(self.dict["ib"][0][i][1],self.dict["ib"][1][i][1]) 
                        link_props.inertia.x.z = np.random.uniform(self.dict["ib"][0][i][2],self.dict["ib"][1][i][2]) 
                        link_props.inertia.y.x = np.random.uniform(self.dict["ib"][0][i][1],self.dict["ib"][1][i][1]) 
                        link_props.inertia.y.y = np.random.uniform(self.dict["ib"][0][i][3],self.dict["ib"][1][i][3]) 
                        link_props.inertia.y.z = np.random.uniform(self.dict["ib"][0][i][4],self.dict["ib"][1][i][4])
                        link_props.inertia.z.x = np.random.uniform(self.dict["ib"][0][i][2],self.dict["ib"][1][i][2]) 
                        link_props.inertia.z.y = np.random.uniform(self.dict["ib"][0][i][4],self.dict["ib"][1][i][4]) 
                        link_props.inertia.z.z = np.random.uniform(self.dict["ib"][0][i][5],self.dict["ib"][1][i][5])
                    else:
                        link_props.inertia.x.x = self.dict["ib"][i][0]
                        link_props.inertia.x.y = self.dict["ib"][i][1]
                        link_props.inertia.x.z = self.dict["ib"][i][2]
                        link_props.inertia.y.x = self.dict["ib"][i][1]
                        link_props.inertia.y.y = self.dict["ib"][i][3] 
                        link_props.inertia.y.z = self.dict["ib"][i][4]
                        link_props.inertia.z.x = self.dict["ib"][i][2] 
                        link_props.inertia.z.y = self.dict["ib"][i][4]
                        link_props.inertia.z.z = self.dict["ib"][i][5]   

            self.gym.set_actor_dof_states(env, handle, self.dof_state , gymapi.STATE_ALL)
            self.gym.set_actor_dof_properties(env, handle, self.dof_prop)
            self.gym.set_actor_rigid_body_properties(env, handle, rigid_body_prop,0)
            self.rigid_body_props.append(rigid_body_prop)
                
            hand_handle = self.gym.find_actor_rigid_body_handle(env, handle, "panda_hand" if self.args.type_of_robot=='franka' else "iiwa7_link_7")
            hand_pose = self.gym.get_rigid_transform(env, hand_handle)
            self.posl.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.ornl.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            hand_idx = self.gym.find_actor_rigid_body_index(env, handle, "panda_hand" if self.args.type_of_robot=='franka' else "iiwa7_link_7", gymapi.DOMAIN_SIM)
            self.handid.append(hand_idx)
            
        print(f"Exerpt dof_state of a single env:\n{self.dof_state}\n")
        print(f"Exerpt dof_props of a single env:\n{self.dof_prop}\n")
        print(f"Exerpt rigid_props of a single env:\nMass of link1: {rigid_body_prop[1].mass}\n"
              f"CoM of link1: {rigid_body_prop[1].com}\n"
              f"Inertia x of link1: {rigid_body_prop[1].inertia.x}\n"
              f"Inertia y of link1: {rigid_body_prop[1].inertia.y}\n"
              f"Inertia z of link1: {rigid_body_prop[1].inertia.z}")
              
        print("\n--- Succesfully Created %d environments ----" % self.args.num_envs)  
        return {
            "rbp" :  self.rigid_body_props, 
            "ipos" : self.posl, 
            "iorn" : self.ornl, 
            "envs" : self.envs, 
            "hdls" : self.handles, 
            "hidx" : self.handid, 
            "fass" : self.asset
        }