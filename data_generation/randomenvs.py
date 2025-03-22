import torch
from matplotlib import pyplot as plt
import numpy as np
from isaacgym import gymapi

class randomize():
    """
    Randomization object to be used whenever there is a joint randomization in the simulation,
    current randomization options are: mass, center of mass, stiffness, damping and initial
    position.
    mass: link rigid_body_properties - link
    com: link rigid_body_properties - link
    inertia: link rigid_body_properties - link
    cstiffness: dof dof_properties - controller
    cdamping: dof dof_properties - controller
    angdamping: asset angular damping - joint
    lindamping: asset linear damping - joint
    coulomb: dof dof_properties - joint
    osc_gains: kp - kv osc task - controller
    initial_position: dof dof_properties - joint
    initial_velocity: dof dof_properties - joint

    For effort control the randomizable variables are:
        mass
        com
        inertia
        angdamping
        coulomb
        osc_gains
        initialvelocity
        initial_position
    For position and velocity control additional randomization is possible on:
        cstiffness
        cdamping
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
        Mass randomization for the specified amount in percentage
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
        Center of Mass randomization for the specified amount in percentage
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
        Inertia randomization for the specified amount in percentage
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
        Stiffness randomization in DOF control for the specified amount in percentage - used for non effort control
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
        Damping randomization in DOF control for the specified amount in percentage - used for non effort control
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
        Angular Damping randomization in asset root for the specified amount in percentage - general use
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
        Linear Damping randomization in asset root for the specified amount in percentage - general use
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
        Coulomb Friction randomization in asset root for the specified amount in percentage - general use
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
        Initial Position randomization
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
        Initial Velocity randomization
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
    is also handled if required by args
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
        self.dynamical_inclusion = torch.zeros(0,totallinks,dtype=torch.float32,device=args.graphics_device_id)

        #self._stiffeff = 10.0
        self._stiffeff = 0.0
        self._stiffpos = 800.0
        
        #self._dampeff =  0.0 
        self._dampeff = 10.0
        self._damppos = 40.0

        self.dict = self.decide_bounds(mass_nom,com_nom,inertia_nom,stiffness_nom,damping_nom,pos_end,vel_end,coulomb_nom,angdamp_nom)

        _asset_root = "./" 
        _franka_asset_file = "franka_description/robots/franka_panda.urdf"
        _asset_options = gymapi.AssetOptions()
        _asset_options.fix_base_link = fix_base_link
        _asset_options.flip_visual_attachments = flip_visual_attachments
        _asset_options.armature = armature
        _asset_options.disable_gravity = disable_gravity    
            
        if self.args.random_angular_damping: 
            _asset_options.angular_damping = np.random.uniform(self.dict["adb"][0],self.dict["adb"][1])
        else:
            _asset_options.angular_damping = angdamp_nom

        print("\n\nLoading asset '%s' from '%s'" % (_franka_asset_file, _asset_root))
        self.franka_asset = self.gym.load_asset(self.sim, _asset_root, _franka_asset_file, _asset_options)
        self.dof_prop = self.gym.get_asset_dof_properties(self.franka_asset)
        self.dof_state = np.ones(totaljoints, gymapi.DofState.dtype)

        self.flower_limits = self.dof_prop['lower']
        self.fupper_limits = self.dof_prop['upper']
        self.fmid = 0.5 * (self.flower_limits + self.fupper_limits)
        self.dof_state["pos"][:7] = np.average([self.flower_limits,self.fupper_limits],0)[:7]
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
        Creates and randomizes envs and assets
        """
        print("Creating %d environments\n" % self.args.num_envs)

        for i in range(self.args.num_envs):
            env = self.gym.create_env(self.sim, self.envl, self.envu, self.npr)
            self.envs.append(env)
            franka_handle = self.gym.create_actor(env, self.franka_asset, self.pose, "franka", i, 1)
            self.handles.append(franka_handle)
            if self.args.measure_force:
                self.gym.enable_actor_dof_force_sensors(env, franka_handle)
            rigid_body_prop = self.gym.get_actor_rigid_body_properties(env, franka_handle)
            link_mass_tensor = torch.zeros(self.tlinks,dtype=torch.float32,device=self.args.graphics_device_id)
            
            if self.args.random_initial_positions:
                magnitude = torch.rand(1).uniform_(0.2,0.8).numpy()
                self.dof_state["pos"][0] =  magnitude * np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                            torch.rand(1).uniform_(self.flower_limits[0],self.fupper_limits[0]).numpy() 
                self.dof_state["pos"][1:7] = self.fmid[1:7] + np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                            0.25 * torch.rand(6).numpy()
                
            if self.args.random_initial_velocities:
                magnitude = torch.rand(1).uniform_(0.2,0.8).numpy()
                self.dof_state["vel"][0] =  magnitude * np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                            torch.rand(1).uniform_(0,0.5).numpy() 
                self.dof_state["vel"][1:7] = magnitude * np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                            torch.rand(6).uniform_(0,2).numpy() 

            if self.args.random_stiffness:
                self.dof_prop["stiffness"] = np.random.uniform(self.dict["sb"][0],self.dict["sb"][1],9) 

            if self.args.random_damping:
                self.dof_prop["damping"] = np.random.uniform(self.dict["db"][0],self.dict["db"][1],9) 

            if self.args.random_coulomb_friction:
                self.dof_prop["friction"] = np.random.uniform(self.dict["cb"][0],self.dict["cb"][1],9)     

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

            self.gym.set_actor_dof_states(env, franka_handle, self.dof_state , gymapi.STATE_ALL)
            self.gym.set_actor_dof_properties(env, franka_handle, self.dof_prop)
            self.gym.set_actor_rigid_body_properties(env, franka_handle, rigid_body_prop,0)
            self.dynamical_inclusion = torch.cat((self.dynamical_inclusion,link_mass_tensor.unsqueeze(0)) , dim = 0)
                
            hand_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
            hand_pose = self.gym.get_rigid_transform(env, hand_handle)
            self.posl.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.ornl.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
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
            "mv" :   self.dynamical_inclusion.unsqueeze(0), 
            "ipos" : self.posl, 
            "iorn" : self.ornl, 
            "envs" : self.envs, 
            "hdls" : self.handles, 
            "hidx" : self.handid, 
            "fass" : self.franka_asset
        }