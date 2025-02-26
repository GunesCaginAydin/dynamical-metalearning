# %%
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from genutil import *
from control import trajectory
from randomize import randomize
import math
import numpy as np
import torch

import os
from pathlib import Path
import time
import sys
import subprocess

torch.cuda.empty_cache()

# %%
args = parser(description="FrankaDataGen",params=[]).parse_arguments()
data = reader(path='data.json').read_data()

# %%
TOTAL_COORDS = data["TOTAL_COORDS"]
TOTAL_JOINTS = data["TOTAL_JOINTS"]
TOTAL_LINKS = data["TOTAL_LINKS"]

POS_END = np.array(data["POS_END"])
VEL_END = np.array(data["VEL_END"])
ACC_END = np.array(data["ACC_END"])
JER_END = np.array(data["JER_END"])
TOR_END = np.array(data["TOR_END"])
TAC_END = np.array(data["TAC_END"])
FRICTION = np.array(data["FRICTION"])
GRAVITY = data["GRAVITY"]

MASS_NOM = np.array(data["MASS_NOM"])
COM_NOM = np.array(data["COM_NOM"])
INERTIA_NOM = np.array(data["INERTIA_NOM"])
STIFFNESS_NOM = np.array(data["STIFFNESS_NOM"])
DAMPING_NOM = np.array(data["DAMPING_NOM"])
OSC_NOM = np.array(data["OSC_NOM"])

SOLVER_TIME = data["SOLVER_TIME"]/60
SUBSTEPS = data["SUBSTEPS"]
SOLVER_TYPE = data["SOLVER_TYPE"]
NUM_POS_ITER = data["NUM_POS_ITER"]
NUM_VEL_ITER = data["NUM_VEL_ITER"]

FIX_BASE_LINK = data["FIX_BASE_LINK"]
FLIP_VISUAL_ATTACHMENTS = data["FLIP_VISUAL_ATTACHMENTS"]
ARMATURE = data["ARMATURE"]

DISABLE_GRAVITY = args.disable_gravity     
DISABLE_FRICTION = args.disable_friction          
CONTROL_IMPOSED = args.control_imposed                           
CONTROL_IMPOSED_FILE = args.control_imposed_file                  
OSC_TASK = args.osc_task                                          
TYPE_OF_OSC = args.type_of_osc                            
NOPLOT = args.no_plot                             

RANDOM_INIT_POS = args.random_initial_positions
RANDOM_INIT_VEL = args.random_initial_velocities        
RANDOM_MASS = float(args.random_masses)
RANDOM_INERTIA = float(args.random_inertias)
RANDOM_COM = float(args.random_coms)
RANDOM_STIFF = float(args.random_stiffness)
RANDOM_DAMP = float(args.random_damping)
RANDOM_KP_KV = args.random_osc_gains                              

INPUT_TYPE = args.type_of_input                          

VISUALIZE = args.visualize                            
NOSAVE = args.no_save
DYNAMICAL_INCLUSION = args.dynamical_inclusion       
TYPE_OF_DATASET = args.type_of_dataset                           
FREQUENCY = args.frequency                                        
NUM_ENVS = args.num_envs                                          
NUM_RUNS = args.num_runs                                    
MAX_ITER = args.num_iters

NUM_THREADS = args.num_threads
USE_GPU = args.use_gpu
USE_GPU_PIPELINE = args.use_gpu_pipeline
COMPUTE_DEVICE_ID = args.compute_device_id
GRAPHICS_DECIDE_ID = args.graphics_device_id
PHYS_ENGINE = args.physics_engine

# %%
iter_run = 1

complementary_dataset = 'train' if TYPE_OF_DATASET == 'test' else 'test'

output_folder = Path("./out_tensors/")
output_folder.mkdir(exist_ok=True)

current_folder = Path("./out_tensors/"+TYPE_OF_DATASET)
current_folder.mkdir(exist_ok=True)

complementary_folder = Path("./out_tensors/"+complementary_dataset)
complementary_folder.mkdir(exist_ok=True)

generated_seed = np.random.randint(0,9999)
print("\nGenerated Seed: "+str(generated_seed))

copies = True
while copies:    
    list_of_tensors = os.listdir("./out_tensors/"+complementary_dataset)
    generated_seed_str = 'seed_'+str(generated_seed)
    if any(generated_seed_str in s for s in list_of_tensors):
        generated_seed = np.random.randint(0,9999)
        print("\nCurrent Folder: "+TYPE_OF_DATASET+" --> Found the same seed in "+complementary_dataset)
        print("\nGenerated seed:"+str(generated_seed)) 
    else:
        copies=False

torch.manual_seed(generated_seed) 
#------------------------------------------------------------------------------------------------#         
print("\n ---------- This is run number :",iter_run, "----------")

# %%
gym = gymapi.acquire_gym()

TYPE_OF_CONTACT = gymapi.ContactCollection.CC_LAST_SUBSTEP 
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = SOLVER_TIME
sim_params.substeps = SUBSTEPS
if gymapi.SIM_PHYSX == PHYS_ENGINE:
    sim_params.physx.solver_type = SOLVER_TYPE
    sim_params.physx.num_position_iterations = NUM_POS_ITER
    sim_params.physx.num_velocity_iterations = NUM_VEL_ITER
    sim_params.physx.num_threads = NUM_THREADS
    sim_params.physx.use_gpu = USE_GPU
    sim_params.physx.contact_collection = TYPE_OF_CONTACT
else:
    raise Exception("Only PhysX is available")
sim_params.use_gpu_pipeline = USE_GPU_PIPELINE
sim = gym.create_sim(COMPUTE_DEVICE_ID, GRAPHICS_DECIDE_ID, PHYS_ENGINE, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

asset_root = "./" 
franka_asset_file = "franka_description/robots/franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = FIX_BASE_LINK
asset_options.flip_visual_attachments = FLIP_VISUAL_ATTACHMENTS
asset_options.armature = ARMATURE
asset_options.disable_gravity = DISABLE_GRAVITY
#asset_options.angular_damping = 100

print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
FRANKA_EFFORT_LIMITS = franka_dof_props['effort']
FRANKA_LOWER_LIMITS = franka_dof_props['lower']
FRANKA_UPPER_LIMITS = franka_dof_props['upper']
FRANKA_RANGES = FRANKA_UPPER_LIMITS - FRANKA_LOWER_LIMITS
FRANKA_MIDS =  0.5 * (FRANKA_UPPER_LIMITS + FRANKA_LOWER_LIMITS) 
FRANKA_NUM_DOFS = len(franka_dof_props)

# %%
default_dof_state = np.ones(TOTAL_JOINTS, gymapi.DofState.dtype)
default_dof_state["pos"][:7] = np.average([franka_dof_props["lower"],franka_dof_props["upper"]],0)[:7]
default_dof_state["vel"][:7].fill(0.0)
#franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
#franka_dof_props["stiffness"][7:].fill(6000.0)
#franka_dof_props["damping"][7:].fill(1000.0)   
#franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)

num_per_row = int(math.sqrt(NUM_ENVS))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)
pose.r = gymapi.Quat(0, 0, 0, 1)

print("Creating %d environments" % NUM_ENVS)

envs = []
franka_handles = []
hand_idxs = []
init_pos_list = []
init_orn_list = []
if RANDOM_MASS or RANDOM_COM or RANDOM_INERTIA or RANDOM_STIFF or RANDOM_DAMP or RANDOM_INIT_POS or RANDOM_INIT_VEL or RANDOM_KP_KV:
    mix = randomize(args,description="FrankaDataGen",params=[])
    mass_bounds = mix.mass(nom=MASS_NOM,amount=RANDOM_MASS)
    com_bounds = mix.com(nom=COM_NOM,amount=RANDOM_COM)
    inertia_bounds = mix.inertia(nom=INERTIA_NOM,amount=RANDOM_INERTIA)
    stiffness_bounds = mix.stiffness(nom=STIFFNESS_NOM,amount=RANDOM_STIFF)
    damping_bounds = mix.damping(nom=DAMPING_NOM,amount=RANDOM_DAMP)
    pos_bounds = mix.initpos(nom=POS_END)
    osck_bounds = mix.osck()
dynamical_inclusion = torch.zeros(0,TOTAL_LINKS,dtype=torch.float32,device=args.graphics_device_id)

print(franka_dof_props)

# %%
for i in range(NUM_ENVS):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    franka_handle = gym.create_actor(env, franka_asset, pose, "franka", i, 2)
    franka_handles.append(franka_handle)
    rigid_body_properties = gym.get_actor_rigid_body_properties(env, franka_handle)
    link_mass_tensor = torch.zeros(TOTAL_LINKS,dtype=torch.float32,device=args.graphics_device_id)
    franka_dof_prop = gym.get_actor_dof_properties(env, franka_handle)    
    franka_dof_prop["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
    franka_dof_prop["driveMode"][7:].fill(gymapi.DOF_MODE_POS) 

    if RANDOM_INIT_POS:
        magnitude = torch.rand(1).uniform_(0.2,0.8).numpy()
        default_dof_state["pos"][0] =  magnitude * np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                    torch.rand(1).uniform_(FRANKA_LOWER_LIMITS[0],FRANKA_UPPER_LIMITS[0]).numpy() 
        default_dof_state["pos"][1:7] = FRANKA_MIDS[1:7] + np.sign(torch.rand(1).uniform_(-1,1).numpy()) * \
                    0.25 * torch.rand(6).numpy()

    if RANDOM_STIFF:
        True
        #franka_dof_prop["stiffness"] = np.random.uniform(stiffness_bounds[0],stiffness_bounds[1],9)     
        
    
    if RANDOM_DAMP:
        True
        #franka_dof_prop["damping"] = np.random.uniform(damping_bounds[0],damping_bounds[1],9)
    
    franka_dof_prop["stiffness"][:7].fill(0.0)        
    franka_dof_prop["damping"][:7].fill(0.0)        
    #franka_dof_prop["friction"][:7].fill(0.1)
    #franka_dof_prop["friction"][7:].fill(0.01)

    franka_dof_prop["stiffness"][7:].fill(800.0)
    franka_dof_prop["damping"][7:].fill(40.0)

    for i,link_props in enumerate(rigid_body_properties):
        if RANDOM_MASS:
            link_props.mass = np.random.uniform(mass_bounds[0][i],mass_bounds[1][i])
        else:
            True
            #continue
            #link_props.mass = MASS_NOM[i]

        link_mass_tensor[i] = link_props.mass
        
        if RANDOM_COM:
            link_props.com.x = np.random.uniform(com_bounds[0][i][0],com_bounds[1][i][0])
            link_props.com.y = np.random.uniform(com_bounds[0][i][1],com_bounds[1][i][1])     
            link_props.com.z = np.random.uniform(com_bounds[0][i][2],com_bounds[1][i][2])   
        else:
            True
            #continue
            #link_props.com.x = COM_NOM[i][0]
            #link_props.com.y = COM_NOM[i][1]
            #link_props.com.z = COM_NOM[i][2]    

        if RANDOM_INERTIA:
            link_props.inertia.x.x = np.random.uniform(inertia_bounds[0][i][0],inertia_bounds[1][i][0])
            link_props.inertia.x.y = np.random.uniform(inertia_bounds[0][i][1],inertia_bounds[1][i][1]) 
            link_props.inertia.x.z = np.random.uniform(inertia_bounds[0][i][2],inertia_bounds[1][i][2]) 
            link_props.inertia.y.x = np.random.uniform(inertia_bounds[0][i][1],inertia_bounds[1][i][1]) 
            link_props.inertia.y.y = np.random.uniform(inertia_bounds[0][i][3],inertia_bounds[1][i][3]) 
            link_props.inertia.y.z = np.random.uniform(inertia_bounds[0][i][4],inertia_bounds[1][i][4])
            link_props.inertia.z.x = np.random.uniform(inertia_bounds[0][i][2],inertia_bounds[1][i][2]) 
            link_props.inertia.z.y = np.random.uniform(inertia_bounds[0][i][4],inertia_bounds[1][i][4]) 
            link_props.inertia.z.z = np.random.uniform(inertia_bounds[0][i][5],inertia_bounds[1][i][5])  
        else: 
            True
            #continue            
            #link_props.inertia.x.x = INERTIA_NOM[i][0]
            #link_props.inertia.x.y = INERTIA_NOM[i][1]
            #link_props.inertia.x.z = INERTIA_NOM[i][2]
            #link_props.inertia.y.x = INERTIA_NOM[i][1] 
            #link_props.inertia.y.y = INERTIA_NOM[i][3] 
            #link_props.inertia.y.z = INERTIA_NOM[i][4]
            #link_props.inertia.z.x = INERTIA_NOM[i][2] 
            #link_props.inertia.z.y = INERTIA_NOM[i][4]
            #link_props.inertia.z.z = INERTIA_NOM[i][5]
    
    gym.set_actor_dof_states(env, franka_handle, default_dof_state , gymapi.STATE_ALL)
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_prop)
    gym.set_actor_rigid_body_properties(env, franka_handle, rigid_body_properties,0)
    dynamical_inclusion = torch.cat( (dynamical_inclusion,link_mass_tensor.unsqueeze(0)) , dim = 0)
        
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_orn_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

dynamical_inclusion = dynamical_inclusion.unsqueeze(0)

print("\n--- Succesfully Created %d environments ----" % NUM_ENVS)    

# %%
gym.prepare_sim(sim)

init_pos = torch.Tensor(init_pos_list).view(NUM_ENVS, 3)
init_orn = torch.Tensor(init_orn_list).view(NUM_ENVS, 4)

if args.use_gpu_pipeline:
    init_pos = init_pos.to('cuda:0')
    init_orn = init_orn.to('cuda:0')

pos_des = init_pos.clone()
orn_des = init_orn.clone()

_jacobian = gym.acquire_jacobian_tensor(sim, "franka") # (10,6,9)
jacobian = gymtorch.wrap_tensor(_jacobian)

hand_index = gym.get_asset_rigid_body_dict(franka_asset)["panda_hand"]
j_eef = jacobian[:, hand_index - 1, :]

_massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
mm = gymtorch.wrap_tensor(_massmatrix)

_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_vel = dof_states[:, 1].view(NUM_ENVS, 9, 1)
dof_pos = dof_states[:, 0].view(NUM_ENVS, 9, 1)

# ------------------------------- initializing buffer tensor ------------------------------------

buffer_position = torch.empty((0,NUM_ENVS,TOTAL_COORDS), dtype=torch.float32).to(device=args.graphics_device_id) 
buffer_target  = torch.empty((0,NUM_ENVS,3), dtype=torch.float32)

if CONTROL_IMPOSED_FILE or OSC_TASK:
    print("Buffer intilized for xy_task or control_imposed_by_file \n ")
    buffer_control_action = torch.zeros((NUM_ENVS,TOTAL_JOINTS,1), dtype=torch.float32).to(device=args.graphics_device_id)
elif DYNAMICAL_INCLUSION:
    buffer_control_action = torch.zeros((1,NUM_ENVS,TOTAL_JOINTS+dynamical_inclusion.shape[2]), dtype=torch.float32).to(device=args.graphics_device_id)  
else:
    print("Buffer intilized for control imposed directly in Nm\n ")
    buffer_control_action = torch.zeros((1,NUM_ENVS,TOTAL_JOINTS), dtype=torch.float32).to(device=args.graphics_device_id) 

# %%
control = trajectory(NUM_ENVS, MAX_ITER+1, TOTAL_JOINTS, FREQUENCY, INPUT_TYPE)
if INPUT_TYPE == 'MS':
    control_action,control_diff = control.sin()
elif INPUT_TYPE == 'CH':
    control_action,control_diff = control.chirp()
elif INPUT_TYPE == 'TRAPZ':
    control_action,control_diff = control.trapz()
elif INPUT_TYPE == 'IMP':
    control_action,control_diff = control.imp()

control_action = control_action.to(device=args.graphics_device_id)
control_diff = control_diff.to(device=args.graphics_device_id)

black_list = []  
out_of_range_quaternion = [] 
saturated_ll_idxs = []
saturated_ul_idxs = []

if not DISABLE_FRICTION or not DISABLE_GRAVITY:
    comp = compensate(args,gravity=GRAVITY,friction_params=FRICTION,description="FrankaDataGen",params=[],num_joints=TOTAL_JOINTS)

print(gym.get_actor_dof_properties(env,franka_handle))
# %%
#control.plot_trajectory(control_action, NUM_ENVS, TOTAL_JOINTS)

# %%
#control.plot_trajectory(control_diff, NUM_ENVS, TOTAL_JOINTS)

# %%
# ================================ SIMULATIONS STARTS =====================================
if VISUALIZE:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")

if VISUALIZE:
    cam_pos = gymapi.Vec3(4, 4, 4)
    cam_target = gymapi.Vec3(-4, -3, -2)
    middle_env = envs[NUM_ENVS // 2 + num_per_row // 2]
    gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

if VISUALIZE:
    condition_window = gym.query_viewer_has_closed(viewer)
else:
    condition_window = 0 

_dof_states = gym.acquire_dof_state_tensor(sim)

itr = 0
ts = time.perf_counter()

while not condition_window  and itr <= MAX_ITER-1: # while not gym.query_viewer_has_closed(viewer):  #ORIGINAL

        itr += 1
        # Update jacobian and mass matrix and contact collection
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        gym.refresh_mass_matrix_tensors(sim)
        gym.refresh_net_contact_force_tensor(sim) 

        # Get current hand poses
        pos_cur = rb_states[hand_idxs, :3]
        orn_cur = rb_states[hand_idxs, 3:7]
        
        # Set desired hand positions # ORIGINAL
        if  OSC_TASK == True:
            # radius = 0.05
            if itr ==1:
                radius = torch.rand((1,NUM_ENVS)).uniform_(0.01,0.12).to(device=args.graphics_device_id)
                period = torch.rand(1).uniform_(20,100).to(device=args.graphics_device_id)
                z_speed = torch.rand(1).uniform_(0.1,0.4).to(device=args.graphics_device_id)
                sign = torch.sign(torch.rand(1).uniform_(-1,1)).to(device=args.graphics_device_id)
                # offset =  torch.sign(torch.rand(1).uniform_(-0.3,0.3)).to(device=args.graphics_device_id)
            #This was used for testC!
            if  TYPE_OF_OSC == 'VS': # Vertical spyral
                pos_des[:, 0] = init_pos[:, 0] + math.sin(itr / period) * radius 
                pos_des[:, 1] = init_pos[:, 1] + math.cos(itr / period) * radius
                pos_des[:, 2] = init_pos[:, 2] - 0.1 + sign * z_speed * itr/MAX_ITER
            elif TYPE_OF_OSC == 'FS':
                radius = 0.1           # Fixed spyral
                pos_des[:, 0] = init_pos[:, 0] + math.sin(itr / 80) * radius 
                pos_des[:, 1] = init_pos[:, 1] + math.cos(itr / 80) * radius
                pos_des[:, 2] = init_pos[:, 2] + - 0.1 + 0.2 * itr/MAX_ITER
            elif TYPE_OF_OSC == 'FC': # Fixed circle
                # radius = 0.1
                pos_des[:, 0] = init_pos[:, 0] 
                pos_des[:, 1] = init_pos[:, 1] + math.sin(itr / 50) * radius #EDITED
                pos_des[:, 2] = init_pos[:, 2] + math.cos(itr / 50) * radius #EDITED

            # pos_des[:, 0] = init_pos[:, 0] - 0.05
            # pos_des[:, 1] = math.sin(itr / 50) * 0.15
            # pos_des[:, 2] = init_pos[:, 2] + math.cos(itr / 50) * 0.15

            # Solve for control (Operational Space Control)
            m_inv = torch.inverse(mm)
            m_eef = torch.inverse(j_eef @ m_inv @ torch.transpose(j_eef, 1, 2)) 
            orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
            orn_err = lambda orn_des, orn_cur : orn_des - orn_cur
            pos_err = kp * (pos_des - pos_cur)
            dpose = torch.cat([pos_err, orn_err(orn_des,orn_cur)], -1)
            u = torch.transpose(j_eef, 1, 2) @ m_eef @ (kp * dpose).unsqueeze(-1) - kv * mm @ dof_vel 

        # In these case, control is manually imposed (directly in Nm)
                
        if CONTROL_IMPOSED and not CONTROL_IMPOSED_FILE:
            # by function
            # u_custom = my_control_action[:,:,itr].unsqueeze(-1).to(device=args.graphics_device_id) 
            u_custom = control_action[:,:,itr].unsqueeze(-1)
            u = u_custom.contiguous().to(device=args.graphics_device_id)

        if CONTROL_IMPOSED_FILE:
            # by file
            u = control_action[:,:,itr].unsqueeze(-1).contiguous()
            
        # ok for OSC file with gravity
        if DISABLE_GRAVITY == False and CONTROL_IMPOSED_FILE:
            compensate = True
        # ok for OSC file without gravity
        if CONTROL_IMPOSED_FILE==True:
            compensate = False

        #if compensate:              #if not disable_gravity_flag:   
            #if itr ==1: 
            #    print("I'm compensating")
            #dof_count = gym.get_asset_dof_count(franka_asset)
            #g = torch.zeros(NUM_ENVS, dof_count+1, 6, 1, dtype=torch.float, device=args.graphics_device_id)
            #g[:, :, 2, :] = 9.81

            # dynamical_inclusion.squeeze(0)[:,1:] --> pick the bodies excluding the first one,
            # which is fixed to the ground! jacobian is [num_envs,10,6,9], where 10 is the number of body.
            #  Look at the documentation for further explanation about jacobian and how they're calculated. 
            # This compensation is taken from https://github.com/NVlabs/oscar/blob/main/oscar/agents/franka.py

            #g_force = dynamical_inclusion.squeeze(0)[:,1:].unsqueeze(-1).unsqueeze(-1) * g
            #j_link = jacobian[:, :dof_count+1, :, :dof_count]
            #g_torque = (torch.transpose(j_link, 2, 3) @ g_force).squeeze(-1)
            #g_torque = torch.sum(g_torque, dim=1, keepdim=False)
            #g_torque = g_torque.unsqueeze(-1)
            #u += g_torque       # u = u + g_torque --> more efficent #######################################################################################


        try:
            gtorque = comp.gravity(jacobian,dynamical_inclusion)
            u = u + gtorque

            ftorque = comp.friction(dof_states[:,1])
        #   u = u + ftorque
            #for i,(env,franka_handle) in enumerate(zip(envs,franka_handles)):
            #    franka_dof_prop = gym.get_actor_dof_properties(env,franka_handle)
            #    franka_dof_prop["damping"] = ftorque.cpu()[i,:,0]
            #    gym.set_actor_dof_properties(env,franka_handle,franka_dof_prop)
        except:
            raise(ValueError)
    # ------------------------------------- APPLICATION OF U -------------------------------------------------
            
        # Set control action as torque tensor, or position tensor
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(u))

    # ------------------------------------ CONTACT COLLECTION -------------------------------------------------

        # ATTENTION! gym.refresh_net_contact_force_tensor(sim) need to be added! remember
        _contact_forces = gym.acquire_net_contact_force_tensor(sim) 
        # returns: (envs x 11, 3) --> 16 envs --> (176,3), xyz components for each body
        # wrap it in a PyTorch Tensor
        contact_forces = gymtorch.wrap_tensor(_contact_forces)

    # -- COLLISION DETECTION  
        #body contact contains all the nonzero indeces about contact forces
        body_contact = torch.nonzero(abs(contact_forces)>0.01)

        # This for processes in all the istant of the simulation the indexes to be taken into account: 
        # Anyway, in the black list they enter only the first time.

        for j in range(body_contact.shape[0]):
            _body_contact = body_contact[j].to("cpu").numpy()
            env_idx_collision = int(np.ceil(_body_contact[0]/TOTAL_LINKS)-1)

            if  not env_idx_collision in black_list : 
                
                black_list.append(env_idx_collision)
                # print("In env: ",env_idx_collision,", body n°",_body_contact[0] - body_links * env_idx_collision, "collided: step ",itr)
                # print(contact_forces[(body_links) * (env_idx_collision) ,:])
                # print("contact indexes:",body_contact[j])
                # print("body_contact: ",contact_forces[body_contact[j,0],:].to("cpu").numpy())

                # Visual purposes - coloring in red the colliding robots
                
                mesh = gymapi.MESH_VISUAL_AND_COLLISION # MESH_VISUAL works fine
                color = gymapi.Vec3(.9,.25,.15)
                env_handle = gym.get_env(sim,env_idx_collision)
                for k in range(TOTAL_LINKS):
                    gym.set_rigid_body_color(env_handle,franka_handle, k , mesh ,color)

        # -------------------------------------- Step the physics --------------------------------------------------
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if VISUALIZE:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False) # True - collision | False - visual
            gym.sync_frame_time(sim)
        # --------------------------------------- Stacking in the buffers -------------------------------------------------
        if CONTROL_IMPOSED and CONTROL_IMPOSED_FILE:
                #from file
                if itr==1:
                    print("\n Torque Imposed by FILE")
                control_action_ext = u
        
        elif CONTROL_IMPOSED and not CONTROL_IMPOSED_FILE:
            if itr==1:
                print("\n Torque Imposed by function")
            control_action_ext = u
            control_action_ext = control_action_ext.view(1,NUM_ENVS,9)
            control_action_ext = control_action_ext[:,:,:TOTAL_JOINTS]

        if OSC_TASK:
            if itr==1:
                print(f"\n Torque Imposed by OSC - {TYPE_OF_OSC}")
            control_action_ext = u  
            # if xy_task , I want [2, 9, 1001]

        # dynamical inclusion happens, both if the masses changes, both if the masses don't change
        if DYNAMICAL_INCLUSION == True:
            control_action_ext = torch.cat((control_action_ext,dynamical_inclusion),dim=2)
        if CONTROL_IMPOSED_FILE or OSC_TASK:
            buffer_control_action = torch.cat((buffer_control_action, control_action_ext), 2)  
        else:
            buffer_control_action = torch.cat((buffer_control_action,control_action_ext), 0) 

        dof_states = gymtorch.wrap_tensor(_dof_states)
        dof_pos = dof_states[:, 0]
        # Be careful using view | movedim is an alternative most of the times
        dof_pos = dof_pos.to("cpu").view(1,NUM_ENVS,9)
        dof_pos = dof_pos[:,:,:7]  # pick up only the 7 dofs! gripper fingers are excluded for output.
        #print(dof_pos)
        pos_cur = pos_cur.to("cpu").view(1,NUM_ENVS,3)  # x y z
        orn_cur = orn_cur.to("cpu").view(1,NUM_ENVS,4)  # orientation angles

        # -------------------------- INCLUDING dof_pos for 7 - dimension state space -------------------------------

        full_pose = torch.cat((pos_cur,orn_cur,dof_pos),dim = 2).to(device=args.graphics_device_id) 
        # stacking onto the 0 dimension, each acquisition        
        buffer_position = torch.cat((buffer_position, full_pose), 0)  
        pos_desired=pos_des.to("cpu").view( 1,NUM_ENVS, 3) 
        if OSC_TASK:
            buffer_target = torch.cat((buffer_target, pos_desired), 0)  #target position

        #print(full_pose.shape)
        #print(buffer_position.shape)

        # --------------------------- Abnormal change in quaternion ---------------------------------------------------
        if itr > 2: #####################################################################3

            #This is the increment of the undesired changes in quaternion
            increment = abs(buffer_position[itr-1,:,4:7]- buffer_position[itr-2,:,4:7])

            #This is the increment of the undesired changes in all dofs!
            # increment = abs(buffer_position[itr-1,:,:7]- buffer_position[itr-2,:,:7])

            #These are the out of range simulations
            out_of_range = torch.nonzero(increment > .1) #  np.rad2deg(.15) = 9°

            if out_of_range.shape[0] != 0:   
                abnormal_idxs = out_of_range.shape[0]
                for j in range(out_of_range.shape[0]):
                    out_of_range_quaternion.append(int(out_of_range[j,0].to('cpu').numpy()))

        # ---------------------------- Saturation check | Position --------------------------------------------------

        ll = torch.tensor(FRANKA_LOWER_LIMITS[:7]).repeat(NUM_ENVS,1) 
        ul = torch.tensor(FRANKA_UPPER_LIMITS[:7]).repeat(NUM_ENVS,1) 
        saturation_ll = torch.nonzero(abs(dof_pos-ll) < 0.05)
        saturation_ul = torch.nonzero(abs(dof_pos-ul) < 0.05)

        if saturation_ll.shape[0] != 0:
            abnormal_idxs = saturation_ll.shape[0]
            for j in range(saturation_ll.shape[0]):
                saturated_ll_idxs.append(int(saturation_ll[j,1].to('cpu').numpy()))

        if saturation_ul.shape[0] != 0:
            abnormal_idxs = saturation_ul.shape[0]
            for j in range(saturation_ul.shape[0]):
                saturated_ul_idxs.append(int(saturation_ul[j,1].to('cpu').numpy()))

        # ---------------------------- Saturation check | Torque --------------------------------------------------
        if OSC_TASK:
            torques_limit = torch.tensor(FRANKA_LOWER_LIMITS[:7]).repeat(NUM_ENVS,1).to(device=args.graphics_device_id) 
            saturation_torques = torch.nonzero( (torques_limit - abs(control_action.squeeze(-1)[:,:7]) ) < 1)

            if saturation_torques.shape[0] != 0:
                abnormal_idxs = saturation_torques.shape[0]
                for j in range(saturation_torques.shape[0]):
                    saturated_ul_idxs.append(int(saturation_torques[j,0].to('cpu').numpy())) # append in another

        # ------------------------------------ CLEANING BUFFERS --------------------------------------------
        
        ## in this block, all the envs that have collided at least one step, are removed from the acquisitioncle

        if itr == MAX_ITER:
            
            saturation_idxs = list(set(saturated_ul_idxs + saturated_ll_idxs))
            print("\n----Number of saturated simulations: ",len(saturation_idxs),"/", NUM_ENVS,"----\n" ) 

            out_of_range_quaternion = list(set(out_of_range_quaternion))
            # this is the numbers of the colliding simulations
            print("---- Number of simulations with abnormal change in quaternions: ",len(out_of_range_quaternion),"/", NUM_ENVS,"----\n" )  
            print("---- Number of the colliding simulations: ",len(black_list),"/", NUM_ENVS,"----\n" ) 

            # Maybe a colliding env and abnormal change env coincide! Use always set
            black_list = black_list + out_of_range_quaternion + saturation_idxs 
            black_list = list(set(black_list))
            failed_percentage = len(black_list)/NUM_ENVS*100
            print("\n---- Number of bad simulations: ",len(black_list),"/", NUM_ENVS,"----\n" )  
            print("Percentage of total discarded simulations:", round(failed_percentage,2), "%") 

            #---- excluding all the invalid environment from the simulation ------ 
            non_valid_envs = len(black_list)
            num_valid_envs = NUM_ENVS - non_valid_envs

            black_list.sort(reverse=True)
            # for i in range(non_valid_envs):

            #     row_exclude = black_list[i]
            #     buffer_control_action = torch.cat((buffer_control_action [:,:row_exclude,:],
            #                                             buffer_control_action [:,row_exclude+1:,:]),1)
            #     buffer_position = torch.cat((buffer_position [:,:row_exclude,:],
            #                                             buffer_position [:,row_exclude+1:,:]),1)
                
            #for i in range(non_valid_envs):

            #    row_exclude = black_list[i]
            #    if CONTROL_IMPOSED and not CONTROL_IMPOSED_FILE:
            #        buffer_control_action = torch.cat((buffer_control_action [:,:row_exclude,:],
            #                                                buffer_control_action [:,row_exclude+1:,:]),1)
            #    elif OSC_TASK:
            #        #in OSC task control action has a different order
            #        buffer_control_action = torch.cat((buffer_control_action [:row_exclude,:,:],
            #                            buffer_control_action [row_exclude+1:,:,:]),0)
            #        buffer_target = torch.cat((buffer_position [:,:row_exclude,:],
            #                            buffer_position [:,row_exclude+1:,:]),1)
            #    
            #    buffer_position = torch.cat((buffer_position [:,:row_exclude,:],
            #        buffer_position [:,row_exclude+1:,:]),1)
                
tf = time.perf_counter()

if VISUALIZE:
    gym.destroy_viewer(viewer)

# =================================== END OF SIMULATION ==================================================
gym.destroy_sim(sim)
print(f"Time taken for simulation is {tf - ts}")

# %%
if NOSAVE==False:
    tensormgmt = savedata(args,description="FrankaDataGen",params=[],
                        control_trajectory=buffer_control_action,pose=buffer_position,
                        seed=generated_seed,valid_envs=num_valid_envs,target=[],
                        dynamical_inclusion=dynamical_inclusion,collision=None)
    tensormgmt.save_tensors()
else:
    print("Input/Output Tensors are not saved")
torch.cuda.empty_cache()
# %%
if NOPLOT==False:
    dataprocessor = postprocessor(TOTAL_JOINTS,TOTAL_COORDS,args,description="FrankaDataGen",params=[],
                                control_trajectory=buffer_control_action,pose=buffer_position,
                                seed=generated_seed,valid_envs=num_valid_envs,target=[],
                                dynamical_inclusion=dynamical_inclusion,collision=None)
    dataprocessor.plot_control()
    dataprocessor.plot_trajectory()
else:
    print("Input/Output Plots are not generated")
