# %%

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from genutil import *
from controllers import action, osc, compensate
from randomenvs import envinit
import math
import numpy as np
import torch

import os
from pathlib import Path
import time
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

COULOMB_NOM = np.array(data["COULOMB_NOM"])
ANGDAMP_NOM = float(data["ANGDAMP_NOM"])
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
INCLUDE_SATURATION = args.include_saturation                             

RANDOM_INIT_POS = args.random_initial_positions
RANDOM_INIT_VEL = args.random_initial_velocities        
RANDOM_MASS = float(args.random_masses)
RANDOM_INERTIA = float(args.random_inertias)
RANDOM_COM = float(args.random_coms)
RANDOM_STIFF = float(args.random_stiffness)
RANDOM_DAMP = float(args.random_damping)
RANDOM_COULOMB = float(args.random_coulomb_friction)
RANDOM_ANGDAMP = float(args.random_angular_damping)
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

if not args.seed:
    generated_seed = np.random.randint(0,9999)
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
else:
    generated_seed = args.seed
print("\nGenerated Seed: "+str(generated_seed))
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

num_per_row = int(math.sqrt(NUM_ENVS))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)
pose.r = gymapi.Quat(0, 0, 0, 1)

ienv = envinit(args,gym,sim,env_lower,env_upper,num_per_row,pose,TOTAL_LINKS,TOTAL_JOINTS,
               FIX_BASE_LINK,FLIP_VISUAL_ATTACHMENTS,ARMATURE,DISABLE_GRAVITY,ANGDAMP_NOM,
               MASS_NOM,COM_NOM, INERTIA_NOM, STIFFNESS_NOM, DAMPING_NOM,POS_END, VEL_END,
               COULOMB_NOM)
mass_vector,init_pos_list,init_orn_list,envs,fhandles,hand_idxs,franka_asset,franka_lower,franka_upper = ienv.create_envs()

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
ll = torch.tensor(POS_END[:7,0]).repeat(NUM_ENVS,1) 
ul = torch.tensor(POS_END[:7,1]).repeat(NUM_ENVS,1) 
tl = torch.tensor(TOR_END[:7]).repeat(NUM_ENVS,1).to(device=args.graphics_device_id) 

# %%
if CONTROL_IMPOSED:
    ct = action(num_envs=NUM_ENVS,
                num_iter=MAX_ITER+1, 
                num_joints=TOTAL_JOINTS, 
                num_coords=TOTAL_COORDS,
                frequency=FREQUENCY,
                input_type=INPUT_TYPE,
                mass_vector=mass_vector,
                args=args)
    cdict = ct.getaction()
    #ct.plot(trajectory=cdict["ac"], num_envs=NUM_ENVS, num_dofs=TOTAL_JOINTS)
    #ct.plot(trajectory=cdict["acd"], num_envs=NUM_ENVS, num_dofs=TOTAL_JOINTS)

elif OSC_TASK:
    cosc = osc(num_envs=NUM_ENVS,
                num_iter=MAX_ITER+1, 
                num_joints=TOTAL_JOINTS, 
                num_coords=TOTAL_COORDS,
                frequency=FREQUENCY,
                input_type=INPUT_TYPE,
                mass_vector=mass_vector,
                args=args)
    
if not DISABLE_FRICTION or not DISABLE_GRAVITY:
    comp = compensate(args=args,
                      gravity=GRAVITY,
                      friction_params=FRICTION,
                      num_joints=TOTAL_JOINTS)
    
black_list = []  
out_of_range_quaternion = [] 
saturated_ll_idxs = []
saturated_ul_idxs = []
mesh = gymapi.MESH_VISUAL_AND_COLLISION
color = gymapi.Vec3(.9,.25,.15)

# %%
# ================================ SIMULATIONS STARTS =====================================
if VISUALIZE:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")
    cam_pos = gymapi.Vec3(4, 4, 4)
    cam_target = gymapi.Vec3(-4, -3, -2)
    middle_env = envs[NUM_ENVS // 2 + num_per_row // 2]
    gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)
    condition_window = gym.query_viewer_has_closed(viewer)
else:
    condition_window = 0 

_dof_states = gym.acquire_dof_state_tensor(sim)

itr = 0
ts = time.perf_counter()
print(cdict["bp"][:,:,3:7].shape)
while not condition_window  and itr <= MAX_ITER-1:
    itr += 1

    # --------------------------- Abnormal change in quaternion --------------------------------------------
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_net_contact_force_tensor(sim) 

    pos_cur = rb_states[hand_idxs, :3]
    orn_cur = rb_states[hand_idxs, 3:7]

    if itr > 2:
        increment = abs(cdict["bp"][itr-2,:,3:7]- cdict["bp"][itr-3,:,3:7])
        out_of_range = torch.nonzero(increment > .1)

        if out_of_range.shape[0] != 0:   
            abnormal_idxs = out_of_range.shape[0]
            for j in range(out_of_range.shape[0]):
                out_of_range_quaternion.append(int(out_of_range[j,0].to('cpu').numpy()))
                cdict["bp"][itr-2,out_of_range[j,0],3:7] = cdict["bp"][itr-2,out_of_range[j,0],3:7]*-1
                orn_cur = -1*orn_cur

    if OSC_TASK:
        u, ud = cosc(pos_des,orn_des,pos_cur,dof_vel,orn_cur,init_pos,j_eef,mm,itr) # DOF VEL PROBLEM INIT
        cdict["bt"] = torch.cat((cdict["bt"], pos_des.to("cpu").view( 1,NUM_ENVS, 3)), 0)  #target position
    elif CONTROL_IMPOSED:
        u = cdict["ac"][:,:,itr].unsqueeze(-1).contiguous().to(device=args.graphics_device_id)
        ud = cdict["acd"][:,:,itr].unsqueeze(-1).contiguous().to(device=args.graphics_device_id)
    if CONTROL_IMPOSED_FILE:
        u = cdict["ac"][:,:,itr].unsqueeze(-1).contiguous().to(device=args.graphics_device_id)      
        ud = cdict["acd"][:,:,itr].unsqueeze(-1).contiguous().to(device=args.graphics_device_id)

    if not args.disable_gravity:
        gtorque = comp.gravity(jacobian,mass_vector)
        u = u + gtorque
        cdict["bg"] = torch.cat((cdict["bg"],gtorque),dim=2)
    if not args.disable_friction:
        ftorque = comp.friction(dof_vel)
        u = u + ftorque
        cdict["bf"] = torch.cat((cdict["bf"],ftorque),dim=2)

    # ------------------------------------- APPLICATION OF U --------------------------------------------------   
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(u))

    # ------------------------------------ CONTACT COLLECTION -------------------------------------------------
    _contact_forces = gym.acquire_net_contact_force_tensor(sim) 
    contact_forces = gymtorch.wrap_tensor(_contact_forces)

    body_contact = torch.nonzero(abs(contact_forces)>0.01)

    for j in range(body_contact.shape[0]):
        _body_contact = body_contact[j].to("cpu").numpy()
        env_idx_collision = int(np.ceil(_body_contact[0]/TOTAL_LINKS)-1)
        if  not env_idx_collision in black_list : 
            black_list.append(env_idx_collision)
            env_handle = gym.get_env(sim,env_idx_collision)    
            for k in range(TOTAL_LINKS):
                gym.set_rigid_body_color(env_handle, fhandles[0], k , mesh ,color)         
                
    # -------------------------------------- Step the physics --------------------------------------------------
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    if VISUALIZE:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

    # --------------------------------------- Stacking in the buffers ------------------------------------------
    if OSC_TASK or CONTROL_IMPOSED_FILE:
        uaug, udaug = u, ud
        if DYNAMICAL_INCLUSION == True:
            uaug = torch.cat((uaug,mass_vector),dim=2)
        cdict["bca"] = torch.cat((cdict["bca"], uaug), 2)
        #cdict["bcad"] = torch.cat((cdict["bcad"], udaug), 2) 
    elif CONTROL_IMPOSED:
        uaug, udaug = u.view(1,NUM_ENVS,9)[:,:,:TOTAL_JOINTS], ud.view(1,NUM_ENVS,9)[:,:,:TOTAL_JOINTS]
        if DYNAMICAL_INCLUSION == True:
            uaug = torch.cat((uaug,mass_vector),dim=2)
        cdict["bca"] = torch.cat((cdict["bca"], uaug), 0)
        #cdict["bcad"] = torch.cat((cdict["bcad"], udaug), 0) 

    dof_states = gymtorch.wrap_tensor(_dof_states)
    dof_pos = dof_states[:, 0]
    dof_vel = dof_states[:, 1]

    dof_pos = dof_pos.to("cpu").view(1,NUM_ENVS,9)
    dof_pos = dof_pos[:,:,:7] 

    pos_cur = pos_cur.to("cpu").view(1,NUM_ENVS,3) 
    orn_cur = orn_cur.to("cpu").view(1,NUM_ENVS,4) 

    # -------------------------- INCLUDING dof_pos for 7 - dimension state space ---------------------------

    full_pose = torch.cat((pos_cur,orn_cur,dof_pos),dim = 2).to(device=args.graphics_device_id)      
    cdict["bp"] = torch.cat((cdict["bp"], full_pose), 0)  
        
    # ---------------------------- Saturation check | Position ---------------------------------------------
    if not OSC_TASK:
        saturation_ll = torch.nonzero(abs(dof_pos-ll) < 0.01)
        saturation_ul = torch.nonzero(abs(dof_pos-ul) < 0.01)

        if saturation_ll.shape[0] != 0:
            abnormal_idxs = saturation_ll.shape[0]
            for j in range(saturation_ll.shape[0]):
                saturated_ll_idxs.append(int(saturation_ll[j,1].to('cpu').numpy()))

        if saturation_ul.shape[0] != 0:
            abnormal_idxs = saturation_ul.shape[0]
            for j in range(saturation_ul.shape[0]):
                saturated_ul_idxs.append(int(saturation_ul[j,1].to('cpu').numpy()))

    # ---------------------------- Saturation check | Torque -----------------------------------------------
    if OSC_TASK:
        saturation_torques = torch.nonzero( (tl - abs(cdict["ac"].squeeze(-1)[:,:7]) ) < 1)

        if saturation_torques.shape[0] != 0:
            abnormal_idxs = saturation_torques.shape[0]
            for j in range(saturation_torques.shape[0]):
                saturated_ul_idxs.append(int(saturation_torques[j,0].to('cpu').numpy()))

    
tf = time.perf_counter()
print(f"Time taken for simulation is {tf - ts}")
if VISUALIZE:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)   
# =================================== SIMULATION ENDS ==================================================

if not INCLUDE_SATURATION:   # INCLUDES EVERYTHING NOT JUST SATURATION
    saturation_idxs = list(set(saturated_ul_idxs + saturated_ll_idxs))
    print("\n---- Number of saturated simulations: ",len(saturation_idxs),"/", NUM_ENVS,"----\n" )    
else:
    saturation_idxs = []
    print("Saturated environments are included in the final dataset")

out_of_range_quaternion = list(set(out_of_range_quaternion))
black_list = black_list + out_of_range_quaternion + saturation_idxs 
black_list = list(set(black_list))
failed_percentage = len(black_list)/NUM_ENVS*100

print("---- Number of simulations with abnormal changes in quaternions: ",len(out_of_range_quaternion),"/", NUM_ENVS,"----\n" )  
print("---- Number of the colliding simulations: ",len(black_list),"/", NUM_ENVS,"----\n" ) 
print("\n---- Number of rejectable simulations: ",len(black_list),"/", NUM_ENVS,"----\n" )  
print("Percentage of total rejectable simulations:", round(failed_percentage,2), "%") 

non_valid_envs = len(black_list)
num_valid_envs = NUM_ENVS - non_valid_envs

black_list.sort(reverse=True)

for i in range(non_valid_envs):
    row_exclude = black_list[i]
    if CONTROL_IMPOSED and not CONTROL_IMPOSED_FILE:
        cdict["bca"] = torch.cat((cdict["bca"] [:,:row_exclude,:],
                                                cdict["bca"] [:,row_exclude+1:,:]),1)
    elif OSC_TASK:
        cdict["bca"] = torch.cat((cdict["bca"] [:row_exclude,:,:],
                            cdict["bca"] [row_exclude+1:,:,:]),0)
        cdict["bt"] = torch.cat((cdict["bp"] [:,:row_exclude,:],
                            cdict["bp"] [:,row_exclude+1:,:]),1)
    
        cdict["bp"] = torch.cat((cdict["bp"] [:,:row_exclude,:],
        cdict["bp"] [:,row_exclude+1:,:]),1)

# %%
if not NOSAVE:
    tensormgmt = savedata(args,
                        control_trajectory=cdict["bca"],
                        pose=cdict["bp"],
                        seed=generated_seed,
                        valid_envs=num_valid_envs,
                        target=cdict["bt"],
                        dynamical_inclusion=mass_vector,
                        collision=None)
    tensormgmt.save_tensors()
else:
    print("Input/Output Tensors are not saved")
torch.cuda.empty_cache()
# %%
if not NOPLOT:
    dataprocessor = postprocessor(TOTAL_JOINTS,
                                TOTAL_COORDS,
                                args,
                                control_trajectory=cdict["bca"],
                                pose=cdict["bp"],
                                seed=generated_seed,
                                valid_envs=num_valid_envs,
                                target=cdict["bt"],
                                dynamical_inclusion=mass_vector,
                                collision=None)
    dataprocessor.plot_masses()
    dataprocessor.plot_control()
    dataprocessor.plot_trajectory()
else:
    print("Input/Output Plots are not generated")
