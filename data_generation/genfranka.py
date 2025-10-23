from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from genutil import *
from controllers import action, osc, compensate, PID, cic, joint_PID
from randomenvs import envinit
import math
import numpy as np
import torch

from pathlib import Path
import time

torch.cuda.empty_cache()
args = parser(description="FrankaDataGen",params=[]).parse_arguments()
data = reader(path='frankadata.json' if args.type_of_robot=='franka' else 'kukadata.json').read_data()

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

CPOS = data["CPOS"]
CORN = data["CORN"]
CVEL = data["CVEL"]
CWEL = data["CWEL"]
CACC = data["CACC"]
CALFA = data["CALFA"]

COULOMB_NOM = np.array(data["COULOMB_NOM"])
ANGDAMP_NOM = float(data["ANGDAMP_NOM"])
MASS_NOM = np.array(data["MASS_NOM"])
COM_NOM = np.array(data["COM_NOM"])
INERTIA_NOM = np.array(data["INERTIA_NOM"])
STIFFNESS_NOM = np.array(data["STIFFNESS_NOM"])
DAMPING_NOM = np.array(data["DAMPING_NOM"])
OSC_NOM = np.array(data["OSC_NOM"])

SOLVER_TIME = data["SOLVER_TIME"]/1000
SUBSTEPS = data["SUBSTEPS"]
SOLVER_TYPE = data["SOLVER_TYPE"]
NUM_POS_ITER = data["NUM_POS_ITER"]
NUM_VEL_ITER = data["NUM_VEL_ITER"]

FIX_BASE_LINK = True if int(data["FIX_BASE_LINK"])==1 else False
FLIP_VISUAL_ATTACHMENTS = True if int(data["FLIP_VISUAL_ATTACHMENTS"])==1 else False
ARMATURE = data["ARMATURE"]

ASSETTYPE = args.type_of_robot
DISABLE_GRAVITY = args.disable_gravity     
DISABLE_FRICTION = args.disable_friction          
NO_CONTROLLER = args.no_controller
CONTROLLER = args.controller                                         
OSC_TASK = args.type_of_controller=='osc'
PD_TASK = args.type_of_controller=='pd'
JOINT_PD_TASK = args.type_of_controller=='joint_pd'
PID_TASK = args.type_of_controller=='pid'
JOINT_PID_TASK = args.type_of_controller=='joint_pid'
CIC_TASK = args.type_of_controller=='cic'                                                      
INCLUDE_SATURATION = args.include_saturation 
FIX_QUARTERNIONS = args.fix_quarternions
ORIENTATION_DIMENSION = args.orientation_dimension                                                       

VISUALIZE = args.visualize  
MEASURE = args.measure_force
HOLDFG = args.measure_gravity_friction                          
NOSAVE = args.no_save
NOPLOT = args.no_plot
TYPE_OF_DATASET = args.type_of_dataset
NAME_OF_DATASET = args.name_of_dataset                          
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
SEED = args.seed

complementary_dataset = 'train' if TYPE_OF_DATASET == 'test' else 'test'

output_folder = Path("./data_tensors/")
output_folder.mkdir(exist_ok=True)

current_folder = Path(f"./data_tensors/{TYPE_OF_DATASET}/{NAME_OF_DATASET}")
current_folder.mkdir(exist_ok=True)

if not SEED:
    generated_seed = np.random.randint(0,1e9)
else:
    generated_seed = SEED
print("\nGenerated Seed: "+str(generated_seed))
torch.manual_seed(generated_seed) 

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

ienv = envinit(args,
               gym, sim,
               env_lower, env_upper, num_per_row, pose,
               TOTAL_LINKS, TOTAL_JOINTS,
               FIX_BASE_LINK, FLIP_VISUAL_ATTACHMENTS, ARMATURE, DISABLE_GRAVITY, 
               ANGDAMP_NOM, MASS_NOM, COM_NOM, INERTIA_NOM, STIFFNESS_NOM, DAMPING_NOM, COULOMB_NOM,
               POS_END, VEL_END
               )
envdict = ienv.create_envs()

gym.prepare_sim(sim)

init_pos = torch.Tensor(envdict["ipos"]).view(NUM_ENVS, 3).to(device=GRAPHICS_DECIDE_ID)
init_orn = torch.Tensor(envdict["iorn"]).view(NUM_ENVS, 4).to(device=GRAPHICS_DECIDE_ID)

if USE_GPU_PIPELINE:
    init_pos = init_pos.to('cuda:0')
    init_orn = init_orn.to('cuda:0')

pos_des = init_pos.clone()
orn_des = init_orn.clone()

_jacobian = gym.acquire_jacobian_tensor(sim, ASSETTYPE)
jacobian = gymtorch.wrap_tensor(_jacobian)

hand_index = gym.get_asset_rigid_body_dict(envdict["fass"])["panda_hand" if ASSETTYPE=='franka' else "iiwa7_link_7"]
j_eef = jacobian[:, hand_index - 1, :]

_massmatrix = gym.acquire_mass_matrix_tensor(sim, ASSETTYPE)
mm = gymtorch.wrap_tensor(_massmatrix)

_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

if MEASURE:
    _simtorques = gym.acquire_dof_force_tensor(sim)
    simtorques = gymtorch.wrap_tensor(_simtorques)

_contact_forces = gym.acquire_net_contact_force_tensor(sim) 
contact_forces = gymtorch.wrap_tensor(_contact_forces)

_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_vel = dof_states[:, 1].view(NUM_ENVS, TOTAL_JOINTS, 1)
dof_pos = dof_states[:, 0].view(NUM_ENVS, TOTAL_JOINTS, 1)
ll = torch.tensor(POS_END[:7,0]).repeat(NUM_ENVS,1).to(device=GRAPHICS_DECIDE_ID)  
ul = torch.tensor(POS_END[:7,1]).repeat(NUM_ENVS,1).to(device=GRAPHICS_DECIDE_ID)  
tl = torch.tensor(TOR_END[:7]).repeat(NUM_ENVS,1).to(device=GRAPHICS_DECIDE_ID)
print(POS_END)
if NO_CONTROLLER:
    ctrl = action(num_envs=NUM_ENVS,
                num_iter=MAX_ITER+1, 
                num_joints=TOTAL_JOINTS, 
                num_coords=TOTAL_COORDS,
                frequency=FREQUENCY,
                rigidbodyprops=envdict['rbp'],
                args=args,
                poslim=POS_END, vellim=VEL_END, acclim=ACC_END)
    cdict = ctrl.getcontrol()

if CONTROLLER:
    if JOINT_PD_TASK or JOINT_PID_TASK:
        ctrl = joint_PID(num_envs=NUM_ENVS,
                    num_iter=MAX_ITER+1, 
                    num_joints=TOTAL_JOINTS, 
                    num_coords=TOTAL_COORDS,
                    frequency=FREQUENCY,
                    rigidbodyprops=envdict['rbp'],
                    args=args,
                    poslim=POS_END, vellim=VEL_END, acclim=ACC_END)
        cdict = ctrl.getcontrol()
    elif PD_TASK or PID_TASK:
        ctrl = PID(num_envs=NUM_ENVS,
                    num_iter=MAX_ITER+1, 
                    num_joints=TOTAL_JOINTS, 
                    num_coords=TOTAL_COORDS,
                    frequency=FREQUENCY,
                    rigidbodyprops=envdict['rbp'], 
                    args=args,
                    cposlim=CPOS, cvellim=CVEL, cacclim=CACC,
                    cornlim=CORN, wellim=CWEL,alfalim=CALFA)
        cdict = ctrl.getcontrol()

    if CIC_TASK:
        ctrl = cic(num_envs=NUM_ENVS,
                    num_iter=MAX_ITER+1, 
                    num_joints=TOTAL_JOINTS, 
                    num_coords=TOTAL_COORDS,
                    frequency=FREQUENCY,
                    rigidbodyprops=envdict['rbp'],
                    args=args,
                    cposlim=CPOS, cvellim=CVEL, cacclim=CACC,
                    cornlim=CORN, wellim=CWEL,alfalim=CALFA)
        cdict = ctrl.getcontrol()

    if OSC_TASK:
        ctrl = osc(num_envs=NUM_ENVS,
                    num_iter=MAX_ITER+1, 
                    num_joints=TOTAL_JOINTS, 
                    num_coords=TOTAL_COORDS,
                    frequency=FREQUENCY,
                    rigidbodyprops=envdict['rbp'],
                    args=args,
                    cposlim=CPOS, cvellim=CVEL, cacclim=CACC,
                    cornlim=CORN, wellim=CWEL,alfalim=CALFA)
        cdict = ctrl.getcontrol()

    Kc = ctrl.getcontrollergains()

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

if VISUALIZE:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")
    cam_pos = gymapi.Vec3(4, 4, 4)
    cam_target = gymapi.Vec3(-4, -3, -2)
    middle_env = envdict["envs"][NUM_ENVS // 2 + num_per_row // 2]
    gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)
    condition_window = gym.query_viewer_has_closed(viewer)
else:
    condition_window = 0 

itr = 0
ts = time.perf_counter()

while not condition_window  and itr <= MAX_ITER-1:
    itr += 1

    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    if MEASURE:
        gym.refresh_dof_force_tensor(sim)
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_net_contact_force_tensor(sim) 

    pos_cur = rb_states[envdict["hidx"], :3]
    orn_cur = rb_states[envdict["hidx"], 3:7]

    if OSC_TASK:
        track = ctrl.obtain_trajectory(pos_des, orn_des, init_pos, init_orn, itr)
        u = ctrl.step_osc(pos_cur, orn_cur, dof_vel, j_eef, mm)
        cdict["bt"] = torch.cat((cdict["bt"], track.view(1,NUM_ENVS, 7)), 0)
    elif PD_TASK:
        track = ctrl.obtaintrajectory(pos_des, orn_des, init_pos, init_orn, itr)
        u = ctrl.step_PD(pos_cur, orn_cur, j_eef, TOR_END)
        cdict["bt"] = torch.cat((cdict["bt"], track.view(1,NUM_ENVS, 7)), 0)
    elif JOINT_PD_TASK:
        track = ctrl.obtainjointvars(itr)
        u = ctrl.step_PD(dof_states[:, 0].view(NUM_ENVS,TOTAL_JOINTS,1), TOR_END) 
        cdict["bt"] = torch.cat((cdict["bt"], track.view(1,NUM_ENVS, TOTAL_JOINTS)), 0)
    elif PID_TASK:
        track = ctrl.obtaintrajectory(pos_des, orn_des, init_pos, init_orn, itr)
        u = ctrl.step_PID(pos_cur, orn_cur, j_eef, TOR_END)
        cdict["bt"] = torch.cat((cdict["bt"], track.view(1,NUM_ENVS, 7)), 0)
    elif JOINT_PID_TASK:
        track = ctrl.obtainjointvars(itr)
        u = ctrl.step_PID(dof_states[:, 0].view(NUM_ENVS,TOTAL_JOINTS,1), TOR_END)
        cdict["bt"] = torch.cat((cdict["bt"], track.view(1,NUM_ENVS, TOTAL_JOINTS)), 0)
    elif CIC_TASK:
        track = ctrl.obtaintrajectory(pos_des, orn_des, init_pos, init_orn, itr)
        u = ctrl.step_cic(pos_cur, orn_cur, j_eef, TOR_END, dof_vel)
        cdict["bt"] = torch.cat((cdict["bt"], track.view(1,NUM_ENVS, 7)), 0)
    elif NO_CONTROLLER:
        u = cdict["ac"][:,:,itr].unsqueeze(-1)

    if not DISABLE_GRAVITY:
        gtorque = comp.gravity(jacobian,envdict["rbp"])
        u = u + gtorque
    if not DISABLE_FRICTION:
        ftorque = comp.friction(dof_vel)
        u = u + ftorque
    if HOLDFG:
        cdict["bg"] = torch.cat((cdict["bg"],gtorque),dim=2)
        cdict["bf"] = torch.cat((cdict["bf"],ftorque),dim=2)
    # -------------------------------------- Application of u ---------------------------------------------
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(u.contiguous()))        
                
    # -------------------------------------- Step the physics ---------------------------------------------
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    if VISUALIZE:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

    # --------------------------------------- Buffer Stack ------------------------------------------------
    if CONTROLLER:
        uaug = u.view(1,NUM_ENVS,TOTAL_JOINTS)[:,:,:TOTAL_JOINTS]
        cdict["bca"] = torch.cat((cdict["bca"], uaug), 0)
    elif NO_CONTROLLER:
        uaug = u.view(1,NUM_ENVS,TOTAL_JOINTS)[:,:,:TOTAL_JOINTS]
        cdict["bca"] = torch.cat((cdict["bca"], uaug), 0)

    dof_states = gymtorch.wrap_tensor(_dof_states) # Remove?
    dof_pos = dof_states[:, 0]
    dof_vel = dof_states[:, 1]
    
    dof_pos = dof_pos.view(1,NUM_ENVS,TOTAL_JOINTS)
    dof_pos = dof_pos[:,:,:7]
    dof_vel = dof_vel.view(1,NUM_ENVS,TOTAL_JOINTS)

    pos_cur = pos_cur.view(1,NUM_ENVS,3)
    orn_cur = decide_orientation(orn_cur.view(1,NUM_ENVS,4),ORIENTATION_DIMENSION)

    # -------------------------- Including dof_pos for 7 - dimension state space ---------------------------
    full_pose = torch.cat((pos_cur,orn_cur,dof_pos),dim = 2)
    cdict["bp"] = torch.cat((cdict["bp"], full_pose), 0)
    #cdict["bv"] = torch.cat((cdict["bv"], dof_vel), 0)        

    # ------------------------------------- Contact Collection ---------------------------------------------
    body_contact = torch.nonzero(abs(contact_forces)>0.01)

    for j in range(body_contact.shape[0]):
        _body_contact = body_contact[j]
        env_idx_collision = torch.ceil(_body_contact[0]/TOTAL_LINKS)-1
        if  not env_idx_collision in black_list : 
            black_list.append(env_idx_collision)
            env_handlec = gym.get_env(sim,env_idx_collision)
            if VISUALIZE:    
                for k in range(TOTAL_LINKS):
                    gym.set_rigid_body_color(env_handlec, envdict["hdls"][0], k , mesh ,color) 

    # ---------------------------------- Abnormal change in quaternion -------------------------------------
    if itr > 2 and ORIENTATION_DIMENSION=='4D':
        increment = abs(cdict["bp"][itr-1,:,3:7] - cdict["bp"][itr-2,:,3:7])
        out_of_range = torch.nonzero(increment > .1)
        if out_of_range.numel()!=0: 
            for x in out_of_range:
                out_of_range_quaternion.append(x[0])
                if FIX_QUARTERNIONS:
                    cdict["bp"][itr-1,x[0],x[1]+3] = cdict["bp"][itr-1,x[0],x[1]+3]*-1
                    rb_states[envdict["hidx"][x[0]], x[1]+3] = -1*rb_states[envdict["hidx"][x[0]], x[1]+3]

    # ---------------------------------- Saturation check | Position ---------------------------------------
    if not INCLUDE_SATURATION:
        saturation_ll = torch.nonzero(abs(dof_pos-ll) < 0.01)
        saturation_ul = torch.nonzero(abs(dof_pos-ul) < 0.01)

        if saturation_ll.shape[0] != 0:
            for j in range(saturation_ll.shape[0]):
                saturated_ll_idxs.append(saturation_ll[j,1])

        if saturation_ul.shape[0] != 0:
            for j in range(saturation_ul.shape[0]):
                saturated_ul_idxs.append(saturation_ul[j,1])

    # ---------------------------------- Saturation check | Torque -----------------------------------------
    if not INCLUDE_SATURATION:
        saturation_torques = torch.nonzero( (tl - abs(u.squeeze(-1)[:,:7]) ) < 1)
        if saturation_torques.shape[0] != 0:
            for j in range(saturation_torques.shape[0]):
                saturated_ul_idxs.append(saturation_torques[j,0])
                env_handlet = gym.get_env(sim,saturated_ul_idxs[j])
                if VISUALIZE:    
                    for k in range(TOTAL_LINKS):
                        gym.set_rigid_body_color(env_handlet, envdict["hdls"][0], k , mesh ,color) 

    # -------------------------------- Sensor Measurement - Ground Truth -----------------------------------
    if MEASURE:
        simtorques = gymtorch.wrap_tensor(_simtorques).view(1,NUM_ENVS,TOTAL_JOINTS)
        cdict["mt"] = torch.cat((cdict["mt"],simtorques),0)


tf = time.perf_counter()
dt = tf-ts
print(f"Time taken for simulation is {dt}")

if VISUALIZE:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

torch.cuda.empty_cache()

if not INCLUDE_SATURATION:
    saturation_idxs = list(set([int(su.to("cpu").numpy()) for su in saturated_ul_idxs] +
                                [int(sl.to("cpu").numpy()) for sl in saturated_ll_idxs]))
    print("\n---- Number of saturated simulations: ",len(saturation_idxs),"/", NUM_ENVS,"----\n" )    
else:
    saturation_idxs = []
    print("Saturated environments are included in the final dataset\n")

if not FIX_QUARTERNIONS and not ORIENTATION_DIMENSION=='6D' and not ORIENTATION_DIMENSION=='3D':
    out_of_range_quaternion = list(set([int(q.to("cpu").numpy()) for q in out_of_range_quaternion]))
    print("---- Number of simulations with abnormal changes in quaternions: ",len(out_of_range_quaternion),"/", NUM_ENVS,"----\n" )
else:
    out_of_range_quaternion = []
    print("Quarternion error is compensated\n")

print("---- Number of the colliding simulations: ",len(black_list),"/", NUM_ENVS,"----\n" ) 

black_list = list(set([int(b.to("cpu").numpy()) for b in black_list] + out_of_range_quaternion + saturation_idxs))
white_list = set([i for i in range(NUM_ENVS)]) - set(black_list)
failed_percentage = len(black_list)/NUM_ENVS*100
print("\n---- Number of rejectable simulations: ",len(black_list),"/", NUM_ENVS,"----\n" )  
print("Percentage of total rejectable simulations:", round(failed_percentage,2), "%") 

non_valid_envs = len(black_list)
num_valid_envs = NUM_ENVS - non_valid_envs

ll = ll[list(white_list),:]
ul = ul[list(white_list),:]
if CONTROLLER:
    for k,v in Kc.items():
        Kc[k] = v[list(white_list),...]
    cdict["bca"] = cdict["bca"][:,list(white_list),:]
    cdict["bp"] = cdict["bp"][:,list(white_list),:]
    cdict["bt"] = cdict["bt"][:,list(white_list),:]
if MEASURE:
    cdict["mt"] = cdict["mt"][:,list(white_list),:]
    cdiff = cdict["bca"][:,:,:TOTAL_JOINTS] + cdict["mt"]
    #cdict["bca"] = -cdict["mt"]   # FOR MG6    

if not NOSAVE:
    tensormgmt = savedata(args,
                        control_trajectory=cdict["bca"],
                        pose=cdict["bp"],
                        seed=generated_seed,
                        valid_envs=num_valid_envs,
                        target=cdict["bt"],
                        rigidbodyprops=envdict["rbp"],
                        collision=None,
                        gentime=dt,
                        path='.',
                        controlgains=Kc if args.controller else None
                        )
    tensormgmt.save_tensors()
    tensormgmt.save_metadata()
else:
    print("Input/Output Tensors are not saved")

if not NOPLOT:
    dataprocessor = postprocessor(TOTAL_JOINTS,
                                TOTAL_COORDS,
                                args,
                                control_trajectory=cdict["bca"],
                                pose=cdict["bp"],
                                seed=generated_seed,
                                valid_envs=num_valid_envs,
                                target=cdict["bt"],
                                rigidbodyprops=envdict["rbp"],
                                controlgains=Kc if args.controller else None
                                )
    
    if MEASURE:
        dataprocessor.plot_secondary_var(var=torch.permute(-1*cdict["mt"],(1,2,0)),varname="ground_truth_control")
        dataprocessor.plot_secondary_var(var=torch.permute(cdiff,(1,2,0)),varname="benchmark_control_error")
    if HOLDFG:
        dataprocessor.plot_secondary_var(var=cdict["bg"],varname="gravity")
        dataprocessor.plot_secondary_var(var=cdict["bf"],varname="friction")
    if INCLUDE_SATURATION:
        dataprocessor.plot_saturation_histogram(pose=cdict["bp"],limits=[ll,ul])
        dataprocessor.plot_collision_histogram(pose=cdict["bp"])
    dataprocessor.plot_control()
    dataprocessor.plot_trajectory()
    if CONTROLLER:
        dataprocessor.plot_trajectory_wrt_control_gains()

    #dataprocessor.plot_secondary_var(var=torch.permute(cdict["bp"],(1,2,0))[:,:,3:9],varname="dof velocity")
else:
    print("Input/Output Plots are not generated")
