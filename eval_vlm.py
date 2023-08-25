import argparse
import os
import random
from distutils.util import strtobool
from pathlib import Path
import sys
sys.path.append(r"/home/liuchang/projects/peract-main")
import cv2
import numpy as np
import torch
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from omegaconf import DictConfig, OmegaConf, ListConfig
from agents import peract_bc
from pyrep.objects.vision_sensor import VisionSensor
from pytorch_transformers import BertTokenizer
from scipy.spatial.transform import Rotation as R
from torch.autograd import Variable
# from train_vlmbench import load
from amsolver.action_modes import ActionMode, ArmActionMode
from amsolver.backend.utils import task_file_to_task_class
from amsolver.environment import Environment
from amsolver.observation_config import ObservationConfig
from helpers import custom_rlbench_env
from clip import tokenize
from helpers import demo_loading_utils, utils
# from amsolver.utils import get_stored_demos


# from param import args
# from pyvirtualdisplay import Display
# disp = Display().start()
class Recorder(object):
    def __init__(self) -> None:
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        self.cam = VisionSensor.create([640, 360])
        self.cam.set_pose(cam_placeholder.get_pose())
        self.cam.set_parent(cam_placeholder)
        self._snaps = []
        self._fps=30

    def take_snap(self):
        self._snaps.append(
            (self.cam.capture_rgb() * 255.).astype(np.uint8))
    
    def save(self, path):
        print('Converting to video ...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        video = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*'MJPG'), self._fps,
                tuple(self.cam.get_resolution()))
        for image in self._snaps:
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video.release()
        self._snaps = []
    def del_snap(self):
        self._snaps = []

def load_test_config(data_folder: Path, task_name):
    episode_list = []
    for path in data_folder.rglob('configs*'):
        t_name = path.parents[3].name
        v_name = path.parents[2].name
        if t_name == task_name :
            episode_list.append(path.parent)
    return episode_list

def set_obs_config(img_size):
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
    obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
    obs_config.overhead_camera.render_mode = RenderMode.OPENGL
    obs_config.wrist_camera.render_mode = RenderMode.OPENGL
    obs_config.front_camera.render_mode = RenderMode.OPENGL
    return obs_config

def set_env(args,obs_config):
    need_post_grap = True
    need_pre_move = False
    if args.task == 'drop':
        task_files = ['drop_pen_color', 'drop_pen_relative', 'drop_pen_size']
    elif args.task == 'pick':
        task_files = ['pick_cube_shape', 'pick_cube_relative', 'pick_cube_color', 'pick_cube_size']
        # task_files = ['pick_cube_color']
    elif args.task == 'stack':
        # task_files = ['stack_cubes_color', 'stack_cubes_relative', 'stack_cubes_shape', 'stack_cubes_size']
        task_files = ['stack_cubes_color']
    elif args.task == 'shape_sorter':
        need_pre_move = True
        args.ignore_collision = True
        task_files = ['place_into_shape_sorter_color', 'place_into_shape_sorter_relative', 'place_into_shape_sorter_shape']
    elif args.task == 'wipe':
        args.ignore_collision = True
        task_files = ['wipe_table_shape', 'wipe_table_color', 'wipe_table_relative', 'wipe_table_size', 'wipe_table_direction']
    elif args.task == 'pour':
        task_files = ['pour_demo_color', 'pour_demo_relative', 'pour_demo_size']
    elif args.task == 'drawer':
        args.ignore_collision = True
        args.renew_obs = False
        need_post_grap=False
        task_files = ['open_drawer']
    elif args.task == 'door':
        args.ignore_collision = True
        need_post_grap=False
        task_files = ['open_door']
    elif args.task == 'door_complex':
        args.ignore_collision = True
        need_post_grap=False
        task_files = ['open_door_complex']
    else:
        task_files = [args.task]
    if args.ignore_collision:
        action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME)
    else:
        action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME_WITH_COLLISION_CHECK)
    env = Environment(action_mode, obs_config=obs_config, headless=True) # set headless=False, if user want to visualize the simulator
    return task_files,env

def get_type(x):
    if x.dtype == np.float64:
        return np.float32
    return x.dtype

# def _extract_obs(obs, channels_last: bool, observation_config):
    
#     ROBOT_STATE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
#                             'gripper_open', 'gripper_pose',
#                             'gripper_joint_positions', 'gripper_touch_forces',
#                             'task_low_dim_state', 'misc',"object_informations"]
#     obs_dict = vars(obs)
#     obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
#     robot_state = obs.get_low_dim_data()
#     # Remove all of the individual state elements
#     obs_dict = {k: v for k, v in obs_dict.items() if k not in ROBOT_STATE_KEYS
#                 }
#     if not channels_last:
#         # Swap channels from last dim to 1st dim
#         obs_dict = {k: np.transpose(
#             v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
#                     for k, v in obs_dict.items()}
#     else:
#         # Add extra dim to depth data
#         obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
#                     for k, v in obs_dict.items()}
#     obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)
#     obs_dict['ignore_collisions'] = np.array([1], dtype=np.float32)
#     for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
#         obs_dict[k] = v.astype(np.float32)

#     for config, name in [
#         (observation_config.left_shoulder_camera, 'left_shoulder'),
#         (observation_config.right_shoulder_camera, 'right_shoulder'),
#         (observation_config.front_camera, 'front'),
#         (observation_config.wrist_camera, 'wrist'),
#         (observation_config.overhead_camera, 'overhead')]:
#         if config.point_cloud:
#             obs_dict['%s_camera_extrinsics' % name] = obs.misc['%s_camera_extrinsics' % name]
#             obs_dict['%s_camera_intrinsics' % name] = obs.misc['%s_camera_intrinsics' % name]
#     return obs_dict

# def extract_obs(obs, t=None, episode_length=None,obs_config=None):
#     obs.joint_velocities = None
#     grip_mat = obs.gripper_matrix
#     grip_pose = obs.gripper_pose
#     joint_pos = obs.joint_positions
#     obs.gripper_pose = None
#     # obs.gripper_pose = None
#     obs.gripper_matrix = None
#     obs.wrist_camera_matrix = None
#     obs.joint_positions = None
#     if obs.gripper_joint_positions is not None:
#         obs.gripper_joint_positions = np.clip(
#             obs.gripper_joint_positions, 0., 0.04)

#     obs_dict = _extract_obs(obs,channels_last=False,observation_config=obs_config)

#     # time = (1. - ((t) / float(
#     #     episode_length - 1))) * 2. - 1.
#     # obs_dict['low_dim_state'] = np.concatenate(
#     #     [obs_dict['low_dim_state'], [time]]).astype(np.float32)

#     obs.gripper_matrix = grip_mat
#     # obs.gripper_pose = grip_pose
#     obs.joint_positions = joint_pos
#     obs.gripper_pose = grip_pose
#     # obs_dict['gripper_pose'] = grip_pose
#     return obs_dict

def add_argments():
    parser = argparse.ArgumentParser(description='')
    #dataset
    parser.add_argument('--data_folder', type=str, default="/home/liuchang/DATA/rlbench_data/vlm_test")
    parser.add_argument('--setd', type=str, default="seen")
    parser.add_argument("--load", type=str, default="/home/liuchang/projects/peract-main/logs_vlm/open_drawer/PERACT_BC/seed1/weights/150000", help='path of the trained model')
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--model_name', type=str, default="cliport_6dof")
    parser.add_argument('--maxAction', type=int, default=5, help='Max Action sequence')
    parser.add_argument('--img_size',nargs='+', type=int, default=[128,128])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--language_padding', type=int, default=80)
    parser.add_argument('--need_test_number', type=int, default=25)
    parser.add_argument('--task', type=str, default="open_drawer")
    parser.add_argument('--replay', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--recorder', type=lambda x:bool(strtobool(x)), default=True)
    parser.add_argument('--relative', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--renew_obs', type=lambda x:bool(strtobool(x)), default=True)
    parser.add_argument('--add_low_lang', type=lambda x:bool(strtobool(x)), default=True)
    parser.add_argument('--ignore_collision', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--goal_conditioned', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--wandb_entity', type=str, default=None, help="visualize the test results. Account Name")
    parser.add_argument('--agent', type=str, default="peract_bc", help="test agent")
    parser.add_argument('--wandb_project', type=str, default=None,  help="visualize the test results. Project Name")
    args = parser.parse_args()
    return args

def set_seed(seed, torch=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if torch:
        import torch
        torch.manual_seed(seed)

if __name__=="__main__":
    args = add_argments()
    set_seed(0)
    obs_config = set_obs_config(args.img_size)
    # recorder = Recorder()
    need_test_numbers = args.need_test_number
    task_files,env = set_env(args,obs_config)
    device = "cuda:"+str(args.gpu)
    device = torch.device(device)

    env.launch()

    if args.recorder:
        recorder = Recorder()
    else:
        recorder = None
    
    if args.agent == "peract_bc":
        train_config_path = os.path.join(Path(args.load).parents[1],"config.yaml")
        with open(train_config_path, 'r') as f:
            train_cfg = OmegaConf.load(f)
        agent = peract_bc.launch_utils.create_agent(train_cfg)
        agent.build(training=False, device=device)
        agent.reset()
    # elif args.agent =="hiveformerAgent":
    #     agent = hiveformerAgent(args)
    # elif args.agent =="peractAgent":
    #     agent = peractAgent(args)
    # else:
    #     agent = ReplayAgent()
    # if args.wandb_entity is not None:
    #     import wandb
    train_tasks = [task_file_to_task_class(t, parent_folder = 'vlm') for t in task_files]
    # data_folder = Path(os.path.join(args.data_folder, args.setd))
    data_folder = Path(args.data_folder)



    # if not replay_test:
    #     checkpoint = args.checkpoints
    #     agent = CliportAgent(args.model_name, device_id=args.gpu,z_roll_pitch=True, checkpoint=checkpoint, args=args)
    # else:
    #     agent = ReplayAgent()


    checkpoint = args.load.split("/")[-2] +"_"+ args.load.split("/")[-1]
    output_file_name = f"/home/liuchang/projects/peract-main/logs_vlm/results/{checkpoint}"
    output_file_name += ".txt"
    file = open(output_file_name, "a")
    for i, task_to_train in enumerate(train_tasks):
        e_path = load_test_config(data_folder, task_files[i])
        success_times,grasp_success_times,all_time = 0,0,0
        task = env.get_task(task_to_train)
        # move = Mover(task, max_tries=10)
        for num, e in enumerate(e_path):
            # if num > args.need_test_number:
            #     break
            if num % 4 != 0:
                continue
            task_base = str(e/"task_base.ttm")
            waypoint_sets = str(e/"waypoint_sets.ttm")
            config = str(e/"configs.pkl")
            descriptions, obs = task.load_config(task_base, waypoint_sets, config)
            waypoints_info = {name: obj for name, obj in obs.object_informations.items() if "waypoint" in name}
            all_time+=1
            high_descriptions = descriptions[0] 
            low_descriptions = []
            step=0
            if args.add_low_lang:
                for waypoint in waypoints_info:
                    if "low_level_descriptions" in waypoints_info[waypoint]:
                        low_descriptions.append( waypoints_info[waypoint]["low_level_descriptions"])
            print(high_descriptions)
            target_grasp_obj_name = None
            try:
                if len(waypoints_info['waypoint1']['target_obj_name'])!=0:
                    target_grasp_obj_name = waypoints_info['waypoint1']['target_obj_name']
                    grasp_pose = waypoints_info['waypoint1']['pose'][0]
                else:
                    grasp_pose = waypoints_info['waypoint1']['pose'][0]
                    target_name = None
                    distance = np.inf
                    for g_obj in task._task.get_graspable_objects():
                        obj_name = g_obj.get_name()
                        obj_pos = g_obj.get_position()
                        c_distance = np.linalg.norm(obj_pos-grasp_pose[:3])
                        if c_distance < distance:
                            target_name = obj_name
                            distance = c_distance
                    if distance < 0.2:
                        target_grasp_obj_name = target_name
            except:
                print(f"need re-generate: {e}")
                continue
            target_pose = torch.from_numpy(obs.object_informations[target_grasp_obj_name]["pose"])[:3]
            offset = train_cfg.method.bounds_offset[0]
            target_bounds=torch.cat([target_pose - offset, target_pose + offset], dim=0).cuda()
            step_tokens = []
            high_lang_tokens = np.array(tokenize(high_descriptions)[0].numpy())
            if args.add_low_lang:
                for lang in low_descriptions:
                    lang_token = tokenize(lang)[0].numpy()
                    step_tokens.append([np.array(lang_token, dtype=get_type(lang_token))])
            reward = 0
            for i in range(10):
                obs_dict = utils.extract_obs(obs, t=i if reward!=0 else 0, prev_action=None,
                                     cameras=train_cfg.rlbench.cameras, episode_length=train_cfg.rlbench.episode_length)
                obs_dict['high_lang_tokens'] = high_lang_tokens
                obs_history = {k: [np.array(v, dtype=get_type(v))] * 1 for k, v in obs_dict.items()} #构造出的obs字典，包括lang_tokens
                if train_cfg.method.use_low_lang: 
                    obs_history["lang_goal_tokens"] = step_tokens[i%len(low_descriptions)]
                prepped_data = {k:torch.tensor([v], device=device) for k, v in obs_history.items()}

                act_result = agent.act(-1, prepped_data,
                                deterministic=eval,target_bounds=target_bounds,use_prompt=True)
                action = act_result.action[:-1]
                try:
                    obs, reward, terminate = task.step(action, None , recorder = recorder, need_grasp_obj = target_grasp_obj_name)
                except:
                    reward = 0 
                    pass
                # rewards.append(reward)
                if reward == 0.5:
                    grasped = True
                    grasp_success_times+=1
                elif reward == 1:
                    success_times+=1
                    successed = True
                    break
                else:
                    grasped = False
            if reward == 1 or grasped == True:
            # if i < 10:
                recorder.save(f"./records_{args.agent}/{task.get_name()}/{checkpoint}_{num+1}.avi")
            recorder.del_snap()
            print(f"{task.get_name()}: success {success_times} times in {all_time} steps! success rate {round(success_times/all_time * 100, 2)}%!")
            print(f"{task.get_name()}: grasp success {grasp_success_times} times in {all_time} steps! grasp success rate {round(grasp_success_times/all_time * 100, 2)}%!")
            file.write(f"{task.get_name()}:grasp success: {grasp_success_times}, success: {success_times}, toal {all_time} steps, success rate: {round(success_times/all_time * 100, 2)}%!\n\n")   
            file.flush()
    file.close()
    env.shutdown()