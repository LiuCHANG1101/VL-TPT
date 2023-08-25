# Adapted from ARM
# Source: https://github.com/stepjam/ARM
# License: https://github.com/stepjam/ARM/LICENSE

import logging
from typing import List

import numpy as np
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
import rlbench.utils as rlbench_utils
from rlbench.demo import Demo
from rlbench.backend.const import *
from yarr.replay_buffer.prioritized_replay_buffer import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer

from helpers import demo_loading_utils, utils
from helpers.preprocess_agent import PreprocessAgent
from helpers.clip.core.clip import tokenize
from agents.peract_bc.perceiver_lang_io_prompt import PerceiverVoxelLangEncoder
from agents.peract_bc.qattention_peract_bc_agent import QAttentionPerActBCAgent
from agents.peract_bc.qattention_stack_agent import QAttentionStackAgent

from amsolver.utils import get_stored_demos as get_vlm_demos
import pickle
from pathlib import Path


import torch
import torch.nn as nn
import multiprocessing as mp
from torch.multiprocessing import Process, Value, Manager
from helpers.clip.core.clip import build_model, load_clip, tokenize
from omegaconf import DictConfig

from os.path import join
REWARD_SCALE = 100.0
LOW_DIM_SIZE = 4


def create_replay(batch_size: int, timesteps: int,
                  prioritisation: bool, task_uniform: bool,
                  save_dir: str, cameras: list,
                  voxel_sizes,
                  image_size=[128, 128],
                  replay_size=3e5,
                  prompt = False):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)
    gripper_pose_size = 7
    target_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement('low_dim_state', (LOW_DIM_SIZE,), np.float32))

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_rgb' % cname, (3, *image_size,), np.float32))
        observation_elements.append(
            ObservationElement('%s_point_cloud' % cname, (3, *image_size),
                               np.float32))  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement('%s_camera_extrinsics' % cname, (4, 4,), np.float32))
        observation_elements.append(
            ObservationElement('%s_camera_intrinsics' % cname, (3, 3,), np.float32))

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    if prompt == True:
        observation_elements.extend([
        ReplayElement('high_lang_emb', (lang_feat_dim,),
                      np.float32),
        ReplayElement('high_token_embs', (max_token_seq_len, lang_emb_dim,),
                      np.float32),
        ReplayElement('target_pose', (target_pose_size,),
                      np.float32),
        ReplayElement('action_type', (),
                      str),
        ]) # extracted from CLIP's language encoder]) ,      
    observation_elements.extend([
        ReplayElement('trans_action_indicies', (trans_indicies_size,),
                      np.int32),
        ReplayElement('rot_grip_action_indicies', (rot_and_grip_indicies_size,),
                      np.int32),
        ReplayElement('ignore_collisions', (ignore_collisions_size,),
                      np.int32),
        ReplayElement('gripper_pose', (gripper_pose_size,),
                      np.float32),
        ReplayElement('lang_goal_emb', (lang_feat_dim,),
                      np.float32),
        ReplayElement('lang_token_embs', (max_token_seq_len, lang_emb_dim,),
                      np.float32), # extracted from CLIP's language encoder
        ReplayElement('task', (),
                      str),
        ReplayElement('lang_goal', (1,),
                      object),  # language goal string for debugging and visualization
    ])

    extra_replay_elements = [
        ReplayElement('demo', (), np.bool),
    ]

    replay_buffer = TaskUniformReplayBuffer(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer


def _get_action(
        obs_tp1: Observation,
        obs_tm1: Observation,
        rlbench_scene_bounds: List[float], # metric 3D bounds of the scene
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution) # 54,36,36
    disc_rot = utils.correct_rotation_instability(disc_rot, rotation_resolution)

    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    try:
        ignore_collisions = int(obs_tm1.ignore_collisions)
    except:
        ignore_collisions = 0
    for depth, vox_size in enumerate(voxel_sizes): # only single voxelization-level is used in PerAct
        if depth > 0:
            if crop_augmentation:
                shift = bounds_offset[depth - 1] * 0.75
                attention_coordinate += np.random.uniform(-shift, shift, size=(3,))
            bounds = np.concatenate([attention_coordinate - bounds_offset[depth - 1],
                                     attention_coordinate + bounds_offset[depth - 1]])
        index = utils.point_to_voxel_index(
            obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return trans_indicies, rot_and_grip_indicies, ignore_collisions, np.concatenate(
        [obs_tp1.gripper_pose, np.array([grip])]), attention_coordinates


def _add_keypoints_to_replay(
        cfg: DictConfig,
        task: str,
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        episode_keypoints: List[int],
        cameras: List[str],
        rlbench_scene_bounds: List[float],
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool,
        description,
        clip_model = None,
        device = 'cpu',
        target_pose=None,
        high_level_description = None):
    copy_desc = description[:]
    prev_action = None
    obs = inital_obs  #obs = demo[i]
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint] #60
        obs_tm1 = demo[max(0, keypoint - 1)] #59
        trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates = _get_action(
            obs_tp1, obs_tm1, rlbench_scene_bounds, voxel_sizes, bounds_offset,
            rotation_resolution, crop_augmentation)

        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0

        obs_dict = utils.extract_obs(obs, t=k, prev_action=prev_action,
                                     cameras=cameras, episode_length=cfg.rlbench.episode_length)
        if not isinstance(copy_desc, str):
            description = copy_desc[k]
        
        tokens = tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)
        obs_dict['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
        obs_dict['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()

        # get_high_level_lang
        if high_level_description is not None:
            tokens = tokenize([high_level_description]).numpy()
            token_tensor = torch.from_numpy(tokens).to(device)
            high_sentence_emb, high_token_embs = clip_model.encode_text_with_embeddings(token_tensor)
            obs_dict['high_lang_emb'] = high_sentence_emb[0].float().detach().cpu().numpy()
            obs_dict['high_token_embs'] = high_token_embs[0].float().detach().cpu().numpy()
            # target_pose
            obs_dict['target_pose'] = target_pose
            # action_type
            action_type = description.split(" ")[0] 
            obs_dict['action_type'] = action_type
        prev_action = np.copy(action)

        others = {'demo': True}
        final_obs = {
            'trans_action_indicies': trans_indicies,
            'rot_grip_action_indicies': rot_grip_indicies,
            'gripper_pose': obs_tp1.gripper_pose,
            'task': task,
            'lang_goal': np.array([description], dtype=object),
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1

    # final step
    obs_dict_tp1 = utils.extract_obs(obs_tp1, t=k + 1, prev_action=prev_action,
                                     cameras=cameras, episode_length=cfg.rlbench.episode_length)
    obs_dict_tp1['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
    obs_dict_tp1['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()
    # final step
    if high_level_description is not None:
        obs_dict_tp1['high_lang_emb'] = high_sentence_emb[0].float().detach().cpu().numpy()
        obs_dict_tp1['high_token_embs'] = high_token_embs[0].float().detach().cpu().numpy()
        obs_dict_tp1['target_pose'] = target_pose
        obs_dict_tp1['action_type'] = "None"
    
    obs_dict_tp1.pop('wrist_world_to_cam', None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)

def _add_keypoints_to_replay_vlm(
        cfg: DictConfig,
        task: str,
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        episode_keypoints: List[int],
        cameras: List[str],
        rlbench_scene_bounds: List[float],
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool,
        target_pose,
        high_level_description,
        description,
        clip_model = None,
        device = 'cpu'):
    copy_desc = description[:]
    copy_desc2 = high_level_description[:]
    prev_action = None
    obs = inital_obs  #obs = demo[i]
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint] #60
        obs_tm1 = demo[max(0, keypoint - 1)] #59
        trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates = _get_action(
            obs_tp1, obs_tm1, rlbench_scene_bounds, voxel_sizes, bounds_offset,
            rotation_resolution, crop_augmentation)

        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0

        obs_dict = utils.extract_obs(obs, t=k, prev_action=prev_action,
                                     cameras=cameras, episode_length=cfg.rlbench.episode_length)
        if not isinstance(description, str) or not isinstance(high_level_description, str):
            description = copy_desc[k]
            high_level_description = copy_desc2[k]
        # get_high_level_lang
        tokens = tokenize([high_level_description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        high_sentence_emb, high_token_embs = clip_model.encode_text_with_embeddings(token_tensor)
        obs_dict['high_lang_emb'] = high_sentence_emb[0].float().detach().cpu().numpy()
        obs_dict['high_token_embs'] = high_token_embs[0].float().detach().cpu().numpy()
        # get_low_level_lang
        if description !=None:
            tokens = tokenize([description]).numpy()
            token_tensor = torch.from_numpy(tokens).to(device)
            sentence_emb, token_embs = clip_model.encode_text_with_embeddings(token_tensor)
            obs_dict['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
            obs_dict['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()
            # target_pose
            obs_dict['target_pose'] = target_pose

        prev_action = np.copy(action)

        others = {'demo': True}
        final_obs = {
            'trans_action_indicies': trans_indicies,
            'rot_grip_action_indicies': rot_grip_indicies,
            'gripper_pose': obs_tp1.gripper_pose,
            'task': task,
            'lang_goal': np.array([description], dtype=object),
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1
    # final step
    obs_dict_tp1 = utils.extract_obs(obs_tp1, t=k + 1, prev_action=prev_action,
                                     cameras=cameras, episode_length=cfg.rlbench.episode_length)
    obs_dict_tp1['lang_goal_emb'] = sentence_emb[0].float().detach().cpu().numpy()
    obs_dict_tp1['lang_token_embs'] = token_embs[0].float().detach().cpu().numpy()
    obs_dict_tp1['high_lang_emb'] = high_sentence_emb[0].float().detach().cpu().numpy()
    obs_dict_tp1['high_token_embs'] = high_token_embs[0].float().detach().cpu().numpy()
    obs_dict_tp1['target_pose'] = target_pose

    obs_dict_tp1.pop('wrist_world_to_cam', None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)


def fill_replay_vlmbench(cfg: DictConfig,
                obs_config: ObservationConfig,
                rank: int,
                replay: ReplayBuffer,
                task: str,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                voxel_sizes: List[int],
                bounds_offset: List[float],
                rotation_resolution: int,
                crop_augmentation: bool,
                clip_model = None,
                device = 'cpu',
                keypoint_method = 'heuristic'):
    logging.getLogger().setLevel(cfg.framework.logging_level)

    if clip_model is None:
        model, _ = load_clip('RN50', jit=False, device=device)
        clip_model = build_model(model.state_dict())
        clip_model.to(device)
        del model

    logging.debug('Filling %s replay ...' % task)
    # for d_idx in range(num_demos):
        # load demo from disk
    data_path = Path(join(cfg.rlbench.demo_path,task))
    num_demos = 0
    for path in data_path.rglob('low_dim_obs*'):
        if num_demos >= cfg.rlbench.demos:
            break    
        num_demos+=1   
        episode_number = str(path.parents[0].name)
        variation = str(path.parents[2].name)
        variation_number = int(variation.replace('variation',''))
        with open(str(path), 'rb') as f:
            demo_temple = pickle.load(f)
        descs = []
        obs_select_inds = []
        previous_waypoint="waypoint0"
        # all_way_points只存了分割点的waypoint
        all_waypoints = [previous_waypoint]
        for i, obs in enumerate(demo_temple._observations):
            if i == 0 :
                descs.append(str(obs.low_level_description))
            if obs.current_waypoint_name == previous_waypoint:
                continue
            else:
                previous_waypoint = obs.current_waypoint_name
                all_waypoints.append(previous_waypoint)
                if cfg.method.use_low_lang:
                    descs.append(str(obs.low_level_description))
                obs_select_inds.append(i)
        high_description=demo_temple.high_level_instructions[0]

        # def get_stored_demos(amount: int, image_paths: bool, dataset_root: str,
        #              variation_number: int, task_name: str,
        #              obs_config: ObservationConfig,
        #              episode_number = None,
        #              fail_demos = False,
        #              selected_frame=None) -> List[Demo]:

        demo = get_vlm_demos(1,False,cfg.rlbench.demo_path,variation_number,task,obs_config,episode_number,False,None)[0]
        obs_0 = demo._observations[0]
        waypoints_info = {name: obj for name, obj in obs_0.object_informations.items() if "waypoint" in name}
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
        target_pose = obs_0.object_informations[target_grasp_obj_name]["pose"]
        # extract keypoints (a.k.a keyframes)
        # episode_keypoints = demo_loading_utils.keypoint_discovery(demo, method=keypoint_method)
        episode_keypoints = obs_select_inds
        if rank == 0:
            logging.info(f"Loading Demo({episode_number}) - found {len(episode_keypoints)} keypoints - {task}")

        key_index = 0
        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0: # choose only every n-th frame
                continue

            obs = demo[i] 
            if not cfg.method.use_low_lang:
                low_desc=descs
            else:
                low_desc = descs[key_index]
            # if our starting point is past one of the keypoints, then remove it
            if len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
                key_index += 1
                continue
            if len(episode_keypoints) == 0:
                break
            _add_keypoints_to_replay_vlm(
                cfg, task, replay, obs, demo, episode_keypoints, cameras,
                rlbench_scene_bounds, voxel_sizes, bounds_offset,
                rotation_resolution, crop_augmentation,target_pose,high_description, description=low_desc,
                clip_model=clip_model, device=device)
            # _add_keypoints_to_replay(
            #     cfg, task, replay, obs, demo, obs_select_inds, cameras,
            #     rlbench_scene_bounds, voxel_sizes, bounds_offset,
            #     rotation_resolution, crop_augmentation, description=descs,
            #     clip_model=clip_model, device=device)
    logging.debug('Replay %s filled with demos.' % task)


def fill_replay(cfg: DictConfig,
                obs_config: ObservationConfig,
                rank: int,
                replay: ReplayBuffer,
                task: str,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                voxel_sizes: List[int],
                bounds_offset: List[float],
                rotation_resolution: int,
                crop_augmentation: bool,
                clip_model = None,
                device = 'cpu',
                keypoint_method = 'heuristic'):
    logging.getLogger().setLevel(cfg.framework.logging_level)

    if clip_model is None:
        model, _ = load_clip('RN50', jit=False, device=device)
        clip_model = build_model(model.state_dict())
        clip_model.to(device)
        del model

    logging.debug('Filling %s replay ...' % task)
    for d_idx in range(num_demos):
        # load demo from disk
        demo = rlbench_utils.get_stored_demos(
            amount=1, image_paths=False,
            dataset_root=cfg.rlbench.demo_path,
            variation_number=-1, task_name=task,
            obs_config=obs_config,
            random_selection=False,
            from_episode_number=d_idx)[0]

        high_level_description = demo._observations[0].misc['descriptions'][0] # 3 descsriptions here
        
        with open (join(cfg.rlbench.demo_path,task,VARIATIONS_ALL_FOLDER,EPISODES_FOLDER,"episode"+str(d_idx),"low_lang.pkl"),"rb") as f:
            try:
                low_lang_list = pickle.load(f)
            except:
                continue
        with open (join(cfg.rlbench.demo_path,task,VARIATIONS_ALL_FOLDER,EPISODES_FOLDER,"episode"+str(d_idx),"target.pkl"),"rb") as f:
            try:
                target_information = pickle.load(f)
                object_pose = target_information[0]
                target_name = target_information[1]
                print(target_name)
                target_pose = object_pose[target_name]["pose"]
            except:
                continue

        # extract keypoints (a.k.a keyframes)
        episode_keypoints = demo_loading_utils.keypoint_discovery(demo, method=keypoint_method)
        if episode_keypoints[0]==1:            
            episode_keypoints = episode_keypoints[1:]

        if rank == 0:
            logging.info(f"Loading Demo({d_idx}) - found {len(episode_keypoints)} keypoints - {task}")


        key_point_index = 0

        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0 or i < 5:
                continue

            obs = demo[i]
            if not cfg.method.prompt:
                high__lang = None
            else:
                high__lang = high_level_description
            # if i == 0:
            #     print(desc)
            # if our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
                low_lang_list = low_lang_list[1:]
                key_point_index += 1
            if len(episode_keypoints) == 0:
                break
            if not cfg.method.use_low_lang:
                desc = high_level_description
            else :
                desc = low_lang_list
            if len(low_lang_list)!=11:
                pass
            _add_keypoints_to_replay(
                cfg, task, replay, obs, demo, episode_keypoints, cameras,
                rlbench_scene_bounds, voxel_sizes, bounds_offset,
                rotation_resolution, crop_augmentation, description=desc,
                clip_model=clip_model, device=device,target_pose=target_pose,high_level_description=high__lang)
    logging.debug('Replay %s filled with demos.' % task)


def fill_multi_task_replay(cfg: DictConfig,
                           obs_config: ObservationConfig,
                           rank: int,
                           replay: ReplayBuffer,
                           tasks: List[str],
                           num_demos: int,
                           demo_augmentation: bool,
                           demo_augmentation_every_n: int,
                           cameras: List[str],
                           rlbench_scene_bounds: List[float],
                           voxel_sizes: List[int],
                           bounds_offset: List[float],
                           rotation_resolution: int,
                           crop_augmentation: bool,
                           clip_model = None,
                           keypoint_method = 'heuristic',
                           dataset="vlmbench"):
    manager = Manager()
    store = manager.dict()

    # create a MP dict for storing indicies
    # TODO(mohit): this shouldn't be initialized here
    del replay._task_idxs
    task_idxs = manager.dict()
    replay._task_idxs = task_idxs
    replay._create_storage(store)
    replay.add_count = Value('i', 0)

    # fill replay buffer in parallel across tasks
    max_parallel_processes = cfg.replay.max_parallel_processes
    processes = []
    n = np.arange(len(tasks))
    split_n = utils.split_list(n, max_parallel_processes)
    for split in split_n:
        for e_idx, task_idx in enumerate(split):
            task = tasks[int(task_idx)]
            model_device = torch.device('cuda:%s' % (e_idx % torch.cuda.device_count())
                                        if torch.cuda.is_available() else 'cpu')
            if dataset == "peract-main":
                p = Process(target=fill_replay, args=(cfg,
                                                    obs_config,
                                                    rank,
                                                    replay,
                                                    task,
                                                    num_demos,
                                                    demo_augmentation,
                                                    demo_augmentation_every_n,
                                                    cameras,
                                                    rlbench_scene_bounds,
                                                    voxel_sizes,
                                                    bounds_offset,
                                                    rotation_resolution,
                                                    crop_augmentation,
                                                    clip_model,
                                                    model_device,
                                                    keypoint_method))
            elif dataset == "vlmbench":
                p = Process(target=fill_replay_vlmbench, args=(cfg,
                                                    obs_config,
                                                    rank,
                                                    replay,
                                                    task,
                                                    num_demos,
                                                    demo_augmentation,
                                                    demo_augmentation_every_n,
                                                    cameras,
                                                    rlbench_scene_bounds,
                                                    voxel_sizes,
                                                    bounds_offset,
                                                    rotation_resolution,
                                                    crop_augmentation,
                                                    clip_model,
                                                    model_device,
                                                    keypoint_method))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


def create_agent(cfg: DictConfig):
    LATENT_SIZE = 64
    depth_0bounds = cfg.rlbench.scene_bounds
    cam_resolution = cfg.rlbench.camera_resolution

    num_rotation_classes = int(360. // cfg.method.rotation_resolution)
    qattention_agents = []
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):
        last = depth == len(cfg.method.voxel_sizes) - 1
        perceiver_encoder = PerceiverVoxelLangEncoder(
            depth=cfg.method.transformer_depth,
            iterations=cfg.method.transformer_iterations,
            voxel_size=vox_size,
            initial_dim = 3 + 3 + 1 + 3,
            low_dim_size=4,
            layer=depth,
            num_rotation_classes=num_rotation_classes if last else 0,
            num_grip_classes=2 if last else 0,
            num_collision_classes=2 if last else 0,
            input_axis=3,
            num_latents = cfg.method.num_latents,
            latent_dim = cfg.method.latent_dim,
            cross_heads = cfg.method.cross_heads,
            latent_heads = cfg.method.latent_heads,
            cross_dim_head = cfg.method.cross_dim_head,
            latent_dim_head = cfg.method.latent_dim_head,
            weight_tie_layers = False,
            activation = cfg.method.activation,
            pos_encoding_with_lang=cfg.method.pos_encoding_with_lang,
            input_dropout=cfg.method.input_dropout,
            attn_dropout=cfg.method.attn_dropout,
            decoder_dropout=cfg.method.decoder_dropout,
            lang_fusion_type=cfg.method.lang_fusion_type,
            voxel_patch_size=cfg.method.voxel_patch_size,
            voxel_patch_stride=cfg.method.voxel_patch_stride,
            no_skip_connection=cfg.method.no_skip_connection,
            no_perceiver=cfg.method.no_perceiver,
            no_language=cfg.method.no_language,
            final_dim=cfg.method.final_dim,
            use_prompt= cfg.method.prompt
        )

        qattention_agent = QAttentionPerActBCAgent(
            layer=depth,
            coordinate_bounds=depth_0bounds,
            perceiver_encoder=perceiver_encoder,
            camera_names=cfg.rlbench.cameras,
            voxel_size=vox_size,
            bounds_offset=cfg.method.bounds_offset[depth - 1],
            image_crop_size=cfg.method.image_crop_size,
            lr=cfg.method.lr,
            training_iterations=cfg.framework.training_iterations,
            lr_scheduler=cfg.method.lr_scheduler,
            num_warmup_steps=cfg.method.num_warmup_steps,
            trans_loss_weight=cfg.method.trans_loss_weight,
            rot_loss_weight=cfg.method.rot_loss_weight,
            grip_loss_weight=cfg.method.grip_loss_weight,
            collision_loss_weight=cfg.method.collision_loss_weight,
            include_low_dim_state=True,
            image_resolution=cam_resolution,
            batch_size=cfg.replay.batch_size,
            voxel_feature_size=3,
            lambda_weight_l2=cfg.method.lambda_weight_l2,
            num_rotation_classes=num_rotation_classes,
            rotation_resolution=cfg.method.rotation_resolution,
            transform_augmentation=cfg.method.transform_augmentation.apply_se3,
            transform_augmentation_xyz=cfg.method.transform_augmentation.aug_xyz,
            transform_augmentation_rpy=cfg.method.transform_augmentation.aug_rpy,
            transform_augmentation_rot_resolution=cfg.method.transform_augmentation.aug_rot_resolution,
            optimizer_type=cfg.method.optimizer,
            num_devices=cfg.ddp.num_devices,
            prompt = cfg.method.prompt
        )
        qattention_agents.append(qattention_agent)

    rotation_agent = QAttentionStackAgent(
        qattention_agents=qattention_agents,
        rotation_resolution=cfg.method.rotation_resolution,
        camera_names=cfg.rlbench.cameras,
    )
    preprocess_agent = PreprocessAgent(
        pose_agent=rotation_agent
    )
    return preprocess_agent
