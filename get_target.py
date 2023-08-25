import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import FS10_V1
from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
from rlbench.backend.utils import task_file_to_task_class
import os
from omegaconf import DictConfig, OmegaConf, ListConfig
from helpers import utils
import pickle

import requests
from PIL import Image,ImageDraw
import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection

def get_oracle_target():
    action_mode = MoveArmThenGripper(
  arm_action_mode=JointVelocity(),
  gripper_action_mode=Discrete()
)
    train_config_path = "/home/liuchang/projects/peract-main/conf/config.yaml"
    eval_config_path = "/home/liuchang/projects/peract-main/conf/eval.yaml"
    with open(train_config_path, 'r') as f:
        train_cfg = OmegaConf.load(f)
    with open(eval_config_path, 'r') as f:
        eval_cfg = OmegaConf.load(f)
    obs_config = utils.create_obs_config(eval_cfg.rlbench.cameras,
                                        eval_cfg.rlbench.camera_resolution,
                                        "PERACT_BC")
# tasks=["close_jar","insert_onto_square_peg","light_bulb_in","meat_off_grill","open_drawer","place_shape_in_shape_sorter","push_buttons","put_groceries_in_cupboard","put_item_in_drawer","put_money_in_safe","reach_and_drag","turn_tap","place_wine_at_rack_location","sweep_to_dustpan_of_size"]
    tasks=["pick_and_lift"]
    task_root = "/home/liuchang/DATA/rlbench_data/ten_tasks_300/"
    # /home/liuchang/DATA/rlbench_data/ten_tasks_300/pick_and_lift/variation0/episodes/episode0
    for task in tasks:
        task_class = task_file_to_task_class(task)
        env_config = (task_class,
                        obs_config,
                        action_mode,
                        task_root,
                        eval_cfg.rlbench.episode_length,
                        eval_cfg.rlbench.headless,
                        train_cfg.rlbench.include_lang_goal_in_obs,
                        eval_cfg.rlbench.time_in_state,
                        eval_cfg.framework.record_every_n,
                        eval_cfg.framework.use_low_lang)
        eval_env = CustomRLBenchEnv(
                    task_class=env_config[0],
                    observation_config=env_config[1],
                    action_mode=env_config[2],
                        dataset_root=env_config[3],
                        episode_length=env_config[4],
                        headless=env_config[5],
                        include_lang_goal_in_obs=env_config[6],
                        time_in_state=env_config[7],
                        record_every_n=env_config[8])
        eval_env.launch()
        for i in range(1):
            step_lang,obs = eval_env.reset_to_demo(i)
            print(step_lang)
            target_objects_information=obs["target_objects_information"]
            print(target_objects_information.keys())
            target_name = obs["target_name"]
        # target_name = "broom_holder"
        # i=0
        # print(target_name)
        # target_name = "drawer_"+step_lang[0].split(" ")[-3]
        # index = 0
        # while target_name not in step_lang[0] and index <len(target_objects_information.keys()):
        #     if len(target_name.split("_")) ==2 and target_name.split("_")[0] in step_lang[0]:
        #         # print(target_name.split("_")[1])
        #         break
        #     target_name = list(target_objects_information.keys())[index]
        #     # print(target_name)
        #     index+=1
        
            episode = "episode"+str(i)
            demo_path = os.path.join(task_root,task,"variation0/episodes",episode,"target.pkl")
        # example_path = os.path.join( "/home/liuchang/projects/peract-main/data/train",task,"all_variations/episodes",episode,"target.pkl")
        # with open(example_path,"rb") as f:
        #     target_name = pickle.load(f)[1]
            print(target_name)    
            with open(demo_path,"wb+") as f:
            # print(target_name,target_objects_information)
                pickle.dump([target_objects_information,target_name],f)
        # with open(demo_path,"rb") as f:
        #     target = pickle.load(f)
        # print(target[1])
        eval_env.shutdown()

from PIL import Image, ImageDraw

def draw_bounding_box(image_path, bbox):
    """
    在图片上绘制bounding box，确保不会超出图片范围

    参数:
        image_path (str): 图片的路径
        bbox (tuple): bounding box的坐标，格式为 (x_min, y_min, x_max, y_max)
    """
    try:
        # 打开图片
        image = Image.open(image_path)

        # 获取图片尺寸
        img_width, img_height = image.size

        # 解析bounding box坐标
        x_min, y_min, x_max, y_max = bbox

        # 创建绘制对象
        draw = ImageDraw.Draw(image)

        # 绘制矩形框
        draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)

        # 展示图片
        # image.show()
        image.save("/home/liuchang/projects/peract-main/img.png")
    
    except Exception as e:
        print("出现错误:", e)

# # 示例用法
# if __name__ == "__main__":
#     image_path = "path_to_your_image.jpg"  # 替换为你的图片路径
#     bounding_box = (50, 50, 200, 200)  # 替换为你的bounding box坐标
#     draw_bounding_box(image_path, bounding_box)

def get_vlm_target():
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    url = "/home/liuchang/projects/peract-main/data/high/open_drawer/all_variations/episodes/episode0/front_rgb/0.png"
    image = Image.open(url)
    
    texts = [["drawer handle"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Print detected objects and rescaled box coordinates
    score_threshold = 0.1
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= score_threshold:
            print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
        draw_bounding_box(url,box)

if __name__ == '__main__':    
   get_oracle_target()

