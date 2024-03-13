from hydra import initialize, compose
import hydra
import gymnasium as gym
from gymnasium import spaces
from calvin_env.envs.play_table_env import PlayTableSimEnv
from stable_baselines3 import SAC
from sb3_contrib import TQC
import numpy as np
import wandb
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import os
from calvin_env.utils.utils import set_egl_device, get_egl_device_ids
import torch
from pathlib import Path
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
import matplotlib.pyplot as plt
import numpy as np

import random
import imageio
from tqdm import tqdm

DATASET_PATH = "/mnt/vol1/ramy/calvin/dataset/calvin_debug_dataset_new/"
NEW_DATA_PATH = "/mnt/vol1/ramy/calvin/dataset/calvin_debug_dataset_new2/"

# DATASET_PATH = "/mnt/vol1/ramy/calvin/dataset/task_D_D_new/"
# NEW_DATA_PATH = "/mnt/vol1/ramy/calvin/dataset/task_D_D_new2/"

CAMERA = "gripper" # "static" or "gripper"

with initialize(config_path="../calvin_env/conf/"):
    cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
    cfg.env["use_egl"] = True
    cfg.env["show_gui"] = False
    cfg.env["use_vr"] = False
    cfg.env["use_scene_info"] = True


class ModPlayTableSimEnv(PlayTableSimEnv):
    def __init__(self,
                 tasks: dict = {},
                 initial_states: list = [],
                 **kwargs):
        super(ModPlayTableSimEnv, self).__init__(**kwargs)
        # For this example we will modify the observation to
        # only retrieve the end effector pose
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(7,))
        # self.observation_space = spaces.Dict(
        #     {
        #         'robot_obs': spaces.Box(low=-1, high=1, shape=(7,)),
        #         'static_camera': spaces.Box(low=0, high=255, shape=(200, 200, 3)),
        #         # 'gripper_camera': spaces.Box(low=0, high=255, shape=(84, 84, 3)),
        #     }
        # )
        # We can use the task utility to know if the task was executed correctly
        self.tasks = hydra.utils.instantiate(tasks)
        # self.render_mode="human"
        self.render_mode="rgb_array"

        self.initial_states = initial_states

    def reset(self, robot_obs=None, scene_obs=None, seed=None):
        if robot_obs is None or scene_obs is None:
            robot_obs, scene_obs = get_env_state_for_initial_condition(random.choice(self.initial_states))
        obs = super().reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self.start_info = self.get_info()
        return obs, {}
    

# BODY_IDS = {
#     0: [0] + [0 + (1+i)<<24 for i in range(0, 16)],
#     1: [1],
#     2: [2],
#     3: [3],
#     4: [4],
#     5: [5],
#     6: [5 + (1+0)<<24],
#     7: [5 + (1+1)<<24],
#     8: [5 + (1+2)<<24],
#     9: [5 + (1+3)<<24],
#     10: [5 + (1+4)<<24],
#     11: [5 + (1+5)<<24],
#     12: [5 + (1+6)<<24],
#     13: [6],
# }

# def get_unique_id(pixel):
#     if (pixel >= 0):
#         obUid = pixel & ((1 << 24) - 1)
#         linkIndex = (pixel >> 24) - 1
#         if obUid in [0,1,2,3,4]:
#             return np.uint8(obUid+1)
#         elif obUid == 5:
#             return np.uint8(5 + 1 + linkIndex)
#         elif obUid == 6:
#             return np.uint8(0)
#     else:
#         return np.uint8(0)

# id_to_part = {}
# for part, id in BODY_IDS.items():
#     for i in id:
#         id_to_part[i] = part

# print(id_to_part)

# convert_to_known_bodies = np.vectorize(lambda x: get_unique_id(x))

if __name__ == '__main__':        
    env_cfg = {**cfg.env}
    env_cfg["tasks"] = cfg.tasks
    env_cfg.pop('_target_', None)
    env_cfg.pop('_recursive_', None)

    env = ModPlayTableSimEnv(**env_cfg)

    # loop over each episode step in the dataset, each one is an npz file in the DATASET_PATH folder
    for type in ["training", "validation"]:
        for episode_step_f in tqdm([f for f in os.listdir(os.path.join(DATASET_PATH, type)) if f.endswith(".npz")]):
            # load the episode step
            episode_step = np.load(os.path.join(os.path.join(DATASET_PATH, type), episode_step_f))
            # get the robot obs
            robot_obs = episode_step['robot_obs']
            # get the scene obs
            scene_obs = episode_step['scene_obs']
            # reset the environment with the robot and scene obs
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            # visualize the environment and save the result in a file to view later
            image = env.render(f"seg_{CAMERA}")
            # Assume 'image' is your segmentation image
            # image = convert_to_known_bodies(image)
            new_image_dict = {f'seg_{CAMERA}': image}        
            np.savez_compressed(os.path.join(os.path.join(NEW_DATA_PATH, type), episode_step_f), **episode_step, **new_image_dict)
            # print(f"Finished {episode_step_f}")
            continue
            print(image)
            print(image.dtype)
            print(image.shape)
            
            
            # Get unique values in the image
            unique_values = np.unique(image)

            # Create a colormap with a unique color for each unique value
            colors = plt.cm.get_cmap('tab20b', len(unique_values))

            # Create a dictionary mapping each unique value to a color
            color_dict = {val:colors(i) for i, val in enumerate(unique_values)}

            color_dict_2 = color_dict.copy()
            for v in color_dict_2:
                color_dict_2[v] = tuple([int(x * 255) for x in color_dict_2[v]])
            print(color_dict_2)
            # Create a new image with the colors
            color_image = np.zeros((image.shape[0], image.shape[1], 4))  # Initialize with zeros
            for val, color in color_dict.items():
                color_image[image == val] = color

            # The color image will have an extra dimension for the color channels. 
            # We can remove the alpha channel and convert it back to 8-bit RGB.
            color_image = (color_image[:, :, :3] * 255).astype(np.uint8)        # convert the numpy array to an image
            image = imageio.imwrite(f"episode_{episode_step_f}.png", color_image)




