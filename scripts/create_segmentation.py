from hydra import initialize, compose
from calvin_env.envs.play_table_env import PlayTableSimEnv
import numpy as np
import os
import numpy as np
from tqdm import tqdm

DATASET_PATH = "/mnt/vol1/ramy/calvin/dataset/task_B_D/"
NEW_DATA_PATH = "/mnt/vol1/ramy/calvin/dataset/task_B_D/"
FILTER_SCENE = "calvin_scene_B"

# DATASET_PATH = "/mnt/vol1/ramy/calvin/dataset/task_D_D/"
# NEW_DATA_PATH = "/mnt/vol1/ramy/calvin/dataset/task_D_D_new2/"
# FILTER_SCENE = None

CAMERA = "static" # "static" or "gripper"


def create_env(scene_name="calvin_scene_D"):
    with initialize(config_path="../calvin_env/conf/"):
        cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper", f"scene={scene_name}"])
        cfg.env["use_egl"] = True
        cfg.env["show_gui"] = False
        cfg.env["use_vr"] = False
        cfg.env["use_scene_info"] = True

    env_cfg = {**cfg.env}
    env_cfg["tasks"] = cfg.tasks
    env_cfg.pop('_target_', None)
    env_cfg.pop('_recursive_', None)
    env = ModPlayTableSimEnv(**env_cfg)
    return env, scene_name


class ModPlayTableSimEnv(PlayTableSimEnv):
    def __init__(self,
                 tasks: dict = {},
                 initial_states: list = [],
                 **kwargs):
        super(ModPlayTableSimEnv, self).__init__(**kwargs)
        # self.render_mode="human"
        self.render_mode="rgb_array"

        self.initial_states = initial_states

    def reset(self, robot_obs=None, scene_obs=None, seed=None):
        obs = super().reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self.start_info = self.get_info()
        # try:
        #     num_bodies = self.p.getNumBodies(self.cid)
        #     print(num_bodies)
        #     for i in range(num_bodies):
        #         # Get body info
        #         body_info = self.p.getBodyInfo(i, self.cid)
        #         print(f"Body ID: {i}, Body info: {body_info}")
        #         # Get the number of joints (links) for this body
        #         num_joints = self.p.getNumJoints(i, self.cid)
        #         print(f"Number of joints (links) for body {i}: {num_joints}")
        #         for j in range(num_joints):
        #             # Get joint info
        #             joint_info = self.p.getJointInfo(i, j, self.cid)
        #             print(f"Joint ID: {j}, Joint info: {joint_info}")
        # except Exception as e:
        #     print(f"Error: {e}")
        return obs, {}
    

if __name__ == '__main__':        

    # get the scene configuration for training set
    scene_info_dict = np.load(os.path.join(os.path.join(DATASET_PATH, "training"), "scene_info.npy"), allow_pickle=True).item()
    
    # create another dictionary to map episode number to scene name
    # episode_to_scene = {}
    # for k, v in scene_info_dict.items():
    #     for i in range(v[0], v[1]+1):
    #         episode_to_scene[i] = k
    """ this will be a slow dictionary, so we will use a function instead"""
                
    def get_scene_name(episode_num):
        for k, v in scene_info_dict.items():
            if v[0] <= episode_num <= v[1]:
                return k
        
    print(scene_info_dict)
    
    # loop over each episode step in the dataset, each one is an npz file in the DATASET_PATH folder
    for type in ["training", "validation"]:
        env, current_scene_name = create_env()
        files_list = [f for f in os.listdir(os.path.join(DATASET_PATH, type)) if f.endswith(".npz")]
        # filter only the files that belong to scene B for the training set
        if FILTER_SCENE is not None:
            if type == "training":
                files_list = [f for f in files_list if get_scene_name(int(f.strip('.npz').split("_")[1])) == FILTER_SCENE]
        files_list.sort()
        for episode_step_f in tqdm(files_list):
            # ignore files that have already been processed
            # if os.path.exists(os.path.join(os.path.join(NEW_DATA_PATH, type), episode_step_f)):
            #     continue
            if type == "training":
                required_scene_name = get_scene_name(int(episode_step_f.strip('.npz').split("_")[1]))
                if required_scene_name != current_scene_name:
                    print(f"Switching to scene {required_scene_name}")
                    env, current_scene_name = create_env(required_scene_name)
            elif type == "validation":
                if current_scene_name != "calvin_scene_D":
                    env, current_scene_name = create_env("calvin_scene_D")
            # load the episode step
            episode_step = np.load(os.path.join(os.path.join(DATASET_PATH, type), episode_step_f))
            # print(episode_step_f)
            # get the robot obs
            robot_obs = episode_step['robot_obs']
            # get the scene obs
            scene_obs = episode_step['scene_obs']
            # reset the environment with the robot and scene obs
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

            image = env.render(f"seg_{CAMERA}")
            
            # if env.p.getNumBodies(env.cid) == 8:
            #     seg = image
            #     import numpy as np
            #     import matplotlib.pyplot as plt
            #     import matplotlib.colors as mcolors
            #     import matplotlib.ticker as ticker

            #     # Define the boundaries for each color segment
            #     boundaries = np.arange(-0.5, len(set(seg.flatten())), 1)  # -0.5 and 13.5 are the outer edges; 0-12 are the segment values

            #     # Create a color map and a normalization instance
            #     cmap = plt.get_cmap('tab20', np.max(seg)-np.min(seg)+1)
            #     norm = mcolors.BoundaryNorm(boundaries, cmap.N)

            #     # Display the image
            #     fig, ax = plt.subplots(figsize=(6, 6))
            #     cax = ax.imshow(seg, cmap=cmap, norm=norm, interpolation='nearest')

            #     # Create the color bar
            #     cbar = fig.colorbar(cax, ticks=np.arange(0, len(set(seg.flatten()))), spacing='proportional')
            #     cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure integer ticks
            #     cbar.set_label('Segment Value')
            #     print(set(seg.flatten()))

            #     # save the image
            #     plt.savefig(episode_step_f.strip('.npz') + ".png")
            #     exit()
            
            # Assume 'image' is your segmentation image
            new_image_dict = {f'seg_{CAMERA}': image}        
            np.savez_compressed(os.path.join(os.path.join(NEW_DATA_PATH, type), episode_step_f), **episode_step, **new_image_dict)



