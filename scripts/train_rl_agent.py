from hydra import initialize, compose
import hydra
import gymnasium as gym
from gymnasium import spaces
from calvin_env.envs.play_table_env import PlayTableSimEnv
from stable_baselines3 import SAC
from sb3_contrib import TQC

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

import random

STORAGE_CAVIN_PATH = "/home/uoft/ramy/calvin"
os.environ['WANDB_DIR'] = str(Path(STORAGE_CAVIN_PATH) / "wandb")

TASK_NAME = "push_into_drawer" # "move_slider_left" # "turn_on_led"

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class SlideEnv(PlayTableSimEnv):
    def __init__(self,
                 tasks: dict = {},
                 initial_states: list = [],
                 **kwargs):
        super(SlideEnv, self).__init__(**kwargs)
        # For this example we will modify the observation to
        # only retrieve the end effector pose
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = spaces.Dict(
            {
                'robot_obs': spaces.Box(low=-1, high=1, shape=(7,)),
                'static_camera': spaces.Box(low=0, high=255, shape=(200, 200, 3)),
                # 'gripper_camera': spaces.Box(low=0, high=255, shape=(84, 84, 3)),
            }
        )
        # We can use the task utility to know if the task was executed correctly
        self.tasks = hydra.utils.instantiate(tasks)
        # self.render_mode="human"
        self.render_mode="rgb_array"

        self.initial_states = initial_states

    def reset(self, seed=None):
        robot_obs, scene_obs = get_env_state_for_initial_condition(random.choice(self.initial_states))
        obs = super().reset(robot_obs=robot_obs, scene_obs=scene_obs)
        self.start_info = self.get_info()
        return obs, {}

    def get_obs(self):
        """Overwrite robot obs to only retrieve end effector position"""
        full_obs = super().get_obs()
        # robot_obs, robot_info = self.robot.get_observation()
        # scene_obs = self.scene.get_obs()
        # print(robot_obs[:7], scene_obs)
        obs = {}
        obs['robot_obs'] = full_obs['robot_obs'][:7]
        obs['static_camera'] = full_obs['rgb_obs']['rgb_static']
        # obs['gripper_camera'] = full_obs['rgb_obs']['rgb_gripper']
        # for k, v in obs.items():
        #     print(v.shape)
        return obs

    def _success(self):
        """ Returns a boolean indicating if the task was performed correctly """
        current_info = self.get_info()
        task_filter = [TASK_NAME]
        task_info = self.tasks.get_task_info_for_set(self.start_info, current_info, task_filter)
        return TASK_NAME in task_info

    def _reward(self):
        """ Returns the reward function that will be used 
        for the RL algorithm """
        reward = int(self._success()) * 10
        r_info = {'reward': reward}
        return reward, r_info

    def _termination(self):
        """ Indicates if the robot has reached a terminal state """
        success = self._success()
        done = success
        d_info = {'success': success}        
        return done, d_info

    def step(self, action):
            """ Performing a relative action in the environment
                input:
                    action: 7 tuple containing
                            Position x, y, z. 
                            Angle in rad x, y, z. 
                            Gripper action
                            each value in range (-1, 1)

                            OR
                            8 tuple containing
                            Relative Joint angles j1 - j7 (in rad)
                            Gripper action
                output:
                    observation, reward, done info
            """
            # Transform gripper action to discrete space
            env_action = action.copy()
            env_action[-1] = (int(action[-1] >= 0) * 2) - 1

            # for using actions in joint space
            if len(env_action) == 8:
                env_action = {"action": env_action, "type": "joint_rel"}

            self.robot.apply_action(env_action)
            for i in range(self.action_repeat):
                self.p.stepSimulation(physicsClientId=self.cid)
            obs = self.get_obs()
            info = self.get_info()
            reward, r_info = self._reward()
            done, d_info = self._termination()
            info.update(r_info)
            info.update(d_info)
            return obs, reward, done, False, info
    

class TruncatedEnv(gym.Wrapper):
    def __init__(self, env, max_steps=50):
        super(TruncatedEnv, self).__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            trunc = True
            info['reason'] = 'truncated'
        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)


class RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        # This callback will call the render method of the environment
        # at each step, rendering the environment.
        self.locals['env'].render(mode='human')
        return True


def train():
    # render_callback = RenderCallback()

    # # Save a checkpoint every 1000 steps
    # checkpoint_callback = CheckpointCallback(
    #   save_freq=1000,
    #   save_path=f"models/{wandb.run.id}",
    #   name_prefix="rl_model",
    #   save_replay_buffer=False,
    #   save_vecnormalize=False,
    # )

    with initialize(config_path="../calvin_env/conf/"):
        cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
        cfg.env["use_egl"] = True
        cfg.env["show_gui"] = False
        cfg.env["use_vr"] = False
        cfg.env["use_scene_info"] = True
        print(cfg.env)

    wandb.init(project="CALVIN_RL",
            entity="uoft",
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game,
            config=flatten_dict(OmegaConf.to_container(cfg, resolve=True))
    )

    # add one more entry to wandb config for the task
    wandb.config.task = TASK_NAME

    wandb_callback = WandbCallback(
        gradient_save_freq=1_000,
        model_save_freq=1_000,
        model_save_path=f"{STORAGE_CAVIN_PATH}/models/{wandb.run.id}",
        verbose=2,
    )

    # get all the egl_ids for each cuda_id if not saved and then save them in a local file
    if not os.path.exists("egl_ids.txt"):
        print("Getting EGL IDs")
        egl_ids = get_egl_device_ids()
        with open("egl_ids.txt", "w") as f:
            for cuda_id, egl_id in egl_ids.items():
                f.write(f"{cuda_id}:{egl_id}\n")
    else:
        print("Loading EGL IDs")
        egl_ids = {}
        with open("egl_ids.txt", "r") as f:
            for line in f:
                cuda_id, egl_id = line.strip().split(":")
                egl_ids[int(cuda_id)] = int(egl_id)
    print(egl_ids)


    # initialize a list of potential initial states for the task
    # use the initial state when the first task in the sequence is equal to the task here
    initial_states = []
    count = 0
    while len(initial_states) < 1 and count < 10:
        proposed_seqs = get_sequences(num_sequences=1000, num_workers=1)
        for proposed_seq in proposed_seqs:
            if proposed_seq[1][0] == TASK_NAME:
                initial_states.append(proposed_seq[0])
        count += 1
    assert len(initial_states) > 0, "No initial states found for the task"

    def make_env(rank):
        def _init():
            # Assign a GPU to each environment
            cuda_id = (rank % num_gpus) +1
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
            os.environ["EGL_VISIBLE_DEVICES"] = str(egl_ids[cuda_id])
            # set_egl_device(torch.device(f'cuda:{cuda_id}'))
            env_cfg = {**cfg.env}
            env_cfg["tasks"] = cfg.tasks
            env_cfg.pop('_target_', None)
            env_cfg.pop('_recursive_', None)
            env_cfg["initial_states"] = initial_states
            env = SlideEnv(**env_cfg)
            env = TruncatedEnv(env, max_steps=250)
            env = Monitor(env)  # record stats such as returns
            return env

        return _init

    # Number of parallel environments
    num_envs = 256 # Number of environments
    num_gpus = 7  # Number of GPUs available, we don't use GPU 0 because it is mostly used by the RL algorithm

    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    # env = VecNormalize(env, norm_obs=True)

    env = VecVideoRecorder(
        env,
        f"{STORAGE_CAVIN_PATH}/videos/{wandb.run.id}",
        record_video_trigger=lambda x: x % 500 == 0,
        video_length=250,
    )


    # model = SAC("MultiInputPolicy", env, verbose=2, buffer_size=256_000, batch_size=1024, tensorboard_log=f"runs/{wandb.run.id}")
    model = TQC("MultiInputPolicy",
                env,
                verbose=2,
                buffer_size=100_000,
                batch_size=1024,
                learning_starts=16_000,
                train_freq=8,
                gradient_steps=8,
                top_quantiles_to_drop_per_net=5,
                use_sde=True,
                policy_kwargs=dict(net_arch=[256, 128]),
                tensorboard_log=f"{STORAGE_CAVIN_PATH}/runs/{wandb.run.id}")
    # model.learn(total_timesteps=1000, log_interval=2, progress_bar=True, callback=[render_callback, checkpoint_callback, wandb_callback])
    model.learn(total_timesteps=10_000_000, progress_bar=True, callback=[wandb_callback])
    # model.save(f"{wandb.run.id}Model")
    # env.save(f"{wandb.run.id}Env")

    wandb.run.finish()

if __name__ == '__main__':
    train()
