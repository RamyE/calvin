from pathlib import Path
from calvin_agent.evaluation.utils import get_default_model_and_env, get_env_state_for_initial_condition
from calvin_agent.utils.utils import get_last_checkpoint
import hydra
from omegaconf import OmegaConf
from calvin_agent.evaluation.multistep_sequences import get_sequences
import random
import numpy as np

DATASET_PATH = "/mnt/vol1/ramy/calvin/dataset/task_B_D"
TRAIN_FOLDER = "/mnt/vol1/ramy/calvin/runs/2024-04-18/08-21-21"
# TRAIN_FOLDER = "/home/uoft/ramy/calvin/pretrained/D_D_static_rgb_baseline/" # does not support decomp obs which is now part of the repo
NEUTRAL_INIT = False

TASK_CFG = OmegaConf.load("/home/uoft/ramy/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
TASK_ORACLE = hydra.utils.instantiate(TASK_CFG)

CHECKPOINTS = [get_last_checkpoint(Path(TRAIN_FOLDER))]

# let's use a predefined checkpoint only
CHECKPOINT_IDX = -1
CHECKPOINT = CHECKPOINTS[CHECKPOINT_IDX]

EPISODE_LEN = 64

MAX_ATTEMPTS = 100 # max number of episodes to generate trying to hit the number of required rollouts

model, env, _ = get_default_model_and_env(TRAIN_FOLDER, DATASET_PATH, CHECKPOINT)
print(env.cameras[0].width, env.cameras[0].height)

def generate_policy_rollouts(lang_annotation, task, reset_info, result="fail", num_rollouts=1, neutral_init=False):
    # We can use model and env to generate policy rollouts
    # We will keep generating rollouts and checking for success untilwe hit the result criterion (success, fail)

    rollouts = []

    # Loop until we reach the desired number of rollouts
    for i in range(MAX_ATTEMPTS):
        print(f"Attempt {i}")
        observations = []
        # Reset the environment and the model
        # reset_info = episode["state_info"]
        if not neutral_init:
            obs = env.reset(robot_obs=reset_info["robot_obs"], scene_obs=reset_info["scene_obs"])
        else:
            initial_states = []
            # use the initial state when the first task in the sequence is equal to the task here
            iterations = 0
            while len(initial_states) < 1 and iterations < 10:
                proposed_seqs = get_sequences(1000)
                for proposed_seq in proposed_seqs:
                    if proposed_seq[1][0] == task:
                        initial_states.append(proposed_seq[0])
                iterations += 1
            if len(initial_states):
                initial_state = random.choice(initial_states)
                robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
                obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs) 
            else:
                print(f"No initial state found for this task {task}. Using the first initial state in the dataset.")
                obs = env.reset(robot_obs=reset_info["robot_obs"], scene_obs=reset_info["scene_obs"])
          
        model.reset()
        start_info = env.get_info()

        # Initialize the flag for successful rollout
        is_success = False

        # Generate a rollout
        for step in range(EPISODE_LEN):
            # Get the action from the model
            action = model.step(obs, lang_annotation)

            # Apply the action in the environment
            obs, _, _, current_info = env.step(action)
            
            # Store the observation
            img = env.render(mode="rgb_array")
            # img = np.transpose(img, (1, 2, 0))
            observations.append(img)
            print(img.shape)

            current_task_info = TASK_ORACLE.get_task_info_for_set(start_info, current_info, {task})
            if len(current_task_info) > 0:
                is_success = True
                break

        # Check if we reached the desired result criterion
        print(f"Success: {is_success}")
        if (result == "success" and is_success) or (result == "fail" and not is_success):
            rollouts.append(observations)
        if len(rollouts) == num_rollouts:
            break
    
    return rollouts