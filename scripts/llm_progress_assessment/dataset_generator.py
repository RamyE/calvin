from pathlib import Path
import numpy as np
from PIL import Image
from generate_policy_rollouts import generate_policy_rollouts
from hydra import initialize, compose
from calvin_env.envs.play_table_env import PlayTableSimEnv
from hydra.core.global_hydra import GlobalHydra
import json
import argparse

FEASIBILITY_INFEASIBLE_GT_PATH = Path('sample_dataset/feasibility_infeasible_gt.json')

ORIGINAL_DATASET_PATH = Path('/mnt/vol1/ramy/calvin/dataset/task_B_D/')
GENERATED_DATASET_PATH = Path('sample_dataset')
parser = argparse.ArgumentParser(description='Dataset Generator')
parser.add_argument('dataset_type', choices=['success', 'progress', 'feasibility'], help='Dataset type')
parser.add_argument('max_dataset_size', type=int, help='Maximum dataset size')
parser.add_argument('--dry_run', action='store_true', help='Dry run mode')
parser.add_argument('--feasibility_type', choices=['proposal', 'groundtruth'], default='groundtruth', help='Feasibility type')
parser.add_argument('--gt_types', nargs='+', choices=['success', 'wrong', 'fail'], default=['success', 'wrong'], help='Ground truth types')
# parse arg for stack type
parser.add_argument('--stack_type', choices=['grid', 'sequential', 'video'], default='sequential', help='Stack type')
# parse args for max width and height
parser.add_argument('--max_width', type=int, default=500, help='Maximum width of the image')
parser.add_argument('--max_height', type=int, default=500, help='Maximum height of the image')
# parse args for skip frames
parser.add_argument('--skip_frames', type=int, default=6, help='Number of frames to skip')

parser.add_argument('--seq_count', type=int, default=None, help='Number of frames to include in the sequence')
parser.add_argument('--total_frames', type=int, default=None, help='Number of frames to include in the video')
parser.add_argument('--grid_size', nargs=2, type=int, default=None, help='Grid size for grid stack type')

args = parser.parse_args()

if args.dataset_type == "progress":
    if args.stack_type == "grid":
        assert args.seq_count is None, "Sequential count should not be provided for grid stack type"
        assert args.total_frames is None, "Total frames should not be provided for grid stack type"
        assert args.grid_size is not None, "Grid size should be provided for grid stack type"
    elif args.stack_type == "sequential":
        assert args.seq_count is not None, "Sequential count should be provided for sequential stack type"
        assert args.total_frames is None, "Total frames should not be provided for sequential stack type"
        assert args.grid_size is None, "Grid size should not be provided for sequential stack type"
    elif args.stack_type == "video":
        assert args.seq_count is None, "Sequential count should not be provided for video stack type"
        assert args.total_frames is not None, "Total frames should be provided for video stack type"
        assert args.grid_size is None, "Grid size should not be provided for video stack type"
elif args.dataset_type == "success":
    assert args.seq_count is None, "Sequential count should not be provided for success dataset type"
    assert args.total_frames is None, "Total frames should not be provided for success dataset type"
    assert args.grid_size is None, "Grid size should not be provided for success dataset type"


MAX_DATASET_SIZE = args.max_dataset_size
DRY_RUN = args.dry_run
DATASET_TYPE = args.dataset_type
FEASIBILITY_TYPE = args.feasibility_type
GT_TYPES = args.gt_types
STACK_TYPE = args.stack_type

# make sure that grid_size x skip_frames is less than or equal to the number of frames in the episode, aim for max 25 so we don't provide too much information
if STACK_TYPE == "grid":
    grid_size = (5,2) # only used for DATASET_TYPE == "progress"
    max_width = args.max_width # previously 2048
    max_height = args.max_height # previously 768
    skip_frames = args.skip_frames # previously 2 --  only used for DATASET_TYPE == "progress" or "success"
elif STACK_TYPE == "sequential":
    max_width = args.max_width # previously 3072
    max_height = args.max_height # previously 3072
    skip_frames = args.skip_frames # previously 6  # only used for DATASET_TYPE == "progress" or "success"
    sequential_count = args.seq_count # previously 6 -- only used for DATASET_TYPE == "progress"
elif STACK_TYPE == "video":
    max_width = args.max_width # previously 200
    max_height = args.max_height # previously 200
    skip_frames = args.skip_frames # previously 3  # only used for DATASET_TYPE == "progress" or "success"
    total_frames = args.total_frames # previously 10 -- only used for DATASET_TYPE == "progress"
    fps=1
else:
    raise ValueError("STACK_TYPE is incorrect")    
    
extra_info = "_may27" # add any extra info to the dataset name

if STACK_TYPE == "grid":
    dataset_name = f"{DATASET_TYPE}_grid_{max_width}_{max_height}p_{grid_size[0] if DATASET_TYPE == 'progress' else 'N'}x{grid_size[1]  if DATASET_TYPE == 'progress' else 'N'}_skip{skip_frames}_frames{extra_info}"
elif STACK_TYPE == "sequential":
    dataset_name = f"{DATASET_TYPE}_seq_{max_width}_{max_height}p_{sequential_count if DATASET_TYPE == 'progress' else 'all'}_skip{skip_frames}_frames{extra_info}"
elif STACK_TYPE == "video":
    dataset_name = f"{DATASET_TYPE}_video_{max_width}_{max_height}p_{total_frames if DATASET_TYPE == 'progress' else 'all'}_fps{5}_frames{extra_info}"
else:
    raise ValueError("STACK_TYPE is incorrect")    

def get_data(subset="training"):
    dataset_path = ORIGINAL_DATASET_PATH / subset
    lang_data = np.load(dataset_path / "lang_one_hot"  / "auto_lang_ann.npy", allow_pickle=True).item()
    # ['slide the door to the left', 'open the drawer', 'turn on the yellow lamp', '
    ann = lang_data['language']['ann']
    # we will use the annotations, these are what matter
    # [[[0. 0. 1. ... 0. 0
    emb = lang_data['language']['emb']
    # ['move_slider_left', 'open_drawer', 'turn_on_lightbulb', 'place_in_drawer',
    task = lang_data['language']['task']
    # the minimum size is 33 and the maximum size is 64
    indices = lang_data['info']['indx']
    return dataset_path, ann, emb, task, indices

class ModPlayTableSimEnv(PlayTableSimEnv):
    def __init__(self,
                 tasks: dict = {},
                 initial_states: list = [],
                 **kwargs):
        super(ModPlayTableSimEnv, self).__init__(**kwargs)
        self.render_mode="rgb_array"

    def reset(self, robot_obs=None, scene_obs=None, seed=None):
        obs = super().reset(robot_obs=robot_obs, scene_obs=scene_obs)
        return obs, {}


def create_env(scene_name="calvin_scene_B", max_width=1500, max_height=1500):
    size = min(max_width, max_height)
    GlobalHydra.instance().clear()
    with initialize(config_path="../../calvin_env/conf/"):
        cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper", f"scene={scene_name}", f"cameras.static.width={size}", f"cameras.static.height={size}"])
        cfg.env["use_egl"] = True
        cfg.env["show_gui"] = False
        cfg.env["use_vr"] = False
        cfg.env["use_scene_info"] = True

    env_cfg = {**cfg.env}
    env_cfg["tasks"] = cfg.tasks
    env_cfg.pop('_target_', None)
    env_cfg.pop('_recursive_', None)
    env = ModPlayTableSimEnv(**env_cfg)
    return env


def render_image_with_dims(env, scene_obs, robot_obs):
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    return env.render(mode="rgb_array")

def is_task_excluded(task, gt_type):
    if gt_type in ["success", "wrong"]:
        if task in ["turn_on_lightbulb", "turn_off_lightbulb", "turn_on_led", "turn_off_led"]:
            return True
        if "slider" in task:
            return True
    return False
    
infeasible_gt = None
if DATASET_TYPE == "feasibility" and FEASIBILITY_TYPE == "groundtruth" and "wrong" in GT_TYPES:
    assert FEASIBILITY_INFEASIBLE_GT_PATH.exists(), "The feasibility infeasible ground truth file does not exist."
    with open(FEASIBILITY_INFEASIBLE_GT_PATH) as f:
        infeasible_gt = json.load(f)
        infeasible_gt = {int(k): v for k, v in infeasible_gt.items()}
        print(infeasible_gt)


# Depending on the DATASET_TYPE, there are different Ground truth options (GT_TYPES)
for GT_TYPE in GT_TYPES:
    # make the dataset directory if it doesn't exist
    (GENERATED_DATASET_PATH / dataset_name / f"{GT_TYPE}_gt").mkdir(parents=True, exist_ok=True)
    
    episodes = []

    if GT_TYPE == "fail":
        dataset_path, ann, emb, task, indices = get_data(subset="validation")
    elif GT_TYPE in ["success", "wrong"]:
        dataset_path, ann, emb, task, indices = get_data(subset="training")
    
    recreate_images = False
    if STACK_TYPE == "sequential":
        if max_width > 200 or max_height > 200:
            recreate_images = True
            env = create_env(max_width=max_width, max_height=max_height)
    elif STACK_TYPE == "grid":
        recreate_images = True
        env = create_env()
        
    for ep_index, (start, end) in enumerate(indices):
        if is_task_excluded(task[ep_index], GT_TYPE):
            continue
        if infeasible_gt is not None and GT_TYPE == "wrong" and ep_index not in infeasible_gt.keys():
            continue
        frames = []
        if GT_TYPE in ["success", "wrong"]:
            for i in range(start, end + 1, skip_frames):
                if recreate_images:
                    step = np.load(f"{dataset_path}/episode_{i:07d}.npz", allow_pickle=True)
                    image = render_image_with_dims(env, step['scene_obs'], step['robot_obs'])
                    print(image.shape)
                else:
                    image = np.load(f"{dataset_path}/episode_{i:07d}.npz", allow_pickle=True)['rgb_static']
                frames.append(image)
                if DATASET_TYPE == "feasibility":
                    break # we only need one frame for feasibility
            if DATASET_TYPE == "success" and (end-start) % skip_frames != 0:
                if recreate_images:
                    step = np.load(f"{dataset_path}/episode_{end:07d}.npz", allow_pickle=True)
                    image = render_image_with_dims(env, step['scene_obs'], step['robot_obs'])
                else:
                    image = np.load(f"{dataset_path}/episode_{end:07d}.npz", allow_pickle=True)['rgb_static']
                frames.append(image)    
        elif GT_TYPE == "fail":
            all_frames = []
            lang_annotation = ann[ep_index]
            target_task = task[ep_index]
            first_scene_obs = np.load(f"{dataset_path}/episode_{(start):07d}.npz", allow_pickle=True)['scene_obs']
            first_robot_obs = np.load(f"{dataset_path}/episode_{(start):07d}.npz", allow_pickle=True)['robot_obs']
            reset_info = {"scene_obs": first_scene_obs, "robot_obs": first_robot_obs}
            observations = generate_policy_rollouts(lang_annotation, target_task, reset_info, result="fail", num_rollouts=1, neutral_init=False)
            if len(observations) == 0:
                print(f"Failed to generate GT for episode {ep_index + 1}.")
                continue
            for observation in observations[0]:
                # img = np.array(observation['rgb_obs']['rgb_static'].cpu()).squeeze()
                # print(img.shape)
                # img = np.transpose(img, (1, 2, 0))  # Transpose the image dimensions
                # convert img from float RGB to uint8 RGB
                # img = (img * 255).astype(np.uint8)
                all_frames.append(observation)

            # TODO: It's much more time saving to only render the images we need, so we can move this logic to the generate policy rollout function
            # we need to make sure that the last frame is included even if skip_frames will go over
            for i in range(0, len(all_frames), skip_frames):
                frames.append(all_frames[i])
            # Check if the last index was included, if not, add it
            if DATASET_TYPE == "success" and len(all_frames)-1 != i:
                frames.append(all_frames[-1])
        else:
            raise ValueError("GT_TYPE should be either 'success' or 'wrong' or 'fail'")
        
        
        # find the task description to use as filename
        if GT_TYPE in ["success", "fail"]:
            task_description = ann[ep_index]
        elif GT_TYPE == "wrong":
            if infeasible_gt is not None:
                task_description = infeasible_gt[ep_index]
            else:
                right_task = task[ep_index]
                # find any wrong task
                # shuffle_annotation and tasks but keep them aligned
                np.random.seed(ep_index)
                shuffle_indices = np.random.permutation(len(task))
                shuffled_ann = np.array(ann)[shuffle_indices]
                shuffled_task = np.array(task)[shuffle_indices]
                for i, (t, a) in enumerate(zip(shuffled_task, shuffled_ann)):
                    if t != right_task and not is_task_excluded(t, GT_TYPE):
                        task_description = a
                        break
        else:
            raise ValueError("GT_TYPE should be either 'success' or 'wrong'")

        episodes.append((ep_index, frames, task_description))
        
        if len(episodes) >= MAX_DATASET_SIZE:
            break
    
    for ep_index, frames, task_description in episodes:
        if STACK_TYPE == "grid":
            # Assuming all frames have the same dimensions
            frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]
            
            # Calculate the number of columns and rows for the grid
            if DATASET_TYPE != "progress":
                n_frames = len(frames)
                n_cols = int(np.ceil(np.sqrt(n_frames)))
                n_rows = int(np.ceil(n_frames / n_cols))
                grid_size = (n_cols, n_rows)
            
            # Create a new image for the grid
            grid_img = Image.new(
                'RGBA',
                (frame_width * grid_size[0], frame_height * grid_size[1])
            )

            # Paste frames into the grid image
            for index, frame in enumerate(frames):
                x = index % grid_size[0] * frame_width
                y = index // grid_size[0] * frame_height
                frame_img = Image.fromarray(frame.astype('uint8'), 'RGB')
                grid_img.paste(frame_img, (x, y))

            # resize the grid image to fit within max_width x max_height pixels maintaining aspect ratio
            # we should maximumze the width or height
            width, height = grid_img.size
            aspect_ratio = width / height
            if max_width / width < max_height / height:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            grid_img = grid_img.resize((new_width, new_height))
            if not DRY_RUN:
                grid_img.save(GENERATED_DATASET_PATH / dataset_name / f"{GT_TYPE}_gt" / f"{ep_index}_{task_description}.png")
                print(f"Saved grid image for episode {ep_index + 1}.")
        elif STACK_TYPE == "sequential":
            # Create a new image per frame and save multiple images
            num_images_in_sequence = sequential_count if DATASET_TYPE == "progress" else len(frames)
            for index, frame in enumerate(frames[:num_images_in_sequence]):
                frame_img = Image.fromarray(frame.astype('uint8'), 'RGB')
                # resize the image to fit within max dims maintaining aspect ratio
                # we should maximumze the width or height
                width, height = frame_img.size
                aspect_ratio = width / height
                if max_width / width < max_height / height:
                    new_width = max_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = max_height
                    new_width = int(new_height * aspect_ratio)
                frame_img = frame_img.resize((new_width, new_height))
                if not DRY_RUN:
                    frame_img.save(GENERATED_DATASET_PATH / dataset_name / f"{GT_TYPE}_gt" / f"{ep_index}_{task_description}_{index}.png")
            if not DRY_RUN:
                print(f"Saved sequential images for episode {ep_index + 1}.")
        elif STACK_TYPE == "video":
            resized_frames = []
            num_frames_in_video = total_frames if DATASET_TYPE == "progress" else len(frames)
            for frame in frames[:num_frames_in_video]:
                frame_img = Image.fromarray(frame.astype('uint8'), 'RGB')
                # resize the image to fit within max dims maintaining aspect ratio
                # we should maximumze the width or height
                width, height = frame_img.size
                aspect_ratio = width / height
                if max_width / width < max_height / height:
                    new_width = max_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = max_height
                    new_width = int(new_height * aspect_ratio)
                resized_frame = frame_img.resize((new_width, new_height))
                resized_frames.append(np.array(resized_frame))
            # create a video containing all the image frames
            from moviepy.editor import ImageSequenceClip
            clip = ImageSequenceClip(resized_frames, fps=fps)
            if not DRY_RUN:
                clip.write_videofile(str(GENERATED_DATASET_PATH / dataset_name / f"{GT_TYPE}_gt" / f"{ep_index}_{task_description}.mp4"), fps=clip.fps, verbose=False, logger=None)
                print(f"Saved video for episode {ep_index + 1}.")
        else:
            raise ValueError("STACK_TYPE is incorrect")
    
print("New Dataset Name is ", dataset_name)

