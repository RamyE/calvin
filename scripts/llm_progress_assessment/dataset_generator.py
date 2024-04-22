from pathlib import Path
import numpy as np
from PIL import Image
from generate_policy_rollouts import generate_policy_rollouts

ORIGINAL_DATASET_PATH = Path('/mnt/vol1/ramy/calvin/dataset/task_B_D/')
GENERATED_DATASET_PATH = Path('sample_dataset')
MAX_DATASET_SIZE = 200

DATASET_TYPE = "success" # "success" or "progress" or "feasibility"

STACK_TYPE = "sequential" # "composite" or "sequential" or "video"
# make sure that grid_size x skip_frames is less than or equal to the number of frames in the episode, aim for max 25 so we don't provide too much information
if STACK_TYPE == "composite":
    grid_size = (5,2) # only used for DATASET_TYPE == "progress"
    max_width = 2048
    max_height = 768
    skip_frames = 2
elif STACK_TYPE == "sequential":
    max_width = 2000
    max_height = 2000
    skip_frames = 8
    sequential_count = 5 # only used for DATASET_TYPE == "progress"
elif STACK_TYPE == "video":
    max_width = 200
    max_height = 200
    skip_frames = 3
    total_frames = 10 # only used for DATASET_TYPE == "progress"
    fps=1
else:
    raise ValueError("STACK_TYPE is incorrect")    
    
if STACK_TYPE == "composite":
    dataset_name = f"{DATASET_TYPE}_grid_{max_width}_{max_height}p_{grid_size[0] if DATASET_TYPE == 'progress' else 'N'}x{grid_size[1]  if DATASET_TYPE == 'progress' else 'N'}_skip{skip_frames}_frames"
elif STACK_TYPE == "sequential":
    dataset_name = f"{DATASET_TYPE}_seq_{max_width}_{max_height}p_{sequential_count if DATASET_TYPE == 'progress' else 'all'}_skip{skip_frames}_frames"
elif STACK_TYPE == "video":
    dataset_name = f"{DATASET_TYPE}_video_{max_width}_{max_height}p_{total_frames if DATASET_TYPE == 'progress' else 'all'}_fps{5}_frames"
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

# Depending on the DATASET_TYPE, there are different Ground truth options (GT_TYPES)
for GT_TYPE in ["fail"]: # ["success", "wrong", "fail"]:
    # make the dataset directory if it doesn't exist
    (GENERATED_DATASET_PATH / dataset_name / f"{GT_TYPE}_gt").mkdir(parents=True, exist_ok=True)
    
    episodes = []

    if GT_TYPE == "fail":
        dataset_path, ann, emb, task, indices = get_data(subset="validation")
    elif GT_TYPE in ["success", "wrong"]:
        dataset_path, ann, emb, task, indices = get_data(subset="training")
    
    for ep_index, (start, end) in enumerate(indices):
        
        all_frames = []
        if GT_TYPE in ["success", "wrong"]:
            for i in range(start, end + 1):
                image = np.load(f"{dataset_path}/episode_{i:07d}.npz", allow_pickle=True)['rgb_static']
                all_frames.append(image)
        elif GT_TYPE == "fail":
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
        else:
            raise ValueError("GT_TYPE should be either 'success' or 'wrong' or 'fail'")
        
        frames = []
        # we need to make sure that the last frame is included even if skip_frames will go over
        for i in range(0, len(all_frames), skip_frames):
            frames.append(all_frames[i])
        # Check if the last index was included, if not, add it
        if len(all_frames)-1 != i:
            frames.append(all_frames[-1])
        
        # find the task description to use as filename
        if GT_TYPE in ["success", "fail"]:
            task_description = ann[ep_index]
        elif GT_TYPE == "wrong":
            right_task = task[ep_index]
            # find any wrong task
            # shuffle_annotation and tasks but keep them aligned
            shuffle_indices = np.random.permutation(len(task))
            shuffled_ann = np.array(ann)[shuffle_indices]
            shuffled_task = np.array(task)[shuffle_indices]
            for i, (t, a) in enumerate(zip(shuffled_task, shuffled_ann)):
                if t != right_task:
                    task_description = a
                    break
        else:
            raise ValueError("GT_TYPE should be either 'success' or 'wrong'")

        episodes.append((ep_index, frames, task_description))
        
        if len(episodes) >= MAX_DATASET_SIZE:
            break
    
    for ep_index, frames, task_description in episodes:
        if STACK_TYPE == "composite":
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

            # resize the grid image to fit within 768 x 2048 pixels maintaining aspect ratio
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
                frame_img.save(GENERATED_DATASET_PATH / dataset_name / f"{GT_TYPE}_gt" / f"{ep_index}_{task_description}_{index}.png")
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
            clip.write_videofile(str(GENERATED_DATASET_PATH / dataset_name / f"{GT_TYPE}_gt" / f"{ep_index}_{task_description}.mp4"), fps=clip.fps, verbose=False, logger=None)
            print(f"Saved video for episode {ep_index + 1}.")
        else:
            raise ValueError("STACK_TYPE is incorrect")
    
print("New Dataset Name is ", dataset_name)
