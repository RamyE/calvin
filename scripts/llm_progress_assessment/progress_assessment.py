import base64
import requests
from pathlib import Path
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image, GenerationConfig
import google
import time
import prompt_templates
import re
import argparse
import wandb
import random
import string
import requests
import time
import os

# general settings
MAX_TOKENS = 500

# OpenAI settings
DETAIL = "high" # low,  high
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL_NAME = "gpt-4-turbo"

# VertexAI Settings
GCP_PROJECT_ID = "modified-badge-419206"
GEMINI_MODEL_NAME = "gemini-1.5-pro-preview-0409" #"gemini-1.5-pro-preview-0409", "gemini-1.0-pro-vision"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def custom_strip(string, strip_sequence):
    if string.startswith(strip_sequence):
        string = string[len(strip_sequence):]  # Remove the sequence at the start
    if string.endswith(strip_sequence):
        string = string[:-len(strip_sequence)]  # Remove the sequence at the end
    return string

def analyze_response(response_text):
    # most of the early promots will have the action mentioned anywhere, but eventually we ask for the action at the end
    try:
        if args.task == "progress":
            if "Keep Going" in response_text and not "Reset" in response_text and not "Unsure" in response_text:
                action = "Keep Going"
            elif "Reset" in response_text and not "Keep Going" in response_text and not "Unsure" in response_text:
                action = "Reset"
            elif "Unsure" in response_text and not "Keep Going" in response_text and not "Reset" in response_text:
                action = "Unsure"
            else:
                action = None
        elif args.task == "success":
            if "Succeeded" in response_text and not "Failed" in response_text and not "Unsure" in response_text:
                action = "Succeeded"
            elif "Failed" in response_text and not "Succeeded" in response_text and not "Unsure" in response_text:
                action = "Failed"
            elif "Unsure" in response_text and not "Succeeded" in response_text and not "Failed" in response_text:
                action = "Unsure"
            else:
                action = None
                
        # if action is None, we will look into the last line only now and be more flexible
        if action is None:
            if args.task == "progress":
                if "keep going" in response_text.split('\n')[-1].lower():
                    action = "Keep Going"
                elif "reset" in response_text.split('\n')[-1].lower():
                    action = "Reset"
                elif "unsure" in response_text.split('\n')[-1].lower():
                    action = "Unsure"
            elif args.task == "success":
                if "succeeded" in response_text.split('\n')[-1].lower():
                    action = "Succeeded"
                elif "failed" in response_text.split('\n')[-1].lower():
                    action = "Failed"
                elif "unsure" in response_text.split('\n')[-1].lower():
                    action = "Unsure"
        reasoning = response_text
    except Exception as e:
        print(f"Failed to analyze response: {e}")
        action = None
        reasoning = None
    return action, reasoning

def majority_voting(actions):
    # we need to decide on the final action based on majority voting and we break ties by choosing "Unsure"
    # we want to do it in a readible way, so we will use the following logic
    action_counts = {}
    for action in actions:
        if action in action_counts:
            action_counts[action] += 1
        else:
            action_counts[action] = 1
    max_count = max(action_counts.values())
    max_actions = [action for action, count in action_counts.items() if count == max_count]
    if len(max_actions) == 1:
        return max_actions[0]
    else:
        return "Unsure"

# Create the argument parser
parser = argparse.ArgumentParser(description='LLM Evaluation Script')

# Add the arguments
parser.add_argument('task', type=str, help='The evaluation task', default='success')
parser.add_argument('dataset', type=str, help='name of the dataset folder')
parser.add_argument('--groundtruth', nargs='+', help='groundtruth to test', default=['wrong', 'fail', 'success'])
parser.add_argument('--platform', type=str, help='Platform to use (OPENAI or VERTEXAI)', default='VERTEXAI')
parser.add_argument('--prompt', type=int, help='Prompt template to use', default=1)
parser.add_argument('--max_test_episodes', type=int, help='Maximum number of episodes to test', default=10)
parser.add_argument('--num_candidates', type=int, help='Number of candidates to generate', default=1)
args = parser.parse_args()

if args.task not in ["success", "progress", "feasibility"]:
    raise ValueError("TASK should be either 'success', 'progress', or 'feasibility'")

PLATFORM = args.platform
PROMPT = eval(f"prompt_templates.{args.task.upper()}_PROMPT_TEMPLATE_{args.prompt}")
MAX_TEST_EPISODES = args.max_test_episodes
NUM_CANDIDATES = args.num_candidates
# figure out the STACK_TYPE based on the DATASET_PATH
if "grid_" in str(args.dataset):
    STACK_TYPE = "composite"
elif "seq_" in str(args.dataset):
    STACK_TYPE = "sequential"
elif "video_" in str(args.dataset):
    STACK_TYPE = "video"
else:
    raise ValueError("STACK_TYPE is incorrect")

random_sequence = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
wandb_run = wandb.init(project="LLM Progress Assessment", name=f"{args.task}_{args.dataset}_{random_sequence}")
wandb.config.update(
    {
        "task": args.task, # "success", "progress", "feasibility
        "dataset": args.dataset,
        "groundtruth": args.groundtruth,
        "platform": PLATFORM,
        "prompt_number": args.prompt,
        "prompt_template": " ".join(PROMPT),
        "stack_type": STACK_TYPE,
        "max_test_episodes": MAX_TEST_EPISODES,
    }
)
if PLATFORM == "OPENAI":
    wandb.config.update({
        "gpt_model_name": GPT_MODEL_NAME,
        "image_detail": DETAIL,
    })
elif PLATFORM == "VERTEXAI":
    wandb.config.update({
        "gcp_project_id": GCP_PROJECT_ID,
        "gemini_model_name": GEMINI_MODEL_NAME,
    })

for GROUNDTRUTH in [option + '_gt' for option in args.groundtruth]:
    print(f"Testing {args.dataset} with {GROUNDTRUTH}")
    DATASET_PATH = Path("/home/uoft/ramy/calvin/sample_dataset/") / args.dataset / GROUNDTRUTH

    wandb_table_columns = ["Episode", "Task Name", "Media", "Decision", "Reasoning", "Input", "Output"]
    logging_table = wandb.Table(columns=wandb_table_columns)
    wandb_run.log({f"results {args.dataset} {GROUNDTRUTH}": logging_table})


    # read the names of all the files inside DATASET_PATH
    if STACK_TYPE != "video":
        file_names = [file.name for file in DATASET_PATH.glob("*.png")]
        file_names.sort(key=lambda x: int(x.split('_')[0]))
    else:
        file_names = [file.name for file in DATASET_PATH.glob("*.mp4")]
        file_names.sort(key=lambda x: int(x.split('_')[0]))

    if STACK_TYPE == "sequential":
        new_file_names = []
        for file in file_names:
            ep_index = int(file.split('_')[0])
            if len(new_file_names) <= ep_index:
                new_file_names.append([file])
            else:
                new_file_names[ep_index].append(file)
        for episode in new_file_names:
            episode.sort(key=lambda x: int(x.split('_')[2].strip('.png')))
        file_names = new_file_names
        

    file_names = file_names[0:MAX_TEST_EPISODES]
    print(file_names)

    num_correct = 0
    total_evals = 0
    for dataset_file in file_names:
        if type(dataset_file) == list:
            image_name = dataset_file[0].split('_')[1]
            image_index = dataset_file[0].split('_')[0]
            image_path = [DATASET_PATH / file for file in dataset_file]
        else:
            image_name = custom_strip(custom_strip(dataset_file.split('_')[1], '.png'), '.mp4')
            image_index = dataset_file.split('_')[0]
            image_path = DATASET_PATH / dataset_file

        if PLATFORM == "OPENAI":
            # Getting the base64 string
            if type(image_path) == list:
                base64_images = [encode_image(image) for image in image_path]
            else:
                base64_images = [encode_image(image_path)]

            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
            }

            # prepare the payload content based on the prompt template
            payload_content = []
            for item in PROMPT:
                if item in  [prompt_templates.GRID_IMAGE_PLACEHOLDER, prompt_templates.SEQUENCE_IMAGES_PLACEHOLDER]:
                    payload_content.extend([
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": DETAIL
                            }
                        } for base64_image in base64_images
                    ])
                elif item == prompt_templates.VIDEO_PLACEHOLDER:
                    raise Exception("Video is not available in OpenAI API")
                else:
                    # we consider it the prompt text, so we just replace the remaining placeholders
                    processed_item = item
                    if prompt_templates.TASK_NAME_PLACEHOLDER in item:
                        assert image_name is not None, "Image name is required for the prompt"
                        processed_item = processed_item.replace(prompt_templates.TASK_NAME_PLACEHOLDER, image_name)
                    if prompt_templates.IMAGE_COUNT_PLACEHOLDER in item:
                        if "seq_" in str(DATASET_PATH):
                            # we need regex to find the number in the dataset path. For example, in seq_512_512p_5_skip6_frames, the count is 5, so we look for the integer between p_ and _skip
                            image_count = int(re.search(r'p_(\d+)_skip', str(DATASET_PATH)).group(1))
                            # print(f"Image count: {image_count}")
                        elif "grid_" in str(DATASET_PATH):
                            # we need regex to find the number again, also between p_ and _skip. However, it will be two numbers separated by x and we need the multiplication of them. For example p_5x2_skip will have a count of 10
                            image_count = int(re.search(r'p_(\d+)x(\d+)_skip', str(DATASET_PATH)).group(1)) * int(re.search(r'p_(\d+)x(\d+)_skip', str(DATASET_PATH)).group(2))
                            print(f"Image count: {image_count}")
                        else:
                            raise Exception("Image count is not available for this dataset")
                        assert image_count == len(image_path), "Image count does not match the number of images"
                        processed_item = processed_item.replace(prompt_templates.IMAGE_COUNT_PLACEHOLDER, str(image_count))
                    payload_content.append({
                        "type": "text",
                        "text": processed_item
                    })
                    
            payload = {
            "model": GPT_MODEL_NAME,
            "messages": [
                {
                "role": "user",
                "content": payload_content
                }
            ],
            "max_tokens": MAX_TOKENS
            }

            max_retries = 5
            retry_count = 0
            while retry_count < max_retries:
                try:
                    response = None
                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                    response_text = response.json()['choices'][0]['message']['content']
                    break
                except Exception as e:
                    print(f"Exception occurred: {e}")
                    if response is not None:
                        print(response.json())
                    retry_count += 1
                    time.sleep(60)

            if retry_count == max_retries:
                raise Exception("Max retries reached. Unable to complete the request.")

            action, reasoning = analyze_response(response_text)
            print(f"\n\n{image_index} - {image_name}: {action} - {reasoning}")
            input = "COMPLTE PROMPT:" + " ".join(PROMPT) + "\n\n" + "TEXT PORTION: " + processed_item
            output = response_text
        elif PLATFORM == "VERTEXAI":        
            # Initialize Vertex AI
            vertexai.init(project=GCP_PROJECT_ID, location="us-central1")
            # Load the model
            multimodal_model = GenerativeModel(GEMINI_MODEL_NAME)
            
            # prepare the prompt content based on the prompt template
            input = []
            for item in PROMPT:
                if item == prompt_templates.GRID_IMAGE_PLACEHOLDER or item == prompt_templates.SEQUENCE_IMAGES_PLACEHOLDER:
                    input.extend(
                        [Part.from_data(encode_image(img), mime_type="image/jpeg") for img in image_path] if type(image_path) == list else [Part.from_data(encode_image(image_path), mime_type="image/jpeg")]
                    )
                elif item == prompt_templates.VIDEO_PLACEHOLDER:
                    def load_video(video_path):
                        with open(video_path, "rb") as video_file:
                            return video_file.read()
                    input.extend([Part.from_data(load_video(image_path), mime_type="video/mp4")])
                else:
                    # we consider it the prompt text, so we just replace the remaining placeholders
                    processed_item = item
                    if prompt_templates.TASK_NAME_PLACEHOLDER in item:
                        assert image_name is not None, "Image name is required for the prompt"
                        print(image_name)
                        processed_item = processed_item.replace(prompt_templates.TASK_NAME_PLACEHOLDER, image_name)
                    if prompt_templates.IMAGE_COUNT_PLACEHOLDER in item:
                        if args.task == "progress":
                            if "seq_" in str(DATASET_PATH):
                                # we need regex to find the number in the dataset path. For example, in seq_512_512p_5_skip6_frames, the count is 5, so we look for the integer between p_ and _skip
                                image_count = int(re.search(r'p_(\d+)_skip', str(DATASET_PATH)).group(1))
                                # print(f"Image count: {image_count}")
                            elif "grid_" in str(DATASET_PATH):
                                # we need regex to find the number again, also between p_ and _skip. However, it will be two numbers separated by x and we need the multiplication of them. For example p_5x2_skip will have a count of 10
                                image_count = int(re.search(r'p_(\d+)x(\d+)_skip', str(DATASET_PATH)).group(1)) * int(re.search(r'p_(\d+)x(\d+)_skip', str(DATASET_PATH)).group(2))
                                print(f"Image count: {image_count}")
                            else:
                                raise Exception("Image count is not available for this dataset")
                        elif args.task == "success":
                            if "seq_" in str(DATASET_PATH):
                                image_count = len(image_path)
                            elif "grid_" in str(DATASET_PATH):
                                raise NotImplementedError("TODO: Image count is not available for this dataset yet")
                            else:
                                raise Exception("Image count is not available for this dataset")
                        processed_item = processed_item.replace(prompt_templates.IMAGE_COUNT_PLACEHOLDER, str(image_count))
                    input.extend(
                            [
                                Part.from_text(processed_item)
                            ]
                        )
            
            # Query the model
            input = "COMPLTE PROMPT:" + " ".join(PROMPT) + "\n\n" + "TEXT PORTION: " + processed_item
            output = []
            actions = []
            reasoning = []
            for candidate_idx in range(NUM_CANDIDATES):
                try:
                    response = multimodal_model.generate_content(input, generation_config=GenerationConfig(candidate_count=1))
                except google.api_core.exceptions.ResourceExhausted:
                    print(f"Resource Exhausted for {image_index}")
                    time.sleep(65)
                    response = multimodal_model.generate_content(input, generation_config=GenerationConfig(candidate_count=1))

                response_text = response.candidates[0].content.parts[0].text
                candidate_action, candidate_reasoning = analyze_response(response_text)
                print(f"\n\n{image_index}.{candidate_idx} - {image_name}: {candidate_action} - {candidate_reasoning}")
                actions.append(candidate_action)
                reasoning.append(candidate_reasoning)
                output.append(response_text)
            action = majority_voting(actions)            
        else:
            raise ValueError("PLATFORM should be either 'OPENAI' or 'VERTEXAI'")
        
        # Log a new row to the table
        # "Episode", "Task Name", "Media", "Decision", "Reasoning", "Result", "Input", "Output"
        if type(image_path) == list:
            media = [wandb.Image(str(img)) for img in image_path]
        else:
            if str(image_path).endswith('.mp4'):
                media = [wandb.Video(str(image_path))]
            else:
                media = [wandb.Image(str(image_path))]

        logging_table.add_data(image_index, image_name, media, action, '\n\n'.join(reasoning), input, '\n\n'.join(output))
        wandb_run.log({f"results {args.dataset} {GROUNDTRUTH}": logging_table})
        if args.task == "progress":
            if "Keep Going" in str(action):
                if "success" in GROUNDTRUTH:
                    num_correct += 1
                total_evals += 1
            elif "Reset" in str(action):
                if "wrong" in GROUNDTRUTH or "fail" in GROUNDTRUTH:
                    num_correct += 1
                total_evals += 1
        elif args.task == "success":
            if "Succeeded" in str(action):
                if "success" in GROUNDTRUTH:
                    num_correct += 1
                total_evals += 1
            elif "Failed" in str(action):
                if "wrong" in GROUNDTRUTH or "fail" in GROUNDTRUTH:
                    num_correct += 1
                total_evals += 1
    
    try:
        percent_correct = num_correct / total_evals * 100
        wandb_run.log({f"percent correct {args.dataset} {GROUNDTRUTH}": percent_correct})
    except Exception as e:
        print(f"Error calculating percentage: {e}")
    

