import base64
import requests
from pathlib import Path
import vertexai
import google
import time
import prompt_templates
import re
import argparse
import wandb
import random
import string
import os
import PIL
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import utils

DEBUG = False

load_dotenv()

# general settings
MAX_TOKENS = 1000

# OpenAI settings
DETAIL = "high" # low,  high
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_GPT_MODEL_NAME = "gpt-4o"

# VertexAI/GenAI Settings
GOOGLE_API = "genai" # genai or vertexai
if GOOGLE_API == "vertexai":
    from vertexai.generative_models import GenerativeModel
    from vertexai.generative_models import Part, GenerationConfig
elif GOOGLE_API == "genai":
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    from google.generativeai import GenerationConfig
    from PIL import Image
GCP_PROJECT_ID = "modified-badge-419206"
VERTEXAI_MODEL_NAME = "gemini-1.5-pro" #"gemini-1.5-pro-preview-0409", "gemini-1.0-pro-vision"
GENAI_MODEL_NAME = "gemini-1.5-pro-latest"

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
    return utils.analyze_response(response_text, task=args.task)

def make_wandb_image(image_path):
    img = PIL.Image.open(image_path)
    img = img.resize((256, 256))
    return wandb.Image(img)
    
def majority_voting(actions):
    # we need to decide on the final action based on majority voting and we break ties by choosing "Unsure"
    # we want to do it in a readable way, so we will use the following logic
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

# Function to write the request to the batch file
def write_to_batch_file(batch_file_path, custom_id, payload):
    with open(batch_file_path, "a") as batch_file:
        batch_request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": payload
        }
        batch_file.write(json.dumps(batch_request) + "\n")
        
def read_from_batch_file(batch_file_results_path, custom_id):
    with open(batch_file_results_path, "r") as batch_file:
        for line in batch_file:
            batch_response = json.loads(line)
            if batch_response.get("custom_id") == custom_id:
                return batch_response.get("response")["body"]
    raise Exception(f"Custom ID {custom_id} not found in the batch file")

# Create the argument parser
parser = argparse.ArgumentParser(description='LLM Evaluation Script')

# Add the arguments
parser.add_argument('task', type=str, help='The evaluation task', default='success')
parser.add_argument('dataset', type=str, help='name of the dataset folder')
parser.add_argument('--groundtruth', nargs='+', help='groundtruth to test', default=['wrong', 'fail', 'success'])
parser.add_argument('--platform', type=str, help='Platform to use (OPENAI or GOOGLE)', default='GOOGLE')
parser.add_argument('--prompt', type=int, help='Prompt template to use', default=1)
parser.add_argument('--max_test_episodes', type=int, help='Maximum number of episodes to test', default=10)
parser.add_argument('--starting_episode', type=int, help='Starting episode number', default=0)
parser.add_argument('--num_candidates', type=int, help='Number of candidates to generate', default=1)
parser.add_argument('--write_batch_file', help='Create a batch file for the dataset', default=False, action='store_true')
parser.add_argument('--read_batch_file', type=str, help='Read a batch file results and process it', default=None)
parser.add_argument('--gpt_model_name', type=str, help='GPT model name to use', default=DEFAULT_GPT_MODEL_NAME)
args = parser.parse_args()

if args.task not in ["success", "progress", "feasibility"]:
    raise ValueError("TASK should be either 'success', 'progress', or 'feasibility'")

if args.task == "feasibility":
    assert all([groundtruth in ["success", "wrong"] for groundtruth in args.groundtruth]), "Groundtruth should be either 'success' or 'wrong' for feasibility task"

PLATFORM = args.platform
PROMPT = eval(f"prompt_templates.{args.task.upper()}_PROMPT_TEMPLATE_{args.prompt}")
MAX_TEST_EPISODES = args.max_test_episodes
STARTING_EPISODE = args.starting_episode
NUM_CANDIDATES = args.num_candidates
GPT_MODEL_NAME = args.gpt_model_name
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
wandb_mode = 'online' if not DEBUG else 'disabled'
if args.write_batch_file:
    assert PLATFORM == "OPENAI", "Batch file creation is only supported for OpenAI API"
    wandb_mode = 'disabled'
    # the path of the batch file should be in a folder called batch_files in the same directory of this python script
    batch_file_path = Path(__file__).parent / "batch_files" / f"{args.task}_{args.dataset}_{args.prompt}_{random_sequence}.jsonl"
if args.read_batch_file:
    assert PLATFORM == "OPENAI", "Batch file reading is only supported for OpenAI API"
    batch_file_results_path = args.read_batch_file
    

wandb_run = wandb.init(project="LLM Progress Assessment", name=f"{args.task}_{args.dataset}_{random_sequence}", mode=wandb_mode)
wandb_run.config.update(
    {
        "task": args.task, # "success", "progress", "feasibility"
        "dataset": args.dataset,
        "groundtruth": args.groundtruth,
        "platform": PLATFORM,
        "prompt_number": args.prompt,
        "prompt_template": " ".join(PROMPT),
        "stack_type": STACK_TYPE,
        "max_test_episodes": MAX_TEST_EPISODES,
        "starting_episode": STARTING_EPISODE,
    }
)
if PLATFORM == "OPENAI":
    wandb_run.config.update({
        "gpt_model_name": GPT_MODEL_NAME,
        "image_detail": DETAIL,
    })
elif PLATFORM == "GOOGLE":
    wandb_run.config.update({
        "gcp_project_id": GCP_PROJECT_ID,
        "gemini_model_name": VERTEXAI_MODEL_NAME,
    })
else:
    raise ValueError("PLATFORM should be either 'OPENAI' or 'GOOGLE'")

for GROUNDTRUTH in [option + '_gt' for option in args.groundtruth]:
    print(f"Testing {args.dataset} with {GROUNDTRUTH}")
    DATASET_PATH = Path("/home/uoft/ramy/calvin/sample_dataset/") / args.dataset / GROUNDTRUTH

    wandb_table_columns = ["Episode", "Task Name", "Media", "Decision", "Reasoning", "Input", "Output"]
    logging_table = wandb.Table(columns=wandb_table_columns)

    # read the names of all the files inside DATASET_PATH
    if STACK_TYPE != "video":
        file_names = [file.name for file in DATASET_PATH.glob("*.png")]
        file_names.sort(key=lambda x: int(x.split('_')[0]))
    else:
        file_names = [file.name for file in DATASET_PATH.glob("*.mp4")]
        file_names.sort(key=lambda x: int(x.split('_')[0]))

    if STACK_TYPE == "sequential":
        new_file_names = []
        last_ep_index = -1
        for file in file_names:
            ep_index = int(file.split('_')[0])
            if last_ep_index != ep_index:
                new_file_names.append([file])
            else:
                new_file_names[-1].append(file)
            last_ep_index = ep_index
        for episode in new_file_names:
            episode.sort(key=lambda x: int(x.split('_')[2].strip('.png')))
        file_names = new_file_names

    file_names = file_names[STARTING_EPISODE:STARTING_EPISODE+MAX_TEST_EPISODES]

    num_correct = 0
    total_evals = 0

    def process_file(dataset_file):
        print(f"Starting to process file {dataset_file}")
        try:
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
                    if item in [prompt_templates.GRID_IMAGE_PLACEHOLDER, prompt_templates.SEQUENCE_IMAGES_PLACEHOLDER]:
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
                                if "success" not in str(DATASET_PATH):
                                    # we need regex to find the number in the dataset path. For example, in seq_512_512p_5_skip6_frames, the count is 5, so we look for the integer between p_ and _skip
                                    image_count = int(re.search(r'p_(\d+)_skip', str(DATASET_PATH)).group(1))
                                    # print(f"Image count: {image_count}")
                                else:
                                    image_count = len(image_path)
                            elif "grid_" in str(DATASET_PATH):
                                # we need regex to find the number again, also between p_ and _skip. However, it will be two numbers separated by x and we need the multiplication of them. For example p_5x2_skip will have a count of 10
                                image_count = int(re.search(r'p_(\d+)x(\d+)_skip', str(DATASET_PATH)).group(1)) * int(re.search(r'p_(\d+)x(\d+)_skip', str(DATASET_PATH)).group(2))
                                print(f"Image count: {image_count}")
                            else:
                                raise Exception("Image count is not available for this dataset")
                            assert image_count == len(image_path), f"Image count does not match the number of images, got {image_count} and {len(image_path)} respectively"
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

                # Query the model
                output = []
                actions = []
                reasoning = []
                for candidate_idx in range(NUM_CANDIDATES):
                    max_retries = 10
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            response = None
                            if args.write_batch_file:
                                custom_id = f"request-{GROUNDTRUTH}-{image_index}.{candidate_idx}"
                                write_to_batch_file(batch_file_path, custom_id, payload)
                            elif args.read_batch_file:
                                custom_id = f"request-{GROUNDTRUTH}-{image_index}.{candidate_idx}"
                                response_text = read_from_batch_file(batch_file_results_path, custom_id)['choices'][0]['message']['content']
                            else:
                                time.sleep(random.randint(1, 10))
                                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                                response_text = response.json()['choices'][0]['message']['content']
                            break
                        except Exception as e:
                            print(f"Exception occurred: {e} {response}")
                            retry_count += 1
                            if retry_count < max_retries:
                                retry_sleep = 60 + random.randint(1, 15)
                                print(f"Retrying in {retry_sleep} seconds...")
                                time.sleep(retry_sleep)
                            else:
                                raise Exception("Max retries reached. Unable to complete the request.")

                    if retry_count == max_retries:
                        raise Exception("Max retries reached. Unable to complete the request.")

                    if args.write_batch_file:
                        continue
                                        
                    candidate_action, candidate_reasoning = analyze_response(response_text)
                    print(f"\n\n{image_index}.{candidate_idx} - {image_name}: {candidate_action} - {candidate_reasoning}")
                    actions.append(candidate_action)
                    reasoning.append(candidate_reasoning)
                    output.append(response_text)

                if args.write_batch_file:
                    return 0, 0

                action = majority_voting(actions)
                text_input = "COMPLTE PROMPT:" + " ".join(PROMPT) + "\n\n" + "TEXT PORTION: " + processed_item
            elif PLATFORM == "GOOGLE":
                if GOOGLE_API == "vertexai":
                    # Initialize Vertex AI
                    vertexai.init(project=GCP_PROJECT_ID, location="us-central1")
                    # Load the model
                    multimodal_model = GenerativeModel(VERTEXAI_MODEL_NAME)
                elif GOOGLE_API == "genai":
                    genai.configure(api_key=GOOGLE_API_KEY)
                    multimodal_model = GenerativeModel(GENAI_MODEL_NAME)
                else:
                    raise ValueError("GOOGLE_API should be either 'vertexai' or 'genai'")

                # prepare the prompt content based on the prompt template
                input = []
                for item in PROMPT:
                    if item == prompt_templates.GRID_IMAGE_PLACEHOLDER or item == prompt_templates.SEQUENCE_IMAGES_PLACEHOLDER:
                        if GOOGLE_API == "vertexai":
                            input.extend(
                                [Part.from_data(encode_image(img), mime_type="image/jpeg") for img in image_path] if type(image_path) == list else [Part.from_data(encode_image(image_path), mime_type="image/jpeg")]
                            )
                        elif GOOGLE_API == "genai":
                            input.extend([Image.open(img) for img in image_path] if type(image_path) == list else [Image.open(image_path)])
                        else:
                            raise ValueError("GOOGLE_API should be either 'vertexai' or 'genai'")
                    elif item == prompt_templates.VIDEO_PLACEHOLDER:
                        def load_video(video_path):
                            with open(video_path, "rb") as video_file:
                                return video_file.read()
                        if GOOGLE_API == "genai":
                            raise Exception("Video is not available in GenAI API")
                        input.extend([Part.from_data(load_video(image_path), mime_type="video/mp4")])
                    else:
                        # we consider it the prompt text, so we just replace the remaining placeholders
                        processed_item = item
                        if prompt_templates.TASK_NAME_PLACEHOLDER in item:
                            assert image_name is not None, "Image name is required for the prompt"
                            # print(image_name)
                            processed_item = processed_item.replace(prompt_templates.TASK_NAME_PLACEHOLDER, image_name)
                        if prompt_templates.IMAGE_COUNT_PLACEHOLDER in item:
                            if args.task == "progress":
                                if "seq_" in str(DATASET_PATH):
                                    if "success" not in str(DATASET_PATH):
                                        # we need regex to find the number in the dataset path. For example, in seq_512_512p_5_skip6_frames, the count is 5, so we look for the integer between p_ and _skip
                                        image_count = int(re.search(r'p_(\d+)_skip', str(DATASET_PATH)).group(1))
                                        # print(f"Image count: {image_count}")
                                    else:
                                        image_count = len(image_path)
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
                            assert image_count == len(image_path), f"Image count does not match the number of images, got {image_count} and {len(image_path)} respectively"
                            processed_item = processed_item.replace(prompt_templates.IMAGE_COUNT_PLACEHOLDER, str(image_count))
                        if GOOGLE_API == "vertexai":
                            input.extend([Part.from_text(processed_item)])
                        elif GOOGLE_API == "genai":
                            input.extend([processed_item])
                        else:
                            raise ValueError("GOOGLE_API should be either 'vertexai' or 'genai'")

                # Query the model
                output = []
                actions = []
                reasoning = []
                for candidate_idx in range(NUM_CANDIDATES):
                    max_retries = 10
                    retry_count = 0
                    while retry_count < max_retries:
                        try:
                            response = None
                            time.sleep(random.randint(1, 10))
                            response = multimodal_model.generate_content(input, generation_config=GenerationConfig(candidate_count=1))
                            response_text = response.candidates[0].content.parts[0].text
                            break
                        except Exception as e:
                            print(f"Exception occurred: {e} {response}")
                            retry_count += 1
                            if retry_count < max_retries:
                                retry_sleep = 45 + random.randint(10, 30)
                                print(f"Retrying in {retry_sleep} seconds...")
                                time.sleep(retry_sleep)
                            else:
                                raise Exception("Max retries reached. Unable to complete the request.")

                    candidate_action, candidate_reasoning = analyze_response(response_text)
                    print(f"\n\n{image_index}.{candidate_idx} - {image_name}: {candidate_action} - {candidate_reasoning}")
                    actions.append(candidate_action)
                    reasoning.append(candidate_reasoning)
                    output.append(response_text)
                action = majority_voting(actions)
                text_input = "COMPLTE PROMPT:" + " ".join(PROMPT) + "\n\n" + "TEXT PORTION: " + processed_item
            else:
                raise ValueError("PLATFORM should be either 'OPENAI' or 'GOOGLE'")

            # Log a new row to the table
            # "Episode", "Task Name", "Media", "Decision", "Reasoning", "Result", "Input", "Output"
            if type(image_path) == list:
                media = [make_wandb_image(str(img)) for img in image_path]
            else:
                if str(image_path).endswith('.mp4'):
                    media = [wandb.Video(str(image_path))]
                else:
                    media = [make_wandb_image(str(image_path))]

            logging_table.add_data(image_index, image_name, media, action, '\n\n'.join(reasoning), text_input, '\n\n'.join(output))
            return utils.evaluate_action_correctness(action, args.task, GROUNDTRUTH)
        except Exception as e:
            print(f"Error processing file {dataset_file}: {e}")
            raise
            return 0, 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_file, dataset_file): dataset_file for dataset_file in file_names}
        for future in as_completed(futures):
            correct, total = future.result()
            num_correct += correct
            total_evals += total

    wandb_run.log({f"results {GROUNDTRUTH}": logging_table})
    
    if not args.write_batch_file:
        try:
            percent_correct = num_correct / total_evals * 100
            wandb_run.log({f"percent correct {GROUNDTRUTH}": percent_correct})
        except Exception as e:
            print(f"Error calculating percentage: {e}")
