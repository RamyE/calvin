import wandb
from utils import analyze_response, evaluate_action_correctness
import pandas as pd
import os
import json
from pathlib import Path
import argparse
import numpy as np

# Initialize wandb with your project details
wandb.login()

entity_name = 'ramye'
project_name = 'LLM Progress Assessment'

# we will read the run_id through argparse
parser = argparse.ArgumentParser()
parser.add_argument("run_id", type=str)
args = parser.parse_args()
run_id = args.run_id

# Resume the run
api = wandb.Api()
run = api.run(f'{entity_name}/{project_name}/{run_id}')
task = run.config['task']


artifacts = run.logged_artifacts()
artifact_names = [artifact.name for artifact in artifacts]

unsure_rates = []

for gt in ['wrong_gt', 'success_gt']:
    # Retrieve the table
    table_key = f'results {gt}'

    artifact_name = None
    for name in artifact_names:
        if gt in name and ':v0' in name:
            artifact_name = name
            break
    assert artifact_name is not None, f"Artifact {table_key} not found."
    artifact = api.artifact(f'{entity_name}/{project_name}/{artifact_name}', type='run_table')
    artifact_dir = artifact.download()
    # print(artifact_dir)
    # find the json file inside artifact_dir
    json_file_path = None
    for file in os.listdir(artifact_dir):
        if file.endswith(".json"):
            json_file_path = os.path.join(artifact_dir, file)
            break
    # print(json_file_path)
    assert json_file_path is not None, "JSON file not found."
    # Load the table
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    table = pd.DataFrame(data['data'], columns=data['columns'])
    
    # print(table.head())
    decision_column_name = 'Decision'
    output_column_name = 'Output'

    unsure_count = sum([1 for value in table[decision_column_name] if str(value).lower() == 'unsure'])
    total_count = len(table)
    
    # print(f"Unsure rate for {gt}: {unsure_count/total_count:.2f} ({unsure_count}/{total_count})")
    unsure_rates.append(unsure_count/total_count)
unsure_rates = [round(rate, 2) for rate in unsure_rates]
average_unsure_rate = np.mean(unsure_rates)
average_unsure_rate = round(average_unsure_rate, 2)
print(f"{run_id} Unsure rate: ", average_unsure_rate, unsure_rates)