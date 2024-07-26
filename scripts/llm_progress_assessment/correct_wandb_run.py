import wandb
from utils import analyze_response, evaluate_action_correctness
import pandas as pd
import os
import json
from pathlib import Path
import argparse

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
print(f"Task: {task}")


# List all artifacts associated with the run
artifacts = run.logged_artifacts()
artifact_names = [artifact.name for artifact in artifacts]

# Print artifact names to identify the one you need
print("Artifacts associated with this run:")
for name in artifact_names:
    print(name)


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
    print(artifact_dir)
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
    # print(table)
    

    decision_column_name = 'Decision'
    output_column_name = 'Output'

    new_values = [analyze_response(value, task=task)[0] for value in table[output_column_name]]
    table[decision_column_name] = new_values
    # print(new_values)
    num_correct = 0
    total_evals = 0
    for action in new_values:
        correct, total = evaluate_action_correctness(action, task, gt)
        num_correct += correct
        total_evals += total
    try:
        percent_correct = num_correct / total_evals * 100
        print(percent_correct)
    except Exception as e:
        print(f"Error calculating percentage: {e}")
        
    # new_data = data.copy()
    # new_data['data'] = table.values.tolist()
    # print(f"Table {gt} updated successfully.")
    # # print(new_data)
    # # overwrite the json file
    # with open(json_file_path, 'w') as json_file:
    #     json.dump(new_data, json_file)
    
    # we will recreate the wandb table from the df
    new_table = wandb.Table(columns=data['columns'])
    for _, row in table.iterrows():
        media = [wandb.Image(str(Path(artifact_dir) / media['path'])) for media in row['Media']]
        new_table.add_data(row['Episode'], row['Task Name'], media, row['Decision'], row['Reasoning'], row['Input'], row['Output'])
    
    # upload the json file back to wandb
    # new_artifact = wandb.Artifact(artifact_name.split(':')[0], type='run_table')
    # new_artifact.add_dir(artifact_dir)
    with wandb.init(project=project_name, entity=entity_name, id=run_id, resume="must") as run:
        # print(run.summary.keys())
        # print(run.summary['results wrong_gt'])
        # print(run.summary['results success_gt'].keys())
        # run.log_artifact(new_artifact)
        # new_artifact.wait()
        run.summary[table_key] = new_table
        run.log({f"percent correct {gt}": percent_correct})
        run.summary[f"percent correct {gt}"] = percent_correct