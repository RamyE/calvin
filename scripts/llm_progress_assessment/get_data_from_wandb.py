import wandb

entity_name = 'ramye'
project_name = 'LLM Progress Assessment'
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run_id", type=str)
args = parser.parse_args()
run_id = args.run_id

api = wandb.Api()
run = api.run(f'{entity_name}/{project_name}/{run_id}')

results = []
for gt in ['success_gt', 'wrong_gt']:
    results.append(run.summary[f"percent correct {gt}"])

print(" - ".join(["%.2f" % x for x in results]))
