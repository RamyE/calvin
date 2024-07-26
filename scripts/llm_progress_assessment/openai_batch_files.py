import argparse
import openai
import json
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

def upload_batch_file(batch_file_path):
    with open(batch_file_path, "rb") as file:
        batch_input_file = client.files.create(file=file, purpose="batch")
    return batch_input_file.id

def create_batch(input_file_id, description="Batch job"):
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": description
        }
    )
    return batch.to_dict()

def cancel_batch(batch_id):
    response = client.batches.cancel(batch_id)
    return response.to_dict()

def check_progress(batch_id):
    response = client.batches.retrieve(batch_id)
    return response.to_dict()

def retrieve_results(batch_id):
    batch_info = client.batches.retrieve(batch_id)
    output_file_id = batch_info.to_dict().get('output_file_id')
    if not output_file_id:
        raise ValueError("Batch results are not ready or do not exist.")
    
    content = client.files.content(output_file_id)
    batch_files_dir = Path(__file__).parent / "batch_files"
    output_file_name = str(batch_files_dir / ("output_" + batch_info.to_dict().get('metadata').get('description', 'unknown_batch_output.jsonl') + '.jsonl'))
    with open(output_file_name, "wb") as f:
        f.write(content.read())
    return f"Results written to {output_file_name}"

def list_batches(limit=10):
    response = client.batches.list(limit=limit)
    batches = [batch.to_dict() for batch in response]
    return batches

def main():
    parser = argparse.ArgumentParser(description="Manage OpenAI Batch API operations.")
    parser.add_argument("operation", choices=["submit", "cancel", "progress", "results", "list"], help="Operation to perform")
    parser.add_argument("batch_file_path_or_id", help="Path to the batch file (for submit operation) or batch ID (for other operations)")
    parser.add_argument("--limit", type=int, default=10, help="Limit for listing batches")

    args = parser.parse_args()

    if args.operation == "submit":
        input_file_id = upload_batch_file(args.batch_file_path_or_id)
        print("Batch file uploaded successfully. File ID:", input_file_id)
        result = create_batch(input_file_id, description="".join(Path(args.batch_file_path_or_id).name.split(".")[:-1]))
        print("Batch created successfully. Batch ID:", result["id"])
    elif args.operation == "cancel":
        result = cancel_batch(args.batch_file_path_or_id)
        print("Batch canceled successfully.")
    elif args.operation == "progress":
        result = check_progress(args.batch_file_path_or_id)
        print("Batch progress:")
        print(json.dumps(result, indent=2))
    elif args.operation == "results":
        result = retrieve_results(args.batch_file_path_or_id)
        print(result)
    elif args.operation == "list":
        result = list_batches(args.limit)
        print("List of batches:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
