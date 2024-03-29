# read the file nohup.out and look for all lines that start with "No initial state found for this task", save all these lines to a list
# open the file from the same directory as this script
with open('nohup.out') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    filtered_lines = [line for line in lines if line.startswith("No initial state found for this task")]
    # strip these lines from "No initial state found for this task " and ". We can use the first initial state in the dataset."
    filtered_lines = [line.replace("No initial state found for this task ", "").replace(". We can use the first initial state in the dataset.", "") for line in filtered_lines]
    # convert these lines to a dictionary with the count of each duplicate
    filtered_tasks = {}
    for line in filtered_lines:
        if line in filtered_tasks:
            filtered_tasks[line] += 1
        else:
            filtered_tasks[line] = 1
    # filtered_tasks = set(filtered_lines)
    print(filtered_tasks)