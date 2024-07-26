def analyze_response(response_text, task=None):
    # most of the early promots will have the action mentioned anywhere, but eventually we ask for the action at the end
    assert task is not None, "Please provide the task type (progress, success, feasibility)"
        
    try:
        if task == "progress":
            if "Keep Going" in response_text and not "Reset" in response_text and not "Unsure" in response_text:
                action = "Keep Going"
            elif "Reset" in response_text and not "Keep Going" in response_text and not "Unsure" in response_text:
                action = "Reset"
            elif "Unsure" in response_text and not "Keep Going" in response_text and not "Reset" in response_text:
                action = "Unsure"
            else:
                action = None
        elif task == "success":
            if "Succeeded" in response_text and not "Failed" in response_text and not "Unsure" in response_text:
                action = "Succeeded"
            elif "Failed" in response_text and not "Succeeded" in response_text and not "Unsure" in response_text:
                action = "Failed"
            elif "Unsure" in response_text and not "Succeeded" in response_text and not "Failed" in response_text:
                action = "Unsure"
            else:
                action = None
        elif task == "feasibility":
            if "Feasible" in response_text and not "Infeasible" in response_text and not "Unsure" in response_text:
                action = "Feasible" 
            elif "Infeasible" in response_text and not "Feasible" in response_text and not "Unsure" in response_text:
                action = "Infeasible"
            elif "Unsure" in response_text and not "Feasible" in response_text and not "Infeasible" in response_text:
                action = "Unsure"
            else:
                action = None
                
        # if action is None, we will look into the last lines only now and be more flexible
        line_to_check = -1
        while action is None:
            if task == "progress":
                if "keep going" in response_text.split('\n')[line_to_check].lower():
                    action = "Keep Going"
                elif "reset" in response_text.split('\n')[line_to_check].lower():
                    action = "Reset"
                elif "unsure" in response_text.split('\n')[line_to_check].lower():
                    action = "Unsure"
            elif task == "success":
                if "succeeded" in response_text.split('\n')[line_to_check].lower():
                    action = "Succeeded"
                elif "failed" in response_text.split('\n')[line_to_check].lower():
                    action = "Failed"
                elif "unsure" in response_text.split('\n')[line_to_check].lower():
                    action = "Unsure"
            elif task == "feasibility":
                if "infeasible" in response_text.split('\n')[line_to_check].lower():
                    action = "Infeasible"
                elif "feasible" in response_text.split('\n')[line_to_check].lower():
                    action = "Feasible"
                elif "unsure" in response_text.split('\n')[line_to_check].lower():
                    action = "Unsure"
            line_to_check -= 1
            if line_to_check < -3:
                break
        reasoning = response_text
    except Exception as e:
        print(f"Failed to analyze response: {e}")
        action = None
        reasoning = None
    return action, reasoning


# returns able_to_eval, correct
def evaluate_action_correctness(action, task, groundtruth):
    if task == "progress":
        if "Keep Going" in str(action):
            if "success" in groundtruth:
                return 1, 1
            return 0, 1
        elif "Reset" in str(action):
            if "wrong" in groundtruth or "fail" in groundtruth:
                return 1, 1
            return 0, 1
    elif task == "success":
        if "Succeeded" in str(action):
            if "success" in groundtruth:
                return 1, 1
            return 0, 1
        elif "Failed" in str(action):
            if "wrong" in groundtruth or "fail" in groundtruth:
                return 1, 1
            return 0, 1
    elif task == "feasibility":
        if "Feasible" in str(action):
            if "success" in groundtruth:
                return 1, 1
            return 0, 1
        elif "Infeasible" in str(action):
            if "wrong" in groundtruth:
                return 1, 1
            return 0, 1
    return 0, 0

