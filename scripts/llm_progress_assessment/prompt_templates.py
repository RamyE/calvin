# non-text placeholders
GRID_IMAGE_PLACEHOLDER = 'GRID_IMAGE_PLACEHOLDER'
SEQUENCE_IMAGES_PLACEHOLDER = 'SEQUENCE_IMAGES_PLACEHOLDER'
VIDEO_PLACEHOLDER = 'VIDEO_PLACEHOLDER'

# text placeholders
TASK_NAME_PLACEHOLDER = 'TASK_NAME_PLACEHOLDER'
IMAGE_COUNT_PLACEHOLDER = 'IMAGE_COUNT_PLACEHOLDER'

# progress assessment prompts
PROGRESS_PROMPT_TEMPLATE_1 = [
    f'According to the attached image containing {IMAGE_COUNT_PLACEHOLDER} video frames showin in chronological order from top to bottom and from left to right, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}" or should it reset and try again from scratch. Please only answer "Reset" or "Keep Going" in one lines and briefly explain the reasoning in another line.',
    GRID_IMAGE_PLACEHOLDER
    ]

PROGRESS_PROMPT_TEMPLATE_2 = [
        # f"""According to the attached images containing 5 video frames in order showing the robotic arm acting towards achieving a task, is the robot making correct progress towards the task "{image_name}" or should it reset and try again from scratch.
    
        # Please only answer "Reset" or "Keep Going" or "Unsure" in one line and briefly explain the reasoning in another line.
        
        # Do not make any assumptions and focus on the accuracy of the task instruction including the accuracy of colors. Please also consider that the robotic arm may still be on its way to perform the task."""
    
        f"""Please analyze the sequence of images provided, where each frame shows a different stage of a robotic arm performing the task of "{TASK_NAME_PLACEHOLDER}". For each frame (total {IMAGE_COUNT_PLACEHOLDER} frames), assess whether the robotic arm is correctly executing the task as per the instructions. Consider the accuracy of the arm's movements, the precision of object handling, and adherence to task instructions including color accuracy. Provide your assessment for each frame along with a brief explanation. Conclude with an overall recommendation: 'Reset', 'Keep Going', or 'Unsure', based on the collective analysis of all frames. Consider that the robot may still be early in its progress and has not make any significant progress towards the task yet. Please be very brief."""
    ] + [
        SEQUENCE_IMAGES_PLACEHOLDER
    ]

PROGRESS_PROMPT_TEMPLATE_3 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} video frames in order showing the robotic arm acting, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}" or should it reset and try again from scratch (if it is completing a different task or acting erroneously) or is it unclear to you.

    Please only answer "Reset" or "Keep Going" or "Unsure" in one line and briefly explain the reasoning in another line. Please think very carefully before making a decision as it is very costly to make an incorrect decision.
    """
    ]

PROGRESS_PROMPT_TEMPLATE_4 = [
    VIDEO_PLACEHOLDER,
    f"""According to the attached video showing the robotic arm acting, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}" or should it reset and try again from scratch (if it is completing a different task or acting erroneously) or is it unclear to you.

    Please only answer "Reset" or "Keep Going" or "Unsure" in one line and briefly explain the reasoning in another line. Please think very carefully before making a decision as it is very costly to make an incorrect decision.
    """
    ]

PROGRESS_PROMPT_TEMPLATE_5 = [
    VIDEO_PLACEHOLDER,
    f"""According to the attached video showing the robotic arm acting, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}" or should it reset and try again from scratch (if it is completing a different task or acting erroneously) or is it unclear to you.
    Note that the robot may still be early in its progress and has not made any significant progress towards the task yet, but also it is important to make sure that the state of the environments allows for the task to be completed or a reset is needed otherwise. Consider all items in the environment and how they interact with the task. Please think very carefully before making a decision as it is very costly to make an incorrect decision. Make sure to answer "Unsure" if you are not sure and only make a decision if you are confident about your understanding of the scene.
    Please only answer "Reset" or "Keep Going" or "Unsure" in one line and explain the reasoning in another line.
    """
    ]

PROGRESS_PROMPT_TEMPLATE_6 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}" or should it reset and try again from scratch (if it is completing a different task or acting erroneously) or is it unclear to you?
    Note that the robot may still be early in its progress and has not made any significant progress towards the task yet, but also it is important to make sure that the state of the environments allows for the task to be completed or a reset is needed otherwise. Consider all items in the environment and how they relate with the task. It may help to think about the colors and the objects and how their states change throughout the frames. Please think very carefully before making a decision as it is very costly to make an incorrect decision. Make sure to answer "Unsure" if you are not sure and only make a decision if you are confident about your understanding of the scene.
    Please only answer "Reset" or "Keep Going" or "Unsure" in one line and explain the reasoning in another line.
    """
    ]

PROGRESS_PROMPT_TEMPLATE_7 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}" or should it reset and try again from scratch (if it is completing a different task or acting erroneously) or is it unclear to you?
    Please only answer "Reset" or "Keep Going" or "Unsure" in one line and explain the reasoning in another line.
    """
    ]

PROGRESS_PROMPT_TEMPLATE_8 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the previously attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}" or should it reset and try again from scratch (if it is completing a different task or acting erroneously) or is it unclear to you?
    Please start by thinking loudly about the task and the actions of the robot in each frame, mainly by considering the difference between frames. Lastly, provide a conclusion based on the collective analysis of all frames saying "Reset" or "Keep Going" or "Unsure".
    """
    ]

PROGRESS_PROMPT_TEMPLATE_9 = [
    f"""According to the following attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}" or should it reset and try again from scratch (if it is completing a different task or acting erroneously) or is it unclear to you?
    Please start by thinking loudly about the task and the actions of the robot in each frame, mainly by considering the difference between frames. Lastly, provide a conclusion based on the collective analysis of all frames saying "Reset" or "Keep Going" or "Unsure".
    """,
    SEQUENCE_IMAGES_PLACEHOLDER
    ]

PROGRESS_PROMPT_TEMPLATE_10 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}" or should it reset and try again from scratch (if it is completing a different task or acting erroneously) or is it unclear to you?
    Please start by thinking loudly about the task and the actions of the robot in each frame, mainly by considering the difference between frames. Lastly, provide a conclusion based on the collective analysis of all frames saying "Reset" or "Keep Going" or "Unsure". Here are the images again, please go over your analysis again to make sure you have a clear understanding of the task and the robot's actions.
    """,
    SEQUENCE_IMAGES_PLACEHOLDER
    ]

PROGRESS_PROMPT_TEMPLATE_11 = [
    VIDEO_PLACEHOLDER,
    f"""According to the previously attached video showing the robotic arm acting through a gripper end-effector, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}" or should it reset and try again from scratch (if it is completing a different task or acting erroneously) or is it unclear to you?
    Please start by thinking loudly about the task and the actions of the robot in each frame, mainly by considering the difference between frames. Lastly, provide a conclusion based on the collective analysis of all frames saying "Reset" or "Keep Going" or "Unsure".
    """
    ]

PROGRESS_PROMPT_TEMPLATE_12 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}"  or should it reset and try again from scratch (if it is completing a different task or acting erroneously) or is it unclear to you?
    Looking at the scene, first think loudly about how the task could be achieved. Do not make any assumptions and focus on the task instruction and how it relates to everything in the scene.
    Think loudly in detail about the needs of the task and the actions of the robot in each frame. Lastly, provide a conclusion based on the collective analysis of all frames saying "Reset" or "Keep Going" or "Unsure".
    """
    ]

PROGRESS_PROMPT_TEMPLATE_13 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}"  or should it reset and try again from scratch (if it is completing a different task or acting erroneously) or is it unclear to you?
    Looking at the scene, first think loudly about how the task could be achieved. Do not make any assumptions and focus on the task instruction and how it relates to everything in the scene.
    Think loudly in detail about the needs of the task and the actions of the robot in each frame (think about each frame individually first and then later you can consider the difference between frames). Lastly, provide a conclusion based on the collective analysis of all frames saying "Reset" or "Keep Going" or "Unsure".
    """
    ]

PROGRESS_PROMPT_TEMPLATE_14 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}"  or should it reset and try again from scratch (if it is completing an incorrect task or acting erroneously) or is it unclear to you?
    Looking at the scene, first think loudly about what everything in the scene is and how it could be manipulated. Next, think how the task could be achieved. Then, think in detail about the needs of the task and the actions of the robot in each frame individually.
    Lastly, provide a conclusion based on the collective analysis of all frames saying "Reset" or "Keep Going" or "Unsure".
    Do not make any assumptions and focus on the task instruction and how it relates to everything in the scene.
    """
    ]

PROGRESS_PROMPT_TEMPLATE_15 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}"  or should it reset and try again from scratch (if it is completing an incorrect task or acting erroneously) or is it unclear to you?
    Looking at the scene, first think loudly about what everything in the scene is and how it could be manipulated. Next, think how the task could be achieved. Then, think in detail about the needs of the task and the actions of the robot in each frame individually. Lastly, provide a conclusion based on the collective analysis of all frames saying "Reset" or "Keep Going" or "Unsure".
    Do not make any assumptions and focus on the task instruction and how it relates to everything in the scene.
    
    Please structure your answer to include the following sections:
    * Scene Description and object affordances
    * Task Description
    * Detailed analysis of each frame
    * Conclusion
    """
    ]

PROGRESS_PROMPT_TEMPLATE_16 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}"  or should it reset and try again from scratch (if it is completing an incorrect task or acting erroneously) or is it unclear to you?
    
    Please only answer with "Keep Going", "Reset" or "Unsure".
    """
    ]


PROGRESS_PROMPT_TEMPLATE_17 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, is the robot making correct progress towards the task "{TASK_NAME_PLACEHOLDER}"  or should it reset and try again from scratch (if it is completing an incorrect task or acting erroneously) or is it unclear to you?
    Looking at the scene, first think loudly about what everything in the scene is and how it could be manipulated. Next, think how the task could be achieved. Then, think in detail about the needs of the task and the actions of the robot in each frame individually. Then, find problems in your analysis so far. After that, write an improved analysis to address these problems. Lastly, provide a conclusion based on the collective analysis of all frames saying "Reset" or "Keep Going" or "Unsure".
    Do not make any assumptions and focus on the task instruction and how it relates to everything in the scene.
    
    Please structure your answer to include the following sections:
    * Scene Description and object affordances
    * Task Description
    * Detailed analysis of each frame
    * Potential Problems in the current analysis
    * Improved Analysis
    * Conclusion
    """
    ]

# success detection prompt
SUCCESS_PROMPT_TEMPLATE_1 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, has the robot completed the task "{TASK_NAME_PLACEHOLDER}" successfully?
    
    Please start by thinking loudly about the task and the actions of the robot in each frame, mainly by considering the difference between frames. Lastly, provide a conclusion based on the collective analysis of all frames saying "Succeeded" or "Failed"
    """
    ]

SUCCESS_PROMPT_TEMPLATE_2 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, has the robot completed the task "{TASK_NAME_PLACEHOLDER}" successfully?
    
    Please start by thinking loudly about the task and the actions of the robot in each frame, mainly by considering the difference between frames and the goal task "{TASK_NAME_PLACEHOLDER}".
    
    Lastly, provide a conclusion concisely based on the collective analysis saying "Succeeded" or "Failed" or "Unsure" on a separate line.
    """
    ]


SUCCESS_PROMPT_TEMPLATE_3 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, has the robot succeeded or failed in completing the task "{TASK_NAME_PLACEHOLDER}"?
    
    Please start by thinking loudly in detail about the needs of the task and the actions of the robot in each frame. Lastly, provide a conclusion based on the collective analysis of all frames saying "Succeeded" or "Failed" .
    """
    ]

SUCCESS_PROMPT_TEMPLATE_4 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, has the robot succeeded or failed in completing the task "{TASK_NAME_PLACEHOLDER}"?
    Looking at the scene, first think loudly about how the task could be achieved. Do not make any assumptions and focus on the task instruction and how it relates to everything in the scene.
    Think loudly in detail about the needs of the task and the actions of the robot in each frame. Lastly, provide a conclusion based on the collective analysis of all frames saying "Succeeded" or "Failed" .
    """
    ]

SUCCESS_PROMPT_TEMPLATE_5 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached images containing {IMAGE_COUNT_PLACEHOLDER} frames in order showing the robotic arm acting through a gripper end-effector, has the robot succeeded or failed in completing the task "{TASK_NAME_PLACEHOLDER}" or are you unsure?
    Looking at the scene, first think loudly about what everything in the scene is and how it could be manipulated. Next, think how the task could be achieved. Then, think in detail about the needs of the task and the actions of the robot in each frame individually. Lastly, provide a conclusion based on the collective analysis of all frames saying "Succeeded" or "Failed" or "Unsure".
    
    Please structure your answer to include the following sections:
    * Scene Description and object affordances
    * Task Description
    * Detailed analysis of each frame
    * Conclusion
    """
    ]

SUCCESS_PROMPT_TEMPLATE_6 = [
    GRID_IMAGE_PLACEHOLDER,
    f"""According to the image containing the frames in order from top to bottom and from left to right showing the robotic arm acting through a gripper end-effector, has the robot succeeded or failed in completing the task "{TASK_NAME_PLACEHOLDER}" or are you unsure?
    Looking at the scene, first think loudly about what everything in the scene is and how it could be manipulated. Next, think how the task could be achieved. Then, think in detail about the needs of the task and the actions of the robot in each frame individually. Lastly, provide a conclusion based on the collective analysis of all frames saying "Succeeded" or "Failed" or "Unsure".
    
    Please structure your answer to include the following sections:
    * Scene Description and object affordances
    * Task Description
    * Detailed analysis of each frame
    * Conclusion
    """
    ]

FEASIBILITY_PROMPT_TEMPLATE_1 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""According to the attached image showing an environment with a robotic arm, is the task "{TASK_NAME_PLACEHOLDER}" considered feasible in this environment?
    Looking at the scene, first think loudly about what everything in the scene is and how it could be manipulated. Next, think in detail about the needs of the task. Lastly, provide a conclusion based on the collective analysis of all frames saying "Feasible" or "Infeasible" or "Unsure".
    
    Please structure your answer to include the following sections:
    * Scene Description and object affordances
    * Task Description
    * Conclusion
    """
]

FEASIBILITY_PROMPT_TEMPLATE_2 = [
    SEQUENCE_IMAGES_PLACEHOLDER,
    f"""Attached is an image of an environment with a robotic arm, is the task "{TASK_NAME_PLACEHOLDER}" considered feasible in this environment?
    Please analyze the scene and task and provide a conclusion saying "Feasible" or "Infeasible" or "Unsure".
    """
]
    # Do not make any assumptions and focus on the accuracy of the task instruction including the accuracy of colors.