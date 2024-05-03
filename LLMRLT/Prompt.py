from argparse import ArgumentParser
from typing import Dict, List
from utils import get_current_file_dir, file_to_string
from Types import PromptConfig

class Prompt:
    # initial: Dict[str, str] = None
    task_name: str = None
    # '''this is a Dict that contains some initial prompts'''
    initial_system: str = ""
    '''the initial system prompt for the task, this will not be changed after __init__'''
    initial_user: str = ""
    '''the initial user prompt for the task, this will not be changed after __init__'''

    def __init__(self, task_name: str, prompt_config: PromptConfig={}):
        self.task_name = task_name

        prompt_dir = f'prompts/'
        initial_system_path   = f'{prompt_dir}/initial_system.txt'   if "initial_system" not in prompt_config else prompt_config["initial_system"]
        code_output_tip_path  = f'{prompt_dir}/code_output_tip.txt'  if "code_output_tip" not in prompt_config else prompt_config["code_output_tip"]
        initial_user_path     = f'{prompt_dir}/initial_user.txt'     if "initial_user" not in prompt_config else prompt_config["initial_user"]
        reward_signature_path = f'{prompt_dir}/reward_signature.txt' if "reward_signature" not in prompt_config else prompt_config["reward_signature"]
        initial_system   = file_to_string(f'{get_current_file_dir()}/' + initial_system_path)
        code_output_tip  = file_to_string(f'{get_current_file_dir()}/' + code_output_tip_path)
        initial_user     = file_to_string(f'{get_current_file_dir()}/' + initial_user_path)
        reward_signature = file_to_string(f'{get_current_file_dir()}/' + reward_signature_path)

        policy_feedback_path = f'{prompt_dir}/policy_feedback.txt' if "policy_feedback" not in prompt_config else prompt_config["policy_feedback"]
        code_feedback_path   = f'{prompt_dir}/code_feedback.txt'   if "code_feedback" not in prompt_config else prompt_config["code_feedback"]
        execution_error_feedback_path = f'{prompt_dir}/execution_error_feedback.txt' if "execution_error_feedback" not in prompt_config else prompt_config["execution_error_feedback"]
        policy_feedback          = file_to_string(f'{get_current_file_dir()}/' + policy_feedback_path)
        code_feedback            = file_to_string(f'{get_current_file_dir()}/' + code_feedback_path)
        execution_error_feedback = file_to_string(f'{get_current_file_dir()}/' + execution_error_feedback_path)
        
        task_dir = f'{get_current_file_dir()}/tasks/{task_name}/'
        task_obs  = file_to_string(f"{task_dir}/obs.py")
        task_desc = file_to_string(f"{task_dir}/desc.txt")

        self.initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
        self.initial_user   = initial_user.format(task_obs_code_string=task_obs, task_description=task_desc)
    
    def get_prompts(self):
        ''' just for testing'''
        return [
            { "role": "system", "content": self.initial_system },
            { "role": "user", "content": self.initial_user },
        ]


if __name__ == "__main__":
    task = "FrankaLift"  # global, only for testing

    p = Prompt(task, prompt_config={
        "initial_system": "fuck/1.txt",
        "code_output_tip": "fuck/2.txt",
        "initial_user": "fuck/3.txt",
        "reward_signature": "fuck/4.txt",
        "policy_feedback": "fuck/3.txt",
        "code_feedback": "fuck/3.txt",
        "execution_error_feedback": "fuck/3.txt",
    })

    from Logger import Logger
    prompt_logger = Logger(task_name=task)
    logger = prompt_logger.getLogger()
    logger.debug( p.get_prompts() )

    
    pass