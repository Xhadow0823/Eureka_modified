from argparse import ArgumentParser
from typing import Dict, List
from utils import get_current_file_dir, file_to_string

class Prompt:
    # initial: Dict[str, str] = None
    task_name: str = None
    # '''this is a Dict that contains some initial prompts'''
    initial_system: str = ""
    '''the initial system prompt for the task, this will not be changed after __init__'''
    initial_user: str = ""
    '''the initial user prompt for the task, this will not be changed after __init__'''

    def __init__(self, task_name: str):
        self.task_name = task_name

        prompt_dir = f'{get_current_file_dir()}/prompts/'
        initial_system   = file_to_string(f'{prompt_dir}/initial_system.txt')
        code_output_tip  = file_to_string(f'{prompt_dir}/code_output_tip.txt')
        initial_user     = file_to_string(f'{prompt_dir}/initial_user.txt')
        reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')

        policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
        code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
        execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
        
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

    p = Prompt(task)

    from Logger import Logger
    prompt_logger = Logger(task_name=task)
    logger = prompt_logger.getLogger()
    logger.DEBUG( p.get_prompts() )

    
    pass