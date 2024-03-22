import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
import openai
import re
import subprocess
from pathlib import Path
import shutil
import time 

from utils.misc import * 
from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import *

EUREKA_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")

    openai.api_key = os.getenv("OPENAI_API_KEY")

    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.env_name.lower()
    env_parent = 'isaac' if f'{env_name}.py' in os.listdir(f'{EUREKA_ROOT_DIR}/envs/isaac') else 'dexterity'
    task_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py'
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py'
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string  = file_to_string(task_file)
    task_obs_code_string  = file_to_string(task_obs_file)
    output_file = f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    task_code_string = task_code_string.replace(task, task+suffix)

    
    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample if "gpt-3.5" in model else 4

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=cfg.temperature,
                        n=chunk_size
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur["choices"])
            prompt_tokens = response_cur["usage"]["prompt_tokens"]
            total_completion_token += response_cur["usage"]["completion_tokens"]
            total_token += response_cur["usage"]["total_tokens"]

        # show every sample responses
        for s in range(cfg.sample):
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[s]["message"]["content"] + "\n")

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        

        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["message"]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
            
            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')
        
        break  # break the iteration loop

if __name__ == "__main__":
    main()
