from Code import Code
import subprocess
import shutil


if __name__ == "__main__":
    task = "FrankaLift2"
    C = Code()
    C.load_env_from_file(f"LLMRLT/tasks/{task}/env.py")
    C.load_obs_function_from_file(f"LLMRLT/tasks/{task}/obs.py")
    C.load_reward_function_from_file("LLMRLT/logs/FrankaLift/24-04-26-13-37-24.json")
    env_code_filepath = f"./LLMRLT/codes/test/{task}.py"
    C.gen_env_code().save(env_code_filepath)
    

    ISAAC_ROOT_DIR = "./isaacgymenvs/isaacgymenvs/"
    eval_log = f"./LLMRLT/codes/test/{task}.log"
    
    shutil.copy(env_code_filepath, ISAAC_ROOT_DIR+f"/tasks/franka_lift2.py")  # TODO: naming problem...
    with open(eval_log, 'w') as f:
        process = subprocess.Popen(['python', '-u', f'{ISAAC_ROOT_DIR}/train.py',  
                                    'hydra/output=subprocess',
                                    f'task={task}',
                                    f'headless={True}', 'force_render=False',
                                    f'max_iterations={100}'],
                                    stdout=f, stderr=f)
    print( process.communicate() )

    pass