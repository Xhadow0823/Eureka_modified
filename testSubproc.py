import subprocess
import time

def get_conda_env_name():
    import os
    return os.environ["CONDA_DEFAULT_ENV"] if "CONDA_DEFAULT_ENV" in os.environ else None

print( "current conda env: " + get_conda_env_name() )
assert (get_conda_env_name() == "rlgpu"), "use rlgpu for conda env please"


exit()
# command = ["pwd;", "date;", "sleep", "2"]
# process = subprocess.Popen(["gnome-terminal", "--", "bash", "-c", *command], stderr=subprocess.STDOUT, stdout=subprocess.PIPE)  # not working
# process = subprocess.Popen(["gnome-terminal",  "--", "python3", "isaacgymenvs/count_down.py", "-t", "2"], stderr=subprocess.STDOUT, stdout=subprocess.PIPE)  # good 
process = subprocess.Popen("gnome-terminal -- bash -c 'pwd; sleep 6;'", stdin=None, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True)  # good
# NOTE: communicate and poll and wait not work on gnome-terminal

# print( process.communicate() )
print( process.poll() )
time.sleep(3)
print( process.poll() )
time.sleep(4)
print( process.poll() )
# print( process.wait() )