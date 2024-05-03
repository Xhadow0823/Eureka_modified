import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--time", "-t", default=5, type=int)

max_time = parser.parse_args().time
for i in reversed(range(max_time)):
    time.sleep(1)
    print(i)