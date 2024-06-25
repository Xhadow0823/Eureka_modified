from typing import List
import re
import time
from pynput.keyboard import Listener

def select(options: List[str]):
    '''
    list all options and prompt user to select and return the selected idx \n
    (1) the first action -> this will return 0 \n
    (2) the second action -> this will return 1
    '''
    result: int = None

    for _ in range(99):
        print("Please select:")
        for i, option in zip( range(1, len(options)+1), options):
            print(f"({i})\t{option}")
        
        try:
            raw_input: str = input()
            selected_idx: int = int( re.search(r'\d+', raw_input).group(0) ) - 1
            if selected_idx < 0 or selected_idx >= len(options):
                raise IndexError("index out of options")
            # print(f"you just selected the ({selected_idx+1}) -> {options[selected_idx]}")  # toggle this for debug
            result = selected_idx
        except Exception as e:
            print("Something wrong, please try again. Error msg:")
            print(e)
            print()
            continue

        return result
    raise Exception("bad select over max try")

def what_do_you_want():
    result_pack = {
        "user_feedback": None,
        "automatically": False,
        "quit": False
    }
    OPTIONS: List[str] = [
        "provide user feedback",
        "feedback automatically",
        "FORCE QUIT",
    ]
    idx = select(OPTIONS)

    if OPTIONS.index("provide user feedback") == idx:
        print("Please input your feedback")
        result_pack["user_feedback"] = input()
    elif OPTIONS.index("feedback automatically") == idx:
        result_pack["automatically"] = True
    elif OPTIONS.index("FORCE QUIT") == idx:
        result_pack["quit"] = True
    else:
        assert False, "IDK, WTF"
    
    return result_pack

def wait_any_key_for(sec: int, action_to_do: str="do nothing") -> bool:
    'will wait if any key pressed in seconds, return bool'
    is_key_pressed = False
    
    def on_press(key):
        nonlocal is_key_pressed
        is_key_pressed = True
        # print('{0} pressed'.format(key))  # toggle this line to debug
        return False # return false to terminate this hook
    
    with Listener(on_press=on_press) as listener:
        # listener.start()
        for i in reversed(range(1, sec+1)):
            print(f"\rpress any key to {action_to_do} ({i}s)\t", end='')
            if is_key_pressed:
                break
            time.sleep(0.5)
            if is_key_pressed:
                break
            time.sleep(0.5)
        listener.stop()
    print()
    return is_key_pressed
    

if __name__ == "__main__":

    # idx = select([
    #     "apple",
    #     "waxapple",
    #     "pineapple",
    #     "apple",
    #     "waxapple",
    #     "pineapple",
    # ])
    # print(f"{idx} is your idx")

    
    is_to_select = wait_any_key_for(5)
    if is_to_select:
        print("What do you want")
    else:
        print("BYE")
    
    exit()
    
    res = what_do_you_want()
    print( res )


    pass