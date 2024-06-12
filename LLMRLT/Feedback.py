from typing import List
import re

def select(options: List[str]):
    '''list all options and prompt user to select and return the selected idx'''
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

    
    res = what_do_you_want()
    print( res )


    pass