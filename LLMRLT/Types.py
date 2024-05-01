from typing import TypedDict

class PromptConfig(TypedDict):
    initial_system: str
    code_output_tip: str
    initial_user: str
    reward_signature: str
    policy_feedback: str
    code_feedback: str
    execution_error_feedback: str

if __name__ == "__main__" :
    
    pc: PromptConfig = {
        "initial_system": "fuck/1.txt",
        "code_output_tip": "fuck/2.txt"
    }
    print(pc)

    pass