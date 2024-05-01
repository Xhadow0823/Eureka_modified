from Prompt import Prompt
from Chat import Chat
from Logger import Logger

task = 'FrankaLift'

prompt = Prompt(task_name=task, prompt_config={
    "code_output_tip": "prompts/FSM/FSM_code_output_tip.txt",
    "reward_signature": "prompts/FSM/FSM_reward_signature.txt"
})
chat = Chat()
chat_logger = Logger(task_name=task)
logger = chat_logger.getLogger()

logger.info("DEMO START")

logger.info(f"SYSTEM PROMPT: {prompt.initial_system}")
chat.set_system_content(prompt.initial_system)

logger.info(f"USER PROMPT: {prompt.initial_user}")
resp = chat.chat(prompt.initial_user)
logger.info(f"LLM RESPONSE: {resp}")

logger.info(f"TOTAL TOKEN: {chat.get_total_usage()}")
logger.info("DEMO END")
chat_logger.save_conversation( chat.get_conversation() )  # save as json