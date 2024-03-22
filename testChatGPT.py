import yaml
import json
import logging
import os
import openai
import re
from pathlib import Path
from argparse import ArgumentParser
import time
from typing import Dict, List



class ChatLogger:
    _all_messages = []
    '''the buffer that storing chat history'''
    _token_usage = None

    def __init__(self):
        self.reset()

    def reset(self):
        self._all_messages = []
        self._token_usage = {
            "total_tokens": 0
        }
    def log(self, message_pairs: List[Dict[str, str]]):
        # assert(len(message_pair) == 2)
        self._all_messages.extend(message_pairs)
    def update_token_usage(self, usage_dict: Dict):
        self._token_usage["total_tokens"] += usage_dict["total_tokens"]
    def get_token_usage(self):
        return self._token_usage
    def get_all_messages(self, last_n_times=None)->List[Dict[str, str]]:
        return self._all_messages
    def show(self):
        pass

class Chat:
    MAX_ATTEMPT = 1000
    # GPT_MODEL="gpt-4-0314"
    GPT_MODEL="gpt-4-0125-preview"
    _chatLogger = None
    '''the helper for saving chat information'''
    temperature = 1.0
    chunk_size = 1
    '''chunk always be 1 in current version'''

    system_content = None

    logger = None

    def __init__(self):
        self.logger = logging.getLogger("Chat")
        # logging.basicConfig(level=logging.INFO)
        logging.basicConfig(level=logging.WARNING)
        openai.api_key = os.getenv("OPENAI_API_KEY")        
        self._chatLogger = ChatLogger()
        pass
    
    def set_system_content(self, content=None):
        if content == None:
            self.system_content = ""
        else:
            self.system_content = content
        system_content_pair = {"role": "system", "content": self.system_content}
        self._chatLogger.log([
            system_content_pair
        ])
        return system_content_pair

    def chat(self, content):
        if self.system_content == None:
            self.set_system_content()

        messages = self._chatLogger.get_all_messages()
        user_content_pair = {"role": "user", "content": content}
        messages += [
            user_content_pair
        ]
        self.logger.info(user_content_pair)

        total_samples = 0
        response = None
        for attempt in range(self.MAX_ATTEMPT):
            try:
                response = openai.ChatCompletion.create(
                    model=self.GPT_MODEL,
                    messages=messages,
                    temperature=self.temperature,
                    n=self.chunk_size
                )
                total_samples += self.chunk_size
                break
            except Exception as e:
                if attempt >= 10:
                    self.chunk_size = max(int(self.chunk_size / 2), 1)
                    print("Current Chunk Size", self.chunk_size)
                self.logger.warning(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
        if response == None:
            self.logger.error(f"reach max attempt {self.MAX_ATTEMPT}")
            return None
        
        assistant_content_pair = response["choices"][0]["message"]
        self.logger.info(assistant_content_pair)
        self._chatLogger.log([user_content_pair, assistant_content_pair])
        self._chatLogger.update_token_usage(response["usage"])
        self.logger.info(f"used tokens: {response['usage']}")

        return assistant_content_pair["content"]
    
    def interactive_chat(self):
        self.logger.warning(f"Interactive chat mode. typing exit to exit this mode")

        system_content = input("enter the system prompt:")
        self.set_system_content(system_content)

        MAX_CHAT_TIMES = 100
        chat_id = 0
        while True:
            user_content = input("Enter the user prompt:")
            if user_content.strip().lower() == "exit":
                self.logger.warning(f"Leaving interactive chat mode...")
                break

            assistant_response = self.chat(user_content)
            # self.logger.info(f"response: \n {assistant_response}")
            print(f"response: \n {assistant_response}\n=====")
            if assistant_response is None:
                chat_id = MAX_CHAT_TIMES
            chat_id += 1
            if chat_id >= MAX_CHAT_TIMES:
                self.logger.warning(f"Leaving interactive chat mode...")
                break
        self.logger.info(f"Total token: {self._chatLogger.get_token_usage()}")
        


def main():
    # c = Chat()
    # c.set_system_content("you are a translator, you translate anything into traditional chinese")
    # print(c.chat("how is your cold today?"))

    ic = Chat()
    ic.interactive_chat()

    

if __name__ == "__main__":
    main()