import os
import openai
import time
from typing import Dict, List, TypedDict, Literal
import json

class MessageDict(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class Messages:
    _all_messages: List[MessageDict]
    '''the buffer that storing whole chat history'''
    _max_queue_length: int = -1
    '''the max length of _all_messages the queue'''

    def __init__(self):
        self.reset()
    def reset(self):
        self._all_messages = []
        self._max_queue_length = -1
    def log(self, message_dict_list: List[MessageDict]):
        self._all_messages.extend(message_dict_list)
    def get_all(self)->List[MessageDict]:
        return self._all_messages.copy()
    def load(self, file_path: str):
        conversation: List[MessageDict] = None
        try:
            with open(file_path, 'r') as f:
                conversation = json.load(f)
        except:
            print("Load conversation json file error")
            return
        self.reset()
        self._all_messages = conversation



class UsageDict(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Usage:
    _last: UsageDict = None
    _cumulated: UsageDict = None

    def __init__(self):
        self._cumulated = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    def update(self, usage: UsageDict):
        self._last = usage
        for k in self._cumulated.keys():
            self._cumulated[k] += self._last[k]
        
    def get_last(self) -> UsageDict:
        return self._last
    def get_all(self) -> UsageDict:
        return self._cumulated

class Chat:
    MAX_ATTEMPT = 100
    GPT_MODEL="gpt-4-0125-preview"  # GPT_MODEL="gpt-4-0314"
    
    ##### GPT configs #####
    temperature = 1.0
    chunk_size = 1
    '''chunk always be 1 in current version'''

    ##### Informations about this conversation #####
    _messages: Messages
    '''contain all messages in this conversation'''
    _usage: Usage
    '''log the token usage in this conversation'''
    system_content: str = None
    '''string for the initial system prompt'''

    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self._messages = Messages()
        self._usage = Usage()
    
    def set_system_content(self, content: str=None):
        if content == None:
            self.system_content = ""
        else:
            self.system_content = content
        system_content_dict: MessageDict = {"role": "system", "content": self.system_content}
        self._messages.log([ system_content_dict ])

    def chat(self, content: str) -> str:
        if self.system_content == None:
            self.set_system_content()

        messages = self._messages.get_all()
        user_content_dict: MessageDict = {"role": "user", "content": content}
        messages.append(user_content_dict)

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
                # TODO: ?
                if attempt >= 10:
                    self.chunk_size = max(int(self.chunk_size / 2), 1)
                    print("Current Chunk Size", self.chunk_size)
                self.logger.warning(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
        if response == None:
            print(f"reach max attempt {self.MAX_ATTEMPT}")
            return None
        
        assistant_content_dict: MessageDict = response["choices"][0]["message"]
        self._messages.log([ user_content_dict, assistant_content_dict ])
        self._usage.update(response['usage'])

        return assistant_content_dict["content"]
    
    def get_conversation(self) -> List[Dict]:
        return self._messages.get_all()

    def get_total_usage(self):
        return self._usage.get_all()["total_tokens"]

    def load_conversation(self, file_path: str):
        self._messages.load(file_path)

    def interactive_chat(self):
        print(f"Interactive chat mode. typing exit to exit this mode")

        system_content = input("Enter the system prompt:")
        self.set_system_content(system_content)

        MAX_CHAT_TIMES = 100
        for chat_id in range(MAX_CHAT_TIMES):
            user_content = input("Enter the user prompt:")
            if user_content.strip().lower() == "exit":
                print(f"Leaving interactive chat mode...")
                break
            assistant_response = self.chat(user_content)
            print(f"response: \n {assistant_response}\n=====")
            if assistant_response is None:
                break
        print(f"Total token cost: {self._usage.get_all()['total_tokens']}")

if __name__ == '__main__':
    chat = Chat()
    from Logger import Logger
    
    chat_logger = Logger("catcatmeow")
    
    chat.set_system_content("你是一隻貓，只能用行為與喵來表示想法。當你做出任何用來表示想法的行為時，你將會使用({動作})來表示。舉例來說，你調皮的將花瓶推下桌子，即為(將花瓶推下桌子)。")

    resp = chat.chat("你好，小貓咪。（輕輕撫摸）")
    print( resp )

    resp = chat.chat("你想來點小點心嗎，貓貓")
    print( resp )

    resp = chat.chat("奴才誠懇的請求您展示您的後空翻，貓貓大人")
    print( resp )

    chat_logger.save_conversation( chat.get_conversation() )

    print(f"Total token: {chat.get_total_usage()}")
    pass