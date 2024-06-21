import logging
import json
from typing import List, Dict
from utils import make_sure_dir_exist, get_current_file_dir, get_timestamp

class Logger:
    '''my logger'''

    __logger: logging.Logger = None
    '''instance of logger from logging module'''
    __is_debug_mode: bool = True
    '''is debug mode or not (for the logger.setLevel(...))'''
    
    task_name: str = None
    log_file_path: str = None
    log_file_name: str = None

    def __init__(self, task_name=None):
        self.__logger = logging.getLogger(__class__.__name__)
        
        formatter = logging.Formatter('%(levelname)s: %(message)s')

        self.task_name = '' if task_name==None else task_name  # using the log dir root if task_name not provided
        if self.__is_debug_mode:
            self.__logger.setLevel(logging.DEBUG)
            stdout_handler = logging.StreamHandler()
            stdout_handler.setFormatter(formatter)
            self.__logger.addHandler(stdout_handler)
        else:
            self.__logger.setLevel(logging.INFO)
        
        self.log_file_path = f"{get_current_file_dir()}/logs/{self.task_name}/"
        self.log_file_name = f"{get_timestamp()}"
        make_sure_dir_exist(self.log_file_path)
        file_handler = logging.FileHandler(f"{self.log_file_path}/{self.log_file_name}.log")
        file_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        
    def getLogger(self) -> logging.Logger:
        return self.__logger
    
    def get_log_path_n_name(self):
        return self.log_file_path, self.log_file_name
    
    def save_conversation(self, conversation: List[Dict]) -> str:
        'save the conversation as a json file and then return the path of json'
        json_file = f"{self.log_file_path}/{self.log_file_name}.json"
        with open(json_file, "w", encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
        return json_file
    

if __name__ == "__main__":
    
    LL = Logger()

    logger = LL.getLogger()

    logger.debug("GG this is DEBUG")
    logger.info("GG this is INFO")
    logger.warning("GG this WARNING")
    logger.error("GG this is ERROR")
    

    pass