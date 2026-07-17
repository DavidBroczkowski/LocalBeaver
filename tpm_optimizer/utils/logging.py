import time
from enum import Enum
from pathlib import Path

class MsgType(Enum):
    INFO = "[INFO] "
    DEBUG = "[DEBUG] "
    ERROR = "[ERROR] "
    WARN = "[WARN] "


class Logger:

    def __init__(self, path: Path):
        self.path = path
        path.touch()
    
    def log(self, msg: str, type: MsgType = MsgType.INFO, console: bool = True):
        """
        Logs a message to the log file at path

        Inputs:
            - msg: the message to log
            - console: a boolean value, when True, prints the message to the console
        """
        suffix = "[" + time.strftime("%m/%d/%Y %H:%M:%S %z", time.localtime()) + "] "
        msg = suffix + type.value + msg
        
        if console:
            print(msg)

        with open(self.path, 'a', encoding='utf-8') as log_file:
            log_file.write(msg + "\n")