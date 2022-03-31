"""
Logger class for all chunc classes.
"""
import warnings
import logging
import platform
import traceback
import socket
import re
import uuid
import psutil
import os

logging_level = {
    'debug':    logging.DEBUG,
}

logging_output = [
    'console',
    'file',
    'both',
]

warning_list = {
    'deprecation':  DeprecationWarning,
    'import':       ImportWarning,
    'resource':     ResourceWarning,
    'user':         UserWarning
}

error_list = {
    'attribute':    AttributeError,
    'index':        IndexError,
    'file':         FileExistsError,
    'memory':       MemoryError,
    'value':        ValueError,
}

class Logger:
    """
    """
    def __init__(self,
        name:   str='default',
        level:  str='debug',
        output: str='file',
        output_file:str='',
        file_mode:  str='a',
    ):
        # check for mistakes
        if level not in logging_level.keys():
            raise ValueError(f"Logging level {level} not in {logging_level}.")
        if output not in logging_output:
            raise ValueError(f"Logging handler {output} not in {logging_output}.")

        # create the logging directory
        if not os.path.isdir('.logs'):
            os.mkdir('.logs')

        # use the name as the default output file name
        self.name = name
        self.level = logging_level[level]
        self.output = output
        if output_file == '':
            self.output_file = name
        else:
            self.output_file = output_file
        self.file_mode = file_mode

        # create logger
        self.logger = logging.getLogger(self.name)

        # set level
        self.logger.setLevel(self.level)

        # set format
        self.dateformat = '%H:%M:%S'
        self.console_formatter = logging.Formatter('[%(levelname)s] [%(name)s]: %(message)s')
        self.file_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',self.dateformat)

        # create handler
        if self.output == 'console' or self.output == 'both':
            self.console = logging.StreamHandler()
            self.console.setLevel(self.level)
            self.console.setFormatter(self.console_formatter)
            self.logger.addHandler(self.console)
        if self.output == 'file' or self.output == 'both':
            self.file = logging.FileHandler('.logs/'+self.output_file+'.log', mode=self.file_mode)
            self.file.setLevel(self.level)
            self.file.setFormatter(self.file_formatter)
            self.logger.addHandler(self.file)

    def info(self,
        message:    str,
    ):
        """ Output to the standard logger "info" """
        return self.logger.info(message)
    
    def debug(self,
        message:    str,
    ):
        """ Output to the standard logger "debug" """
        return self.logger.debug(message)

    def warning(self,
        message:    str,
        warning_type:str='user',
    ):
        """ Output to the standard logger "warning" """
        formatted_lines = traceback.format_stack()[-2]
        if warning_type not in warning_list.keys():
            warning_type = 'user'
        warnings.warn(f"traceback: {formatted_lines}\nerror: {message}", warning_list[warning_type])
        if self.output == 'file':
            return self.logger.warning(f"traceback: {formatted_lines}\nerror: {message}")
        return

    def error(self,
        message:    str,
        error_type: str='value',
    ):
        """ Output to the standard logger "error" """
        formatted_lines = str(traceback.format_stack()[-1][0])
        if error_type not in error_list.keys():
            error_type = 'value'
        if self.output == 'file':
            self.logger.error(f"traceback: {formatted_lines}\nerror: {message}")
        raise error_list[error_type](f"traceback: {formatted_lines}\nerror: {message}")
    
    def get_system_info(self):
        info={}
        try:
            info['platform']=platform.system()
            info['platform-release']=platform.release()
            info['platform-version']=platform.version()
            info['architecture']=platform.machine()
            info['hostname']=socket.gethostname()
            info['ip-address']=socket.gethostbyname(socket.gethostname())
            info['mac-address']=':'.join(re.findall('..', '%012x' % uuid.getnode()))
            info['processor']=platform.processor()
            info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"   
        except Exception as e:
            self.logger.error(f"Unable to obtain system information: {e}.")
        return info

# create global logger
chunc_logger = Logger(
    "chunc",
    output="both",
    file_mode="w"
)