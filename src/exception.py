import os
import sys
from logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tab=error_detail.exc_info()
    file_name=exc_tab.tb_frame.f_code.co_filename
    line_no=exc_tab.tb_lineno
    error_message = (
        f"error message occured in the script {file_name},"
        f"error in line no {line_no},"
        f"error message: {error}"

        )
    return error_message
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_msg=error_message_detail(error_message,error_detail)
        logging.error(self.error_msg)

    def __str__(self):
        return self.error_msg



