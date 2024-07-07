import requests
import json
from constants import *

class APIController:
    def __init__(self):
        # read initial configurations from the json file
        with open('config.json') as json_file:
            data = json.load(json_file)
            self.url = data['base_url']
            
            
    def get(self, endpoint, params=None):
        response = requests.get(self.url + endpoint, params=params)
        return response.json()
    
    def post(self, endpoint, data):
        response = requests.post(self.url + endpoint, data=data)
        return response.json()
    



    # File System Endpoints
    def create_file(self, fileName, extension, content):
        data = {
            "fileName": fileName,
            "extension": extension,
            "content": content
        }
        return self.post(FILE_SYSTEM_CREATE_FILE, data)
    
    def create_directory(self, name):
        data = {
            "name": name
        }
        return self.post(FILE_SYSTEM_CREATE_DIRECTORY, data)
    
    def copy_file(self, source, destination):
        data = {
            "source": source,
            "destination": destination
        }
        return self.post(FILE_SYSTEM_COPY_FILE, data)
    
    def copy_directory(self, source, destination):
        data = {
            "source": source,
            "destination": destination
        }
        return self.post(FILE_SYSTEM_COPY_DIRECTORY, data)
    
    
    def delete(self, source):
        data = {
            "source": source
        }
        return self.post(FILE_SYSTEM_DELETE, data)
    
    def rename(self, source, destination):
        data = {
            "source": source,
            "destination": destination
        }
        return self.post(FILE_SYSTEM_RENAME, data)
    
    def save(self):
        data = {}
        return self.post(FILE_SYSTEM_SAVE, data)
    
    def get_files(self):
        return self.post(FILE_SYSTEM_GET_FILES, data={})
    
    # IDE Endpoints
    def go_to_line(self, line, character):
        data = {
            "line": line,
            "character": character
        }
        return self.post(IDE_GO_TO_LINE, data)
    
    def go_to_file(self, path):
        data = {
            "path": path
        }
        return self.post(IDE_GO_TO_FILE, data)
    
    def focus_terminal(self):
        return self.get(IDE_FOCUS_TERMINAL)
    
    def undo(self):
        return self.get(IDE_UNDO)
    
    def new_terminal(self):
        return self.get(IDE_NEW_TERMINAL)
    
    def kill_terminal(self):
        return self.get(IDE_KILL_TERMINAL)
    
    def cut(self):
        return self.get(IDE_CUT)
    
    def copy(self):
        return self.get(IDE_COPY)
    
    def paste(self):
        return self.get(IDE_PASTE)
    
    def redo(self):
        return self.get(IDE_REDO)
    
    def select(self, line):
        data = {
            "line": line
        }
        return self.get(IDE_SELECT, data)
    
    def find(self, line):
        data = {
            "line": line
        }
        return self.get(IDE_FIND, data)
    
    def run_notebook(self):
        return self.get(IDE_RUN_NOTEBOOK)
    
    def run_python_file(self, path):
        data = {
            "path": path
        }
        return self.post(IDE_RUN_PYTHON_FILE, data)
    
    def run_notebook_cell(self,data):
        return self.get(IDE_RUN_NOTEBOOK_CELL, data)
    
    def select_kernel(self, data):
        return self.post(IDE_SELECT_KERNEL, data)
    
    # Git Endpoints
    def discard(self):
        return self.get(GIT_DISCARD_ENDPOINT)
    
    def pull(self):
        return self.get(GIT_PULL_ENDPOINT)
    
    def stage(self):
        return self.get(GIT_STAGE_ENDPOINT)
    
    def stash(self):
        return self.get(GIT_STASH_ENDPOINT)
    
    # Code Execution Endpoints
    def try_except(self, tryBody, exception, exceptionInstance, exceptBody):
        data = {
            "tryBody": tryBody,
            "exception": exception,
            "exceptionInstance": exceptionInstance,
            "exceptBody": exceptBody
        }
        return self.post(CODE_TRY_EXCEPT, data)    
    
    def declare_constant(self, name, value):
        data = {
            "name": name,
            "value": value
        }
        return self.post(CODE_DECLARE_CONSTANT, data)

    def declare_variable(
        self, 
        variable_name, 
        variable_value, 
        variable_type=None, 
    ):
        data = {
            "variable_name": variable_name,
            "variable_value": variable_value,
        }
        if variable_type:
            data["variable_type"] = variable_type
        return self.post(CODE_DECLARE_VARIABLE, data)

    def write_file(
        self,
        path,
        content
    ):
        data = {
            "path": path,
            "content": content
        }
        return self.post(CODE_WRITE_FILE, data)
    
        
    def read_file(
        self,
        path,
        variable = None
    ):
        data = {
            "path": path
        }
        if variable:
            data["variable"] = variable
        return self.post(CODE_READ_FILE, data)
    
    def block_comment(
        self,
        content
    ):
        data = {
            "content": content
        }
        return self.post(CODE_BLOCK_COMMENT, data)
    
    def line_comment(
        self,
        content
    ):
        data = {
            "content": content
        }
        return self.post(CODE_LINE_COMMENT, data)
    

    def print_code(
        self,
        variable,
        type= None
    ):
        data = {
            "variable": variable
        }
        if type:
            data["type"] = type
        return self.post(CODE_PRINT, data)
    
    def user_input(
        self,
        variable,
        message
    ):
        data = {
            "variable": variable
        }
        if message:
            data["message"] = message
        return self.post(CODE_USER_INPUT, data)
    
    def type_casting(
        self,
        variable,
        type
    ):
        data = {
            "variable": variable,
            "type": type
        }
        return self.post(CODE_TYPE_CASTING, data)
    

    def assertion(
        self,
        variable,
        type,
        value
    ):
        data = {
            "variable": variable,
            "type": type,
            "value": value
        }
        return self.post(CODE_ASSERTION, data)
    
    def import_library(
        self,
        library
    ):
        data = {
            "library": library
        }
        return self.post(CODE_IMPORT_LIBRARY, data)
    
    
    def import_module(
        self,
        library,
        modules
    ):
        data = {
            "library": library,
            "modules": modules
        }
        return self.post(CODE_IMPORT_MODULE, data)
  
        
    def declare_class(self, name, properties, methods):
        data = {
            "name": name,
        }
        if properties:
            data["properties"] = properties
        if methods:
            data["methods"] = methods
        return self.post(CODE_DECLARE_CLASS, data)
    
    def conditional(self, keyword, conditions):
        data = {
            "keyword": keyword,
            "conditions": conditions
        }
        return self.post(CODE_CONDITIONAL, data)
        

    def add_whitespace(
        self,
        type,
        count = 1):
        
        return self.get(
            CODE_ADD_WHITESPACE,
            {
                "type": type,
                "count": count
            }
        )
    
    def declare_function(self, name, parameters):
        data = {
            "name": name,
            "parameters": parameters
        }
        return self.post(CODE_DECLARE_FUNCTION, data)
        pass

    def function_call(self, names, args):
        data = {
            "names": names,
            "args": args
        }
        return self.post(CODE_FUNCTION_CALL, data)

    def operation(self,right,operator, left):
        data = {
            "right": right,
            "operator": operator,
            "left": left
        }
        return self.post(CODE_OPERATION, data)

    def for_loop(self, type, iterators, iterable):
        data = {
            "type": type,
            "iterators": iterators,
            "iterable": iterable
        }
        return self.post(CODE_FOR_LOOP, data)

    def while_loop(
        self,
        condition
    ):
        data = {
            "condition": condition
        }
        return self.post(CODE_WHILE_LOOP, data)
    

    def assign_variable(
        self,
        name,
        type,
        value
    ):
        data = {
            "name": name,
            "type": type,
            "value": value
        }
        return self.post(CODE_ASSIGN_VARIABLE, data)