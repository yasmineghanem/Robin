import requests
import json
from constants import *


class APIController:
    def __init__(self):
        # read initial configurations from the json file
        with open('config.json') as json_file:
            data = json.load(json_file)
            self.url = data['base_url']

    def __get(self, endpoint, params=None):
        response = requests.get(self.url + endpoint, params=params, timeout=15)
        return response.json()

    def __post(self, endpoint, data):
        response = requests.post(self.url + endpoint, json=data,
                                 headers={'Content-Type': 'application/json',
                                          'Accept': 'application/json'
                                          },
                                 timeout=15
                                 )

        # print(response.text)

        return response.json
    # File System Endpoints

    def create_file(self, fileName, extension, content):
        data = {
            "fileName": fileName,
            "extension": extension,
            "content": content
        }
        return self.__post(FILE_SYSTEM_CREATE_FILE, data)

    def create_directory(self, name):
        data = {
            "name": name
        }
        return self.__post(FILE_SYSTEM_CREATE_DIRECTORY, data)

    def copy_file(self, source, destination):
        data = {
            "source": source,
            "destination": destination
        }
        return self.__post(FILE_SYSTEM_COPY_FILE, data)

    def copy_directory(self, source, destination):
        data = {
            "source": source,
            "destination": destination
        }
        return self.__post(FILE_SYSTEM_COPY_DIRECTORY, data)

    def delete(self, source):
        data = {
            "source": source
        }
        return self.__post(FILE_SYSTEM_DELETE, data)

    def rename(self, source, destination):
        data = {
            "source": source,
            "destination": destination
        }
        return self.__post(FILE_SYSTEM_RENAME, data)

    def save(self):
        data = {}
        return self.__post(FILE_SYSTEM_SAVE, data)

    def get_files(self):
        return self.__post(FILE_SYSTEM_GET_FILES, data={})

    # IDE Endpoints
    def go_to_line(self, line, character):
        data = {
            "line": line,
            "character": character
        }
        return self.__post(IDE_GO_TO_LINE, data)

    def go_to_file(self, path):
        data = {
            "path": path
        }
        return self.__post(IDE_GO_TO_FILE, data)

    def focus_terminal(self):
        return self.__get(IDE_FOCUS_TERMINAL)

    def undo(self):
        return self.__get(IDE_UNDO)

    def new_terminal(self):
        return self.__get(IDE_NEW_TERMINAL)

    def kill_terminal(self):
        return self.__get(IDE_KILL_TERMINAL)

    def cut(self):
        return self.__get(IDE_CUT)

    def copy(self):
        return self.__get(IDE_COPY)

    def paste(self):
        return self.__get(IDE_PASTE)

    def redo(self):
        return self.__get(IDE_REDO)

    def select(self, line):
        data = {
            "line": line
        }
        return self.__get(IDE_SELECT, data)

    def find(self, line):
        data = {
            "line": line
        }
        return self.__get(IDE_FIND, data)

    def run_notebook(self):
        return self.__get(IDE_RUN_NOTEBOOK)

    def run_python_file(self, path):
        data = {
            "path": path
        }
        return self.__post(IDE_RUN_PYTHON_FILE, data)

    def run_notebook_cell(self, data):
        return self.__get(IDE_RUN_NOTEBOOK_CELL, data)

    def select_kernel(self, data):
        return self.__post(IDE_SELECT_KERNEL, data)

    # Git Endpoints
    def discard(self):
        return self.__get(GIT_DISCARD_ENDPOINT)

    def pull(self):
        return self.__get(GIT_PULL_ENDPOINT)

    def stage(self):
        return self.__get(GIT_STAGE_ENDPOINT)

    def stash(self):
        return self.__get(GIT_STASH_ENDPOINT)

    # Code Execution Endpoints
    def try_except(self, tryBody, exception, exceptionInstance, exceptBody):
        data = {
            "tryBody": tryBody,
            "exception": exception,
            "exceptionInstance": exceptionInstance,
            "exceptBody": exceptBody
        }
        return self.__post(CODE_TRY_EXCEPT, data)

    def declare_constant(self, name, value):
        data = {
            "name": name,
            "value": value
        }
        return self.__post(CODE_DECLARE_CONSTANT, data)

    def declare_variable(
        self,
        variable_name,
        variable_value,
        variable_type=None,
    ):
        var_data = {
            "name": variable_name,
            "value": variable_value,
        }
        if variable_type:
            var_data["type"] = variable_type
        return self.__post(CODE_DECLARE_VARIABLE, var_data)

    def write_file(
        self,
        path,
        content
    ):
        data = {
            "path": path,
            "content": content
        }
        return self.__post(CODE_WRITE_FILE, data)

    def read_file(
        self,
        path,
        variable=None
    ):
        data = {
            "path": path
        }
        if variable:
            data["variable"] = variable
        return self.__post(CODE_READ_FILE, data)

    def block_comment(
        self,
        content
    ):
        data = {
            "content": content
        }
        return self.__post(CODE_BLOCK_COMMENT, data)

    def line_comment(
        self,
        content
    ):
        data = {
            "content": content
        }
        return self.__post(CODE_LINE_COMMENT, data)

    def print_code(
        self,
        variable,
        type=None
    ):
        data = {
            "variable": variable
        }
        if type:
            data["type"] = type
        return self.__post(CODE_PRINT, data)

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
        return self.__post(CODE_USER_INPUT, data)

    def type_casting(
        self,
        variable,
        type
    ):
        data = {
            "variable": variable,
            "type": type
        }
        return self.__post(CODE_TYPE_CASTING, data)

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
        return self.__post(CODE_ASSERTION, data)

    def import_library(
        self,
        library
    ):
        data = {
            "library": library
        }
        return self.__post(CODE_IMPORT_LIBRARY, data)

    def import_module(
        self,
        library,
        modules
    ):
        data = {
            "library": library,
            "modules": modules
        }
        return self.__post(CODE_IMPORT_MODULE, data)

    def declare_class(self, name, properties, methods):
        data = {
            "name": name,
        }
        if properties:
            data["properties"] = properties
        if methods:
            data["methods"] = methods
        return self.__post(CODE_DECLARE_CLASS, data)

    def conditional(self, data: [{"keyword": str, "condition": [{"left": str, "operator": str, "right": str}]}]):
        return self.__post(CODE_CONDITIONAL, data)

    def add_whitespace(
            self,
            type,
            count=1):

        return self.__get(
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
        return self.__post(CODE_DECLARE_FUNCTION, data)

    def function_call(self, name, args):
        data = {
            "name": name,
            "args": args
        }
        return self.__post(CODE_FUNCTION_CALL, data)

    def operation(self, right, operator, left):
        data = {
            "right": right,
            "operator": operator,
            "left": left
        }
        return self.__post(CODE_OPERATION, data)

    def for_loop(self, loop_type, rest):
        data = {
            "type": loop_type,
            **rest
        }
        return self.__post(CODE_FOR_LOOP, data)

    def while_loop(
        self,
        condition
    ):
        data = {
            "condition": condition
        }
        return self.__post(CODE_WHILE_LOOP, data)

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
        return self.__post(CODE_ASSIGN_VARIABLE, data)

    def exit_scope(self):
        return self.__get(CODE_EXIT_SCOPE)
