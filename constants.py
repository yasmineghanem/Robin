
'''
code endpoints
'''
CODE_DECLARE_VARIABLE = '/code/declare-variable'
CODE_ASSIGN_VARIABLE = '/code/assign-variable'
CODE_DECLARE_FUNCTION = '/code/declare-function'
CODE_FUNCTION_CALL = '/code/function-call'
CODE_DECLARE_CONSTANT = '/code/declare-constant'
CODE_FOR_LOOP = '/code/for-loop'
CODE_WHILE_LOOP = '/code/while-loop'
CODE_ADD_WHITESPACE = '/code/add-whitespace'
CODE_IMPORT_LIBRARY = '/code/import-library'
CODE_IMPORT_MODULE = '/code/import-module'
CODE_OPERATION = '/code/operation'
CODE_ASSERTION = '/code/assertion'
CODE_TYPE_CASTING = '/code/type-casting'
CODE_USER_INPUT = '/code/user-input'
CODE_PRINT = '/code/print'
CODE_LINE_COMMENT = '/code/line-comment'
CODE_BLOCK_COMMENT = '/code/block-comment'
CODE_READ_FILE = '/code/read-file'
CODE_WRITE_FILE = '/code/write-file'
CODE_CONDITIONAL = '/code/conditional'
CODE_DECLARE_CLASS = '/code/declare-class'
CODE_GET_AST = '/code/ast'
CODE_TRY_EXCEPT = '/code/try-except'


'''
file system endpoints
'''

# Constants for filesystem-related endpoints
FILE_SYSTEM_CREATE_FILE = '/file-system/create-file'
FILE_SYSTEM_CREATE_DIRECTORY = '/file-system/create-directory'
FILE_SYSTEM_COPY_FILE = '/file-system/copy-file'
FILE_SYSTEM_COPY_DIRECTORY = '/file-system/copy-directory'
FILE_SYSTEM_DELETE = '/file-system/delete'
FILE_SYSTEM_RENAME = '/file-system/rename'
FILE_SYSTEM_SAVE = '/file-system/save'
FILE_SYSTEM_GET_FILES = '/file-system/get-files'

''' 
IDE endpoints
'''


IDE_GO_TO_LINE = '/ide/go-to-line'
IDE_GO_TO_FILE = '/ide/go-to-file'
IDE_FOCUS_TERMINAL = '/ide/focus-terminal'
IDE_NEW_TERMINAL = '/ide/new-terminal'
IDE_KILL_TERMINAL = '/ide/kill-terminal'
IDE_COPY = '/ide/copy'
IDE_PASTE = '/ide/paste'
IDE_CUT = '/ide/cut'
IDE_UNDO = '/ide/undo'
IDE_REDO = '/ide/redo'
IDE_FIND = '/ide/find'
IDE_SELECT = '/ide/select'
IDE_SELECT_KERNEL = '/ide/select-kernel'
IDE_RUN_NOTEBOOK_CELL = '/ide/run-notebook-cell'
IDE_RUN_NOTEBOOK = '/ide/run-notebook'
IDE_RUN_PYTHON_FILE = '/ide/run-python-file'


'''
Git endpoints
'''
GIT_PUSH_ENDPOINT = '/git/push'
GIT_PULL_ENDPOINT = '/git/pull'
GIT_DISCARD_ENDPOINT = '/git/discard'
GIT_STAGE_ENDPOINT = '/git/stage'
GIT_STASH_ENDPOINT = '/git/stash'