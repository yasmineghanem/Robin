from enum import Enum

VARIABLE = 'variable declaration'
CONSTANT = 'constant declaration'
FUNCTION = 'function declaration'
CLASS = 'class declaration'
FOR = 'for loop'
WHILE = 'while loop'
ASSIGNMENT = 'assignment operation'
BITWISE = 'bitwise operation'
MATHEMATICAL = 'mathematical operation'
CASTING = 'casting'
INPUT = 'input'
OUTPUT = 'output'
ASSERTION = 'assertion'
LIBRARY = 'libraries'
COMMENT = 'comment'
CONDITIONAL = 'conditional operation'
FILE = 'file system'
GIT = 'git operation'
INTERACTIVE = 'activate interactive'
MOUSE = 'activate mouse'
MEMBERSHIP = 'membership operation'
IDE = 'ide operation'
ARRAY = 'array operation'


class Intents(Enum):
    VARIABLE = 1
    CONSTANT = 2
    FUNCTION = 3
    CLASS = 4
    FOR = 5
    WHILE = 6
    ASSIGNMENT = 7
    CONDITIONAL = 8
    ARRAY = 9
    BITWISE = 10
    MATHEMATICAL = 11
    MEMBERSHIP = 12
    CASTING = 13
    INPUT = 14
    OUTPUT = 15
    ASSERTION = 16
    LIBRARIES = 17
    FILE = 18
    IDE = 19
    COMMENTS = 20
    MOUSE = 21
    INTERACTIVE = 22
    INTERACTIVE_COMMANDS = 23
    GIT = 24


TYPES = {
    'int': ['int', 'integer', 'number'],
    'float': ['float', 'decimal'],
    'double': ['double'],
    'str': ['string', 'text', 'str'],
    'bool': ['bool', 'boolean'],
    'list': ['list', 'array'],
    'dictionary': ['dictionary', 'dict', 'map'],
    'tuple': ['tuple'],
    'set': ['set'],
}

CONDITIONS = {
    '!': ['not', 'different'],
    '>': ['greater', 'more', 'bigger', 'larger'],
    '<': ['less', 'fewer', 'lesser', 'smaller'],
    # add is with some conditions for membership operation
    '==': ['equal', 'same', 'is'],
}

OPERATORS = {
    'bitwise': {
        'And': ['and'],
        'Or': ['or'],
        'Xor': ['xor'],
        'Not': ['not'],
        'LeftShift': ['shift left', 'left shift'],
        'RightShift': ['shift right', 'right shift'],

    }, 
    'mathematical': {
        'Addition': ['add', 'plus', 'sum', 'increase', 'increment', 'total', 'combine', 'join', 'append', 'attach', 'connect', 'merge'],
        'Subtraction': ['subtract', 'minus', 'difference', 'decrease', 'decrement', 'remove', 'take away', 'exclude', 'eliminate', 'deduct', 'reduce', 'sub', 'less', 'shorten'],
        'Multiplication': ['multiply', 'times', 'product', 'repeat', 'multiplied by', 'multiplication'],
        'Division': ['divide', 'division', 'divided by', 'split', 'separate', 'partition'],
        'Modulus': ['modulus', 'mod', 'remainder', 'modulo', 'remainder of', 'remainder when divided by'],
        'Exponentiation': ['exponent', 'power', 'raise', 'to the power of', 'power of', 'exponentiation', 'exponential'],
    },
    'conditional':{
        'And': ['and'],
        'Or': ['or', 'either', 'alternatively', 'otherwise', 'instead', 'or else'],
        'not': ['not', 'neither', 'nor', 'except', 'but', 'however', 'nevertheless', 'nonetheless', 'although', 'though', 'yet', 'still', 'even so', 'conversely', 'on the other hand', 'on the contrary', 'in contrast', 'despite', 'in spite of', 'regardless', 'anyway', 'anyhow', 'anywise', 'at any rate', 'at all events', 'in any case', 'nevertheless', 'nonetheless', 'however', 'yet', 'still', 'even so', 'conversely', 'on the other hand', 'on the contrary', 'in contrast', 'despite', 'in spite of', 'regardless', 'anyway', 'anyhow', 'anywise', 'at any rate', 'at all events', 'in any case'],
    }

}

ACTIONS = {
    'file': {
        'create': ['create', 'make', 'generate'],
        'delete': ['delete', 'remove', 'erase'],
        'rename': ['rename', 'change'],
        'copy': ['copy', 'duplicate'],
        'save': ['save', 'store']
    },
    'git': {
        'push': ['push', 'upload', 'commit'],
        'pull': ['pull', 'download', 'fetch', 'get'],
        'discard': ['revert', 'undo', 'remove', 'discard', 'reset'],
        'stash': ['stash', 'store', 'save'],
        'stage': ['stage', 'add', 'track'],
    },
    'ide': {
        'goto': ['go to', 'navigate', 'move to', 'open'],
        'focus': ['focus'],
        'undo': ['undo', 'revert', 'reset', 'go back', 'remove'],
        'kill': ['kill', 'close', 'end', 'exit'],
        'copy': ['copy', 'duplicate'],
        'cut': ['cut'],
        'paste': ['paste'],
        'redo': ['redo', 'do it again'],
        'select': ['select', 'highlight'],
        'find': ['fine', 'search', 'look for', 'search for'],
        'run': ['run', 'execute'],
        'new': ['new', 'open', 'create']
    }
}
