from enum import Enum


class Intents(Enum):
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


TYPES = {
    'Integer': ['int', 'integer', 'number'],
    'Float': ['float', 'decimal'],
    'Double': ['double'],
    'String': ['string', 'text', 'str'],
    'Boolean': ['bool', 'boolean'],
    'List': ['list', 'array'],
    'Dictionary': ['dictionary', 'dict', 'map'],
    'Tuple': ['tuple'],
    'Set': ['set']
}

CONDITIONS = {
    '!': ['not', 'different'],
    '>': ['greater', 'more', 'bigger', 'larger'],
    '<': ['less', 'fewer', 'lesser', 'smaller'],
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
        'Addition': ['add', 'plus', 'sum', 'increase', 'increment', 'total', 'combine', 'join', 'append', 'attach', 'connect', 'merge', 'addition'],
        'Subtraction': ['subtract', 'minus', 'difference', 'decrease', 'decrement', 'remove', 'take away', 'exclude', 'eliminate', 'deduct', 'reduce', 'sub', 'less', 'shorten', 'subtraction'],
        'Multiplication': ['multiply', 'times', 'product', 'repeat', 'multiplied by', 'multiplication'],
        'Division': ['divide', 'division', 'divided by', 'split', 'separate', 'partition'],
        'Modulus': ['modulus', 'mod', 'remainder', 'modulo', 'remainder of', 'remainder when divided by'],
        'Exponentiation': ['exponent', 'power', 'raise', 'to the power of', 'power of', 'exponentiation', 'exponential'],
    },
    'conditional': {
        'And': ['and'],
        'Or': ['or', 'either', 'alternatively', 'otherwise', 'instead', 'or else'],
        'Not': ['not'],
    },
    'membership': {
        'In': ['in', 'inside', 'contained in', 'within'],
        'NotIn': ['not in', 'not inside', 'not contained in', 'not within'],
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
    },
    'activation': {
        'activate': ['activate', 'enable', 'start', 'turn on', 'go to', 'switch on', 'open', 'begin', 'initiate', 'launch', 'run', ],
        'deactivate': ['deactivate', 'disable', 'stop', 'turn off', 'close', 'exit', 'kill', 'switch off', 'end', 'terminate', 'shut down', 'quit', 'remove'],
    }
}

FALLBACK_INTENTS = [
    'mouse click',
    'activate interactive',
    'activate mouse',
    'exit block'
]

NO_ENTITIES = [
    "exit block"
]
