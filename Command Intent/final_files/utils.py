from enum import Enum


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
    '==': ['equal', 'is', 'same'],
}

BITWISE_OPERATORS = {
    'and': '&',
    'or': '|',
    'not': '~',
    'shift left': '<<',
    'shift right': '>>',
    'xor': '^'
}

def clean(line):
    '''
    This function cleans the line by removing all the special characters and punctuations

    Args:
        - line (str) : a sentence from the dataset that needs to be cleaned

    Returns:
        - cleaned_line (str) : cleaned sentence
    '''
    cleaned_line = ''
    for char in line:
        if char.isalpha():
            cleaned_line += char
        else:
            cleaned_line += ' '
    cleaned_line = ' '.join(cleaned_line.split())
    return cleaned_line
