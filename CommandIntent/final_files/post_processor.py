import utils
import re
from constants import *
from DesktopApplication.api import APIController
# TODO: handle if the intent is wrong


class PostProcessor:
    '''
        PostProcessor class is used to post process the output of the two models.
        It takes the detected intent along with the entities and maps them to the final output.
        The final output is then sent to the command execution module using the API.
        The main purpose is to send th ecorrect values with the correct types to the command execution module.

        Args:
            - sentence (str) : the user input
            - intent (str) : the detected intent
            - tags (list) : the detected entities
            - intent_to_tags (dict) : a mapping between the intent and the entities
    '''

    def __init__(self, intent_to_tags):
        self.intent_to_tags = intent_to_tags
        self.api = APIController()

    def post_process(self, sentence, intent, tags):
        '''
            post_process function is used to post process the output of the two models.
            It takes the detected intent along with the entities and maps them to the final output.
            The final output is then sent to the command execution module using the API.
            The main purpose is to send th correct values with the correct types to the command execution module.
        '''
        sentence = sentence.split()

        intent_tags = self.intent_to_tags[intent]

        parameters = self.__get_parameters(intent, sentence, tags, intent_tags)

        final_parameters = None

        match intent:
            case 'variable declaration':
                final_parameters = self.post_process_declaration(parameters)
                response = self.api.declare_variable(final_parameters['name'], final_parameters['value'], final_parameters['type'])

            case 'constant declaration':
                final_parameters = self.post_process_declaration(parameters)
                response = self.api.declare_constant(final_parameters['name'], final_parameters['value'])

            case 'function declaration':
                final_parameters = self.post_process_function_declaration(
                    parameters)
                response = self.api.declare_function(final_parameters['name'], final_parameters['parameters'])

            case 'class declaration':
                final_parameters = self.post_process_class_declaration(
                    parameters)
                response = self.api.declare_class(final_parameters['name'])

            case 'for loop':
                final_parameters = self.post_process_for_loop(parameters)
                response = self.api.for_loop(final_parameters['type'], final_parameters['iterators'], final_parameters['start'], final_parameters['end'], final_parameters['step'], final_parameters['iterable'], final_parameters['body'])

            case 'while loop':
                final_parameters = self.post_process_while_loop(parameters)
                response = self.api.conditional(final_parameters)

            case 'assignment operation':
                final_parameters = self.post_process_assignment_operation(
                    parameters)
                response = self.api.assignment(final_parameters)

            case 'bitwise operation':
                final_parameters = self.post_process_operation(parameters)
                response = self.api.operation(final_parameters)

            case 'casting':
                final_parameters = self.post_process_casting(parameters)
                response = self.api.casting(final_parameters)

            case 'input':
                final_parameters = self.post_process_input(parameters)
                response = self.api.user_input(final_parameters)
            case 'output':
                final_parameters = self.post_process_output(parameters)
                response = self.api.print(final_parameters)

            case 'assertion':
                final_parameters = self.post_process_assertion(parameters)
                response = self.api.assertion(final_parameters)

            case 'libraries':
                final_parameters = self.post_process_libraries(parameters)
                response = self.api.import_library(final_parameters)

            case 'comment':
                final_parameters = self.post_process_comment(parameters)
                response = self.api.comment(final_parameters)

            case 'conditional operation':
                final_parameters = self.post_process_conditional_operation(
                    parameters)
                response = self.api.conditional(final_parameters)

            case 'file system':
                final_parameters = self.post_process_file_system(parameters)
                response = self.api.file_system(final_parameters)

            case 'git operation':
                self.post_process_git_operation(parameters)
                response = self.api.git_operation(final_parameters)

            case 'interactive commands':
                self.post_process_interactive_commands(parameters)
                response = self.api.interactive_commands(final_parameters)

            case 'membership operation':
                final_parameters = self.post_process_conditional_operation(
                    parameters)
                response = self.api.conditional(final_parameters)

            case 'mathematical operation':
                final_parameters = self.post_process_operation(parameters)
                response = self.api.operation(final_parameters)

            case 'ide operation':
                final_parameters = self.post_process_ide_operation(parameters)
                response = self.api.ide_operation(final_parameters)

            case 'array operation':
                final_parameters = self.post_process_array_operation(parameters)
                response = self.api.operation(final_parameters)

        return response

    def __get_parameters(self, intent, sentence, tags, intent_tags):
        '''
            This function is used to get the parameters of the detected entities.
            The function takes the indices of the B-tags and the tags of the entities and the sentence.
            It returns a dictionary of the parameters of the detected entities.
        '''
        intent_tags = self.intent_to_tags[intent]

        parameters = {tag: None for tag in intent_tags if tag != 'O'}

        if intent == 'function declaration':
            parameters['PARAM'] = []
        elif intent == 'bitwise operation' or intent == 'mathematical operation':
            parameters['OPERAND'] = []
        elif intent == 'ide operation':
            parameters['LINE'] = []

        for tag in intent_tags:
            if tag == 'O':
                continue

            B_indices = [index for index, entity in enumerate(
                tags) if entity.startswith('B-' + tag)]

            if tag != 'OPERAND' and tag != 'LINE':
                if (len(B_indices) > 1):
                    B_indices = B_indices[:1]

            print(f"Tag: {tag}, B_indices: {B_indices}")

            # loop from the index until the next tag that is not the current tag
            # get all the indices of the entity

            for index in B_indices:
                full_entity = []
                full_entity.append(sentence[index])

                for i in range(index+1, len(tags)):

                    if tags[i].startswith('I-' + tag):
                        full_entity.append(sentence[i])
                    else:
                        break

                print(f"Full entity: {full_entity}")

                if tag == 'PARAM' or tag == 'OPERAND' or tag == 'LINE':
                    parameters[tag].append(' '.join(full_entity))
                else:
                    parameters[tag] = ' '.join(full_entity)

                print(f"Parameters: {parameters}")

        return parameters

    def __map_values(self, value, type):
        final_value = None

        integer_regex = r'^-?\d+$'
        float_regex = r'^-?\d*\.\d+$'
        boolean_regex = r'^([Tt]rue|[Ff]alse)$'

        if type is not None:
            if type in utils.TYPES['int']:
                final_value = int(value)
            elif type in utils.TYPES['float']:
                final_value = float(value)
            elif type in utils.TYPES['double']:
                final_value = float(value)
            elif type in utils.TYPES['str']:
                final_value = value
            elif type in utils.TYPES['bool']:
                final_value = bool(value)
            elif type in utils.TYPES['list']:
                final_value = []
            elif type in utils.TYPES['dictionary']:
                final_value = {}
            elif type in utils.TYPES['tuple']:
                final_value = tuple()
            elif type in utils.TYPES['set']:
                final_value = set()
        else:
            # if a type is not specified then check the value and map it to correct type
            # based on intuition and regex
            # for example if the value is 'True/False' then the type is probably boolean
            # if the value is a number then the type is probably int or float
            # if not the previous then the type is probably string
            # if the value is empty and the type is not specified -> leave it as None
            if re.match(integer_regex, value) is not None:
                # integer value
                final_value = int(value)
            elif re.match(float_regex, value) is not None:
                # float value
                final_value = float(value)
            elif re.match(boolean_regex, value) is not None:
                # boolean value
                final_value = bool(value.capitalize())
            else:
                # string value
                final_value = value

        return final_value

    def __map_type(self, type):

        if type in utils.TYPES['int']:
            final_type = 'int'
        elif type in utils.TYPES['float']:
            final_type = 'float'
        elif type in utils.TYPES['double']:
            final_type = 'double'
        elif type in utils.TYPES['str']:
            final_type = 'string'
        elif type in utils.TYPES['bool']:
            final_type = 'bool'
        elif type in utils.TYPES['list']:
            final_type = 'list'
        elif type in utils.TYPES['dictionary']:
            final_type = 'dict'
        elif type in utils.TYPES['tuple']:
            final_type = 'tuple'
        elif type in utils.TYPES['set']:
            final_type = 'set'

        return final_type

    def __map_condition(self, condition):
        '''
            This function takes the condition as a string and maps it to the correct operator.
            The operators are:
            - == : equal
            - != : not equal
            - > : greater than
            - < : less than
            - >= : greater than or equal
            - <= : less than or equal

        '''
        final_operator = None

        if condition is not None:
            # check for all the conditions
            # first we check the not equal condition

            for operator, values in utils.CONDITIONS.items():
                print(f"Final operator: {final_operator}")

                print(f"Operator: {operator}, Values: {values}")

                # if the operator is not equal then we check for the words in equal and not equal
                # this id for the processing of equal and not equal conditions
                if operator == '!':
                    not_present = False
                    equal_present = False

                    # check for not keywords
                    for value in values:
                        if value in condition:
                            not_present = True
                            break

                    # check for equal keywords
                    for value in utils.CONDITIONS['==']:
                        if value in condition:
                            equal_present = True
                            break

                    if not_present and equal_present:
                        final_operator = '!='
                    elif not_present:
                        final_operator = 'not'

                    if final_operator is not None:
                        break

                elif operator == '==':

                    for value in values:
                        if value in condition:
                            final_operator = operator
                            break

                    if final_operator is not None:
                        break

                # processing for the > or >= conditions
                elif operator == '>' or operator == '<':
                    # if the operator is ! then the other values are for the == and vice vers
                    for value in values:
                        if value in condition:
                            final_operator = operator
                            break

                    # check if theres equal as well
                    if final_operator is not None:
                        for value in utils.CONDITIONS['==']:
                            if value in condition:
                                final_operator += '='
                                break

                    if final_operator is not None:
                        break

        print(f"Final operator: {final_operator}")

        return final_operator

    def __map_operator(self, operator, intent):

        final_operator = None

        if intent == 'bitwise operation':
            for key, value in utils.BITWISE_OPERATORS.items():
                if key in operator:
                    final_operator = value
                    break

        elif intent == 'mathematical operation':
            pass

        return final_operator

    def __map_actions(self, action, intent):
        '''
            Maps actions for:
            - file system operations
            - git operations

            The actions are:
            1. File system operations: create | copy | delete | rename | save 
            2. Git operations: pull | push | discard | stage | stash
        '''
        final_action = None

        intent_actions = utils.ACTIONS[intent]

        for key, value in intent_actions.items():
            if action in value:
                final_action = key
                break

        return final_action

    def __get_file_extension(self):
        return None

    def __map_array_function(self, function):
        pass

    def __map_ide_type(self, operation_type):
        final_type = None
        return final_type

    # DONE: variable and constant declaration
    def post_process_declaration(self, parameters):
        '''
            This function is used to post process the variable declaration intent to send to the execution
            we need to map the tags to the correct format.
            and check the type to get the correct value.
            the tags for this intent are:
            - VAR : the variable name
            - VALUE : the value of the variable (optional)
            - TYPE : the type of the variable (optional)
        '''
        # declare a variable the command execution needs
        '''
            {
                "name":
                "value":
                "type": (lesa)
            }
        '''
        final_parameters = {}

        final_parameters['name'] = parameters['VAR']

        final_parameters['value'] = self.__map_values(
            parameters['VALUE'], parameters['TYPE'])

        final_parameters['type'] = parameters['TYPE']

        return final_parameters

    # DONE: mapping final parameters
    def post_process_function_declaration(self, parameters):
        '''
            This function is used to post process the function declaration intent.
            to get the final parameters to send to the command execution module.
            the tags for this intent are:
            - FUNC : the function name
            - PARAM : the parameters that the function takes (optional)
            - TYPE : the return type of the function (optional)    

            The format of the final parameters is:
            {
                "name": function_name,
                "parameters": [
                    {
                        "name": "x_variable",
                    },
                    {
                        "name": "y"
                    }
                ],
                "return_type": 'void'
            }
        '''
        final_parameters = {}

        final_parameters['name'] = parameters['FUNC']
        final_parameters['parameters'] = []
        for param in parameters['PARAM']:
            final_parameters['parameters'].append({'name': param})

        if parameters['TYPE'] is None:
            final_parameters['return_type'] = 'void'
        else:
            final_parameters['return_type'] = self.__map_type(
                parameters['TYPE'])

        return final_parameters

    # DONE: class declaration intent
    def post_process_class_declaration(self, parameters):
        '''
            the tags are:
            CLASS : the class name

            final format:
            {
                "name": class_name
            } 
        '''

        final_parameters = {"name": parameters['CLASS']}

        return final_parameters

    # DONE: for loop intent
    def post_process_for_loop(self, parameters):
        '''
            There are two types of for loops:
            1. for loop with list
            2. for loop with range
            based on the parameter we can determine the type of the for loop

            the tags for this intent are:
            - VAR : the variable name
            - START : the start value of the loop
            - END : the end value of the loop
            - COLLECTION : the collection to loop over (optional)
            - STEP: the step value of the loop (optional)

            The format of the final parameters is:
            1. {
                    "type": "iterable",
                    "iterators": [
                        "i"
                    ],
                    "iterable": "s",
                    "body":[
                        "x = 5",
                        "print(x+5)"
                    ]
                } -> no body => None

            2. {
                    "type": "range",
                    "iterators": [
                        "i"
                    ],
                    "start" : "0",
                    "end" : "10",
                    "step" : "1"
                }

            (Could change later but if collection present then it is iterable else range)
        '''
        final_parameters = {}

        if parameters['COLLECTION'] is not None:
            final_parameters['type'] = 'iterable'
            final_parameters['iterators'] = [parameters['VAR'] if parameters['VAR'] is not None else 'i']
            final_parameters['iterable'] = parameters['COLLECTION']
            final_parameters['body'] = None

        else:
            final_parameters['type'] = 'range'
            final_parameters['iterators'] = [parameters['VAR'] if parameters['VAR'] is not None else 'i']
            
            # TODO : check if the start, end, and step are numbers

            # the start and the step are optional
            final_parameters['start'] = parameters['START']
            final_parameters['step'] = parameters['STEP'] 

            # must be provided by the user
            final_parameters['end'] = parameters['END']

        return final_parameters

    # DONE: while loop intent
    def post_process_while_loop(self, parameters):
        '''
            the tags for this intent are:
            - CONDITION : the condition of the loop
            - LHS : the left hand side of the condition
            - RHS : the right hand side of the condition

            final format:
            {
                "condition": [
                    {
                        "left": "x",
                        "operator": ">",
                        "right": "5"
                    }, 
                    {
                        "logicalOperator": "and",
                        "left": "x",
                        "operator": ">",
                        "right": "5"
                    }
                ]
            }
        '''
        final_parameters = {}

        final_parameters['condition'] = []
        final_parameters['condition'].append({
            'left': parameters['LHS'],
            'operator': self.__map_condition(parameters['CONDITION']),
            'right': parameters['RHS']
        })
        pass

    # DONE: casting intent
    def post_process_casting(self, parameters):
        final_parameters = {}
        '''
            the tags for casting intent are:
            - VAR : the variable name
            - TYPE : the type to cast to

            final format:
            {
                "variable": "c",
                "type": "int"
            }
        '''

        final_parameters['variable'] = parameters['VAR']
        final_parameters['type'] = self.__map_type(parameters['TYPE'])

        return final_parameters

    # DONE: assignment operation intent
    def post_process_assignment_operation(self, parameters):
        '''
            The tags for the assignment intent:
            - LHS -> always has to be a variable
            - RHS -> could be a variable or value (need to map to the correct type)

            final format:
            {
                "name": LHS,
                "type": "=s", (is always =)
                "value": RHS
            }
        '''

        final_parameters = {}

        final_parameters['name'] = parameters['LHS']
        final_parameters['type'] = '=s'
        final_parameters['value'] = self.__map_values(parameters['RHS'], None)

        return final_parameters

    # DONE: assertion intent
    def post_process_assertion(self, parameters):
        '''
            The tags for assertion:
            - VAR: the name of the variable
            - VAL: the value
            - CONDITION: the condition

            final format:
            {
                "variable": the name of the variable,
                "type": condition,
                "value": the value
            }
        '''

        final_parameters = {}

        final_parameters['variable'] = parameters['VAR']
        final_parameters['type'] = self.__map_condition(
            parameters['CONDITION'])
        final_parameters['value'] = self.__map_values(
            parameters['VAL'], None) if parameters['VAL'] is not None else None

        return final_parameters

    # DONE: libraries intent
    def post_process_libraries(self, parameters):
        '''
            the tags for libraries intent:
            - LIB_NAME : the name of the library

            final format:
            {
                "library": "sklearn"
            }
        '''
        final_parameters = {}

        final_parameters['library'] = parameters['LIB_NAME']

        return final_parameters

    # DONE: bitwise operation intent
    def post_process_operation(self, parameters, intent):
        '''
            intent tags are:
            - VAR (not always)
            - OPERAND : the operands of the bitwise operation
            - OPERATOR : the bitwise operation (and, or, not, shift left, shift right, xor) -> map the operator

            final format:
            {
                "right": "a",
                "operator": "and",
                "left": "b"
            }
        '''
        final_parameters = {}

        if len(parameters['OPERAND']) < 2:
            # either not or wrong
            if parameters['OPERATOR'] in utils.BITWISE_OPERATORS['not']:
                final_parameters['left'] = parameters['OPERAND'][0]
                final_parameters['right'] = None
                final_parameters['operator'] = self.__map_operator(
                    parameters['OPERATOR'], intent)
        else:
            final_parameters['left'] = parameters['OPERAND'][0]
            final_parameters['right'] = parameters['OPERAND'][1]
            final_parameters['operator'] = self.__map_operator(
                parameters['OPERATOR'], intent)

        return final_parameters

    # DONE: comment intent
    def post_process_comment(self, parameters):
        '''
            This will be the line comment 
                tags:
                COMMENT

            final format:
            {
                "content": "This is a one line comment"
            }
        '''

        final_parameters = {}

        final_parameters['content'] = parameters['COMMENT']

        return final_parameters

    # DONE: input intent
    def post_process_input(self, parameters):
        '''
            The tags are:
            - VAR
            - MESSAGE

            final format:
            {
                "variable": variable name,
                "message": message
            }
        '''

        final_parameters = {}

        final_parameters['variable'] = parameters['VAR']
        final_parameters['message'] = parameters['MESSAGE']

        return final_parameters

    # DONE: output intent
    def post_process_output(self, parameters):
        '''
            The tags for output intent are:
            - VAR
            - VAL
            - MESSAGE

            should be at most 1

            final format:
            {
                "variable": content to print,
                "type": variable, message, or value
            }
        '''
        final_parameters = {}

        if parameters['VAR'] is not None:
            final_parameters['variable'] = parameters['VAR']
            final_parameters['type'] = 'variable'

        elif parameters['MESSAGE'] is not None:
            final_parameters['variable'] = parameters['MESSAGE']
            final_parameters['type'] = 'message'

        elif parameters['VAL'] is not None:
            final_parameters['variable'] = self.__map_values(
                parameters['VAL'], type=None)
            final_parameters['type'] = 'value'

        return final_parameters

    # DONE: needs revising gamed
    def post_process_conditional_operation(self, parameters):
        '''
            tags are:
            - LHS: list of LHSs
            - RHS: list of RHSs
            - CONDITION
            - LOG: logical operator for compound conditions     

            len(LHS) should be equal to len(RHS) if condition at the same index in not "not" otherwise len(RHS) > len(LHS)

            final format:
            [
                {
                    "keyword": "if",
                    "condition": [
                        {
                            "left": "x",
                            "operator": ">",
                            "right": "5"
                        },
                        {
                            "logicalOperator": "and",
                            "left": "x",
                            "operator": ">",
                            "right": "5"
                        }
                    ]
                }
            ]
        '''

        final_parameters = {}
        final_parameters['keyword'] = 'if'
        final_parameters['condition'] = []

        # foe now they should be equal
        if len(parameters['LHS']) != len(parameters['RHS']) != len(parameters['CONDITION']):
            # do something
            pass

        for lhs, rhs, condition in list(zip(parameters['LHS'], parameters['RHS'], parameters['CONDITION'])):
            final_parameters['condition'].append(
                {
                    'left': lhs,
                    'operator': self.__map_condition(condition),
                    'right': rhs
                }
            )

        return final_parameters

    # DONE: git operation intent
    def post_process_git_operation(self, parameters):
        '''
            the tags for git operation are:
            - ACTION : the operation to perform
            - MESSAGE : the message for the commit operation

            available commands:
            - discard
            - pull
            - push (commit and push) -> message is required
            - stage
            - stash
        '''
        final_parameters = {}

        # TODO: map actions to the correct git commands
        final_parameters['action'] = self.__map_actions(
            parameters['ACTION'], 'git')

        if final_parameters['action'] == 'push':
            final_parameters['message'] = parameters['MESSAGE']

        return final_parameters

    # DONE file system intent ish
    def post_process_file_system(self, parameters):
        '''
            the tags for file system operation are:
            - ACTION : the operation to perform
            - FILE : the file name
            - DIR : the message for the operation

            available commands:
            - create
            - delete
            - read
            - write
        '''
        final_parameters = {}

        action = self.__map_actions(parameters['ACTION'], 'file')
        final_parameters['action'] = parameters['ACTION']

        if action == 'create':  # could be file or directory
            if parameters['FILE'] is not None:
                # {
                #     "fileName": "test file - ahmed",
                #     "extension": ".txt",
                #     "content": "n"
                # }
                final_parameters['fileName'] = parameters['FILE']
                if parameters['DIR'] is not None:
                    final_parameters['directory'] = parameters['DIR']
                final_parameters['extension'] = self.__get_file_extension()
                final_parameters['content'] = None

            elif parameters['DIR'] is not None:  # then create directory
                # final format:
                # {
                #     "name" : "new_folder/c"
                # }
                final_parameters['name'] = parameters['DIR']

        elif action == 'delete':
            # {
            #  "source" : "aa.a"
            # }
            final_parameters['source'] = parameters['FILE']

        elif action == 'copy':
            # {
            #  "source" : "aa.b",
            #  "destination" : "copied.py"
            # }
            if parameters['FILE'] is not None:
                final_parameters['source'] = parameters['FILE']
                final_parameters['destination'] = None

            elif parameters['DIR'] is not None:  # then create directory
                final_parameters['source'] = parameters['DIR']
                final_parameters['destination'] = None

        elif action == 'rename':
            # final format
            #  {
            #     "source": "test file - ahmed.txt",
            #     "destination": "renamed - ahmed"
            # }
            final_parameters['source'] = parameters['FILE']
            final_parameters['destination'] = parameters['FILE']

        return final_parameters

    # DONE: interactive commands
    def post_process_interactive_commands(self, parameters):
        '''
            tags for interactive commands:
            - action
            - type: file | folders | code | functions | classes
        '''
        # map to single type
        final_parameters = {'type': parameters['TYPE']}

        return final_parameters

    # DONE: ide operations
    def post_process_ide_operation(self, parameters):
        '''
            the tags for the ide operation:
            - ACTION: in general : undo | redo | copy | paste | find | cut | run
                type specific:
                    file: goto 
                    line: goto | select | copy | paste
                    terminal: new | kill | focus

            - TYPE -> file | terminal | line 
            - LINE -> numbers
            - FILE -> filename
        '''
        final_parameters = {}

        action = self.__map_actions(parameters['ACTION'], 'ide')
        final_parameters['action'] = action

        operation_type = self.__map_ide_type(parameters['TYPE'])

        if operation_type == 'terminal':
            final_parameters['type'] = 'terminal'

        elif operation_type == 'file':
            # possible actions
            # goto
            # {
            #  "path" : "aa.x"
            # }
            final_parameters['type'] = 'file'
            final_parameters['path'] = parameters['FILE']

        elif operation_type == 'line':
            # possible actions
            # goto line
            # select line (one or multiple (range))
            # copy
            # paste
            # {
            #     "line": 5,
            #     "character" :6
            # }
            final_parameters['type'] = 'line'
            if len(parameters['LINE'] > 1):
                # {
                #     "startLine":4,
                #     "startCharacter": 0,
                #     "endLine": 6,
                #     "endCharacter": 10
                # }
                final_parameters['startLine'] = parameters['LINE'][0]
                final_parameters['endLine'] = parameters['LINE'][1]
                final_parameters['startCharacter'] = None
                final_parameters['endCharacter'] = None
            else:
                final_parameters['line'] = parameters['LINE']
                final_parameters['character'] = None

        return final_parameters

    # DONE but not done in the command execution module
    def post_process_array_operation(self, parameters):
        '''
            tags:
            OPERATION
            ARRAY
            ELEMENT (optional)
        '''
        final_parameters = {}

        final_parameters['function'] = self.__map_array_function(
            parameters['OPERATION'])
        final_parameters['element'] = self.__map_values(parameters['ELEMENT'])
        final_parameters['array'] = parameters['ARRAY']
