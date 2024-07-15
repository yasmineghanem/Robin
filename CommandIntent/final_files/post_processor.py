# import command_constants
import re
import command_constants
from api import APIController
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

    def post_process(self, sentence, intent, tags=None, fallback=False):
        '''
            post_process function is used to post process the output of the two models.
            It takes the detected intent along with the entities and maps them to the final output.
            The final output is then sent to the command execution module using the API.
            The main purpose is to send th correct values with the correct types to the command execution module.
        '''

        print("fallback from post process", fallback)

        sentence = sentence.split()

        # intent_tags = self.intent_to_tags[intent]
        intent_tags = None

        parameters = tags if fallback else self.__get_parameters(
            intent, sentence, tags, intent_tags)

        final_parameters = None

        response = None

        match intent:
            case 'variable declaration':

                final_parameters = self.post_process_declaration(
                    parameters, fallback=fallback)

                response = self.api.declare_variable(final_parameters)

            case 'constant declaration':

                final_parameters = self.post_process_declaration(
                    parameters, fallback=fallback)

                response = self.api.declare_constant(final_parameters)

            case 'function declaration':
                final_parameters = self.post_process_function_declaration(
                    parameters, fallback=fallback)

                response = self.api.declare_function(final_parameters)

            case 'class declaration':
                final_parameters = self.post_process_class_declaration(
                    parameters, fallback=fallback)

                response = self.api.declare_class(final_parameters)

            case 'for loop':
                final_parameters = self.post_process_for_loop(
                    parameters, fallback=fallback)
                response = self.api.for_loop(final_parameters)

            case 'while loop':
                final_parameters = self.post_process_while_loop(
                    parameters, fallback=fallback)
                response = self.api.while_loop(final_parameters)

            case 'assignment operation':
                final_parameters = self.post_process_assignment_operation(
                    parameters, fallback=fallback)
                response = self.api.assign_variable(final_parameters)

            case 'bitwise operation':
                final_parameters = self.post_process_operation(
                    parameters, intent, fallback=fallback)
                response = self.api.operation(final_parameters)

            case 'casting':
                final_parameters = self.post_process_casting(
                    parameters, fallback=fallback)
                response = self.api.type_casting(final_parameters)

            case 'input':
                final_parameters = self.post_process_input(
                    parameters, fallback=fallback)
                response = self.api.user_input(final_parameters)

            case 'output':
                final_parameters = self.post_process_output(
                    parameters, fallback=fallback)
                response = self.api.print_code(final_parameters)

            case 'assertion':
                final_parameters = self.post_process_assertion(
                    parameters, fallback=fallback)
                response = self.api.assertion(final_parameters)

            case 'libraries':
                final_parameters = self.post_process_libraries(
                    parameters, fallback=fallback)
                response = self.api.import_library(final_parameters)

            case 'comment':
                final_parameters = self.post_process_comment(
                    parameters, fallback=fallback)
                response = self.api.line_comment(final_parameters)

            case 'conditional statement':
                final_parameters = self.post_process_conditional_operation(
                    parameters, intent, fallback=fallback)
                response = self.api.conditional(final_parameters)

            case 'membership operation':
                final_parameters = self.post_process_operation(
                    parameters, intent, fallback=fallback)
                response = self.api.conditional(final_parameters)

            case 'mathematical operation':
                final_parameters = self.post_process_operation(
                    parameters, intent, fallback=fallback)
                response = self.api.operation(final_parameters)

            case 'file system':
                final_parameters = self.post_process_file_system(
                    parameters, fallback=fallback)
                # response = self.api.file_system(final_parameters)

            case 'git operation':
                final_parameters = self.post_process_git_operation(
                    parameters, fallback=fallback)
                response = self.api.git(final_parameters)

            case 'interactive commands':
                final_parameters = self.post_process_interactive_commands(
                    parameters, fallback=fallback)
                # response = self.api.interactive_commands(final_parameters)

            case 'ide operation':
                final_parameters = self.post_process_ide_operation(
                    parameters, fallback=fallback)

                response = self.api.ide_operation(final_parameters)

            case 'array operation':
                final_parameters = self.post_process_array_operation(
                    parameters)
                # response = self.api.operation(final_parameters)

            case 'activate interactive':
                final_parameters = self.post_process_fallback(parameters, intent)
                # response = self.api.interactive_commands(final_parameters)

            case 'activate mouse':
                final_parameters = self.post_process_fallback(
                    parameters, intent)
                # response = self.api.interactive_commands(final_parameters)

            case 'mouse click':
                final_parameters = self.post_process_fallback(
                    parameters, intent)
                # response = self.api.interactive_commands(final_parameters)

            case 'exit block':
                # final_parameters = self.post_process_fallback(
                #     parameters, intent)
                response = self.api.exit_scope()
        print("Final parameters: ", final_parameters)

        return response

    def __get_parameters(self, intent, sentence, tags, intent_tags):
        '''
            This function is used to get the parameters of the detected entities.
            The function takes the indices of the B-tags and the tags of the entities and the sentence.
            It returns a dictionary of the parameters of the detected entities.
        '''
        intent_tags = self.intent_to_tags[intent]
        # print(f"Intent tags: {intent_tags}")
        parameters = {tag: None for tag in intent_tags if tag != 'O'}

        # if intent == 'function declaration':
        #     parameters['PARAM'] = []
        # elif intent == 'bitwise operation' or intent == 'mathematical operation':
        #     parameters['OPERAND'] = []
        # elif intent == 'ide operation':
        #     parameters['LINE'] = []

        for tag in intent_tags:
            if tag == 'O':
                continue

            B_indices = [index for index, entity in enumerate(
                tags) if entity.startswith('B-' + tag)]

            # if tag != 'OPERAND' and tag != 'LINE':
            #     if (len(B_indices) > 1):
            #         B_indices = B_indices[:1]

            if len(B_indices) != 0:
                parameters[tag] = []

            # print(f"Tag: {tag}, B_indices: {B_indices}")

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

                # print(f"Full entity: {full_entity}")

                # if tag == 'PARAM' or tag == 'OPERAND' or tag == 'LINE':
                parameters[tag].append(' '.join(full_entity))
                # else:
                #     parameters[tag] = ' '.join(full_entity)

                # print(f"Parameters: {parameters}")

        # print(f"Parameters: {parameters}")
        return parameters

    def __map_values(self, value, var_type=None):
        final_value = None

        integer_regex = r'^-?\d+$'
        float_regex = r'^-?\d*\.\d+$'
        boolean_regex = r'^([Tt]rue|[Ff]alse)$'
        print(var_type)
        print(value)
        # try:
        if var_type is not None:
            # if var_type in command_constants.TYPES['int']:
            #     final_value = int(value)
            # elif var_type in command_constants.TYPES['float']:
            #     final_value = float(value)
            # elif var_type in command_constants.TYPES['double']:
            #     final_value = float(value)
            # elif var_type in command_constants.TYPES['str']:
            #     final_value = value
            # elif var_type in command_constants.TYPES['bool']:
            #     final_value = bool(value)
            # elif var_type in command_constants.TYPES['list']:
            #     final_value = []
            # elif var_type in command_constants.TYPES['dictionary']:
            #     final_value = {}
            # elif var_type in command_constants.TYPES['tuple']:
            #     final_value = tuple()
            # elif var_type in command_constants.TYPES['set']:
            #     final_value = set()

            for key, values in command_constants.TYPES.items():
                if var_type in values:
                    if key == 'Integer':
                        final_value = int(value)
                    elif key == 'Float':
                        final_value = float(value)
                    elif key == 'String':
                        final_value = value
                    elif key == 'Boolean':
                        final_value = bool(value.capitalize())
                    elif key == 'List':
                        final_value = []
                    elif key == 'Dictionary':
                        final_value = {}
                    elif key == 'Tuple':
                        final_value = tuple()
                    elif key == 'Set':
                        final_value = set()
                    break
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

    def __map_type(self, var_type):

        final_type = None

        for key, values in command_constants.TYPES.items():
            if var_type in values:
                final_type = key
                break

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

            for operator, values in command_constants.CONDITIONS.items():
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
                    for value in command_constants.CONDITIONS['==']:
                        if value in condition:
                            equal_present = True
                            break

                    if not_present and equal_present:
                        final_operator = 'NotEqual'
                    elif not_present:
                        final_operator = 'Not'

                    if final_operator is not None:
                        break

                elif operator == '==':

                    for value in values:
                        if value in condition:
                            final_operator = 'Equal'
                            break

                    if final_operator is not None:
                        break

                # processing for the > or >= conditions
                elif operator == '>' or operator == '<':
                    # if the operator is ! then the other values are for the == and vice vers
                    for value in values:
                        if value in condition:
                            final_operator = 'GreaterThan' if operator == '>' else 'LessThan'
                            break

                    # check if theres equal as well
                    if final_operator is not None:
                        for value in command_constants.CONDITIONS['==']:
                            if value == 'is':
                                continue
                            if value in condition:
                                final_operator += 'OrEqual'
                                break

                    if final_operator is not None:
                        break

        print(f"Final operator: {final_operator}")

        return final_operator

    def __map_operator(self, operator, intent):

        operators = command_constants.OPERATORS[intent]
        final_operator = None

        if intent in ['bitwise', 'membership', 'conditional']:
            for key, values in operators.items():
                if operator in values:
                    final_operator = key
                    break

        elif intent == 'mathematical':
            for key, values in operators.items():
                for value in values:
                    if operator in value:
                        final_operator = key
                        break

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

        intent_actions = command_constants.ACTIONS[intent]

        for key, values in intent_actions.items():
            for value in values:
                if action in values:
                    final_action = key
                    break

            if final_action is not None:
                break

        return final_action

    def __map_ide_type(self, operation_type):
        final_type = None
        if 'line' in operation_type:
            final_type = 'line'
        elif 'file' in operation_type:
            final_type = 'file'
        else:
            final_type = 'directory'
        return final_type

    def __is_number(self, string):
        return bool(re.fullmatch(r'\d+(\.\d+)?', string))

    def __get_file_extension(self):
        return None

    def __handle_conditions(self, lhs, rhs, conditions):
        condition = {
            'left': None,
            'condition': None,
            'right': None
        }

        if lhs is not None and rhs is not None:
            condition['left'] = lhs[0]
            condition['right'] = rhs[0]

        elif lhs is not None:
            condition['left'] = lhs[0]
            if len(lhs) > 1:
                condition['right'] = lhs[1]

        elif rhs is not None:
            if len(rhs) > 1:
                condition['right'] = rhs[1]
                condition['left'] = rhs[0]
            else:
                condition['left'] = rhs[0]

        if conditions is not None:
            condition['operator'] = self.__map_condition(conditions[0])

    # DONE: variable and constant declaration
    def post_process_declaration(self, parameters, fallback=False):
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
        # post process the variable declaration before assigning
        # TODO: handle inconsistencies
        # ideal case => all are present:
        # values will be VAR[0], VAL[0], TYPE[0] all will have length = 1
        # next best case is to have VAR not None and either VAL or TYPE not None
        # if VAL and TYPE are None then handle the error => check if VAR has more than 1 entry probably the VAL was detected as VAR
        # if VAR is None and VAL is not None check for multiple entries

        final_parameters = {
            'name': None,
            'value': None,
            'type': None
        }

        print("fall back from post process declaration", fallback, parameters)
        if fallback:
            for key, value in parameters.items():
                print(key, value)
                if key == 'type':
                    final_parameters[key] = self.__map_type(value)
                elif key == 'value':
                    final_parameters[key] = self.__map_values(value)
                else:
                    final_parameters[key] = value

            print(final_parameters)

            return final_parameters

        # case 1(ideal case): all are present
        if parameters['VAR'] is not None and parameters['VAL'] is not None and parameters['TYPE'] is not None:
            final_parameters['name'] = parameters['VAR'][0]
            final_parameters['value'] = self.__map_values(
                parameters['VAL'][0], parameters['TYPE'][0])
            final_parameters['type'] = self.__map_type(parameters['TYPE'][0])
            # return final_parameters

        # case 2(2nd best): VAR is present with either VAL or TYPE
        elif parameters['VAR'] is not None:
            print("alooooo")
            final_parameters['name'] = parameters['VAR'][0]

            if parameters['VAL'] is not None:  # then type is None just send the value
                final_parameters['value'] = self.__map_values(
                    parameters['VAL'][0], None)

            # then value is None check for multiple values then just send the type if there is only one
            elif parameters['TYPE'] is not None:
                # map and set the type
                final_parameters['type'] = self.__map_type(
                    parameters['TYPE'][0])

                # first check for multiple VARS
                if len(parameters['VAR']) > 1:
                    print('hena')
                    parameters['VAL'] = parameters['VAR'][1:]
                    # parameters['VAR'] = parameters['VAR'][0] # the variable will most likely be the first one
                    final_parameters['value'] = self.__map_values(
                        parameters['VAL'][0], None)

            else:  # both VAL and TYPE are None => check for multiple VALs

                if len(parameters['VAR']) > 1:
                    print('hena')
                    parameters['VAL'] = parameters['VAR'][1:]
                    # parameters['VAR'] = parameters['VAR'][0] # the variable will most likely be the first one
                    final_parameters['value'] = self.__map_values(
                        parameters['VAL'][0], None)

        # case 3: VAR is not present (worst case) should raise an error in execution
        # can be either because the user didn't mention the variable name or the NER model didn't detect it
        # check for multiple VALs
        elif parameters['VAR'] is None:
            # check for multiple VALs
            if parameters['TYPE'] is not None:
                final_parameters['type'] = self.__map_type(
                    parameters['TYPE'][0])

            if parameters['VAL'] is not None:
                if len(parameters['VAL']) > 1:
                    # get the most likely variable name
                    parameters['VAR'] = parameters['VAL'][0]
                    parameters['VAL'] = parameters['VAL'][1:]

                    # set the final parameters
                    final_parameters['value'] = self.__map_values(
                        parameters['VAL'][0], final_parameters['type'])
                    final_parameters['name'] = parameters['VAR']
                else:
                    final_parameters['value'] = self.__map_values(
                        parameters['VAL'][0], final_parameters['type'])

        final_parameters['type'] = final_parameters['type'].capitalize(
        ) if final_parameters['type'] is not None else None

        print(final_parameters)

        return final_parameters

    # DONE: mapping final parameters
    def post_process_function_declaration(self, parameters, fallback=False):
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
        final_parameters = {
            'name': None,
            'parameters': [],
            'return_type': None
        }

        if parameters['FUNC'] is not None:
            final_parameters['name'] = parameters['FUNC'][0]

        for param in parameters['PARAM']:
            final_parameters['parameters'].append({'name': param})

        if parameters['TYPE'] is not None:
            final_parameters['return_type'] = self.__map_type(
                parameters['TYPE'][0])

        return final_parameters

    # DONE: class declaration intent
    def post_process_class_declaration(self, parameters, fallback=False):
        '''
            the tags are:
            CLASS : the class name

            final format:
            {
                "name": class_name
            } 
        '''
        final_parameters = {
            'name': None
        }
        if fallback:
            final_parameters['name'] = parameters['name']
            return final_parameters
        if parameters['CLASS'] is not None:
            final_parameters['name'] = parameters['CLASS'][0]

        return final_parameters

    # DONE: for loop intent #TODO: MSH 3ARFA EH EL 7ASALAHA
    def post_process_for_loop(self, parameters, fallback=False):
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
        # post processing of returned list
        print("for loop parameters")

        final_parameters_range = {
            'type': 'Range',
            'iterators': [],
            'start': None,
            'end': None,
            'step': None
        }
        final_parameters_iterable = {
            'type': 'Iterable',
            'iterators': [],
            'iterable': None,
            'body': None
        }

        loop_type = True  # true for range and false for iterable

        if fallback:
            if 'iterable' in parameters.keys():  # iterable loop type
                for key, value in parameters.items():
                    if key == 'iterator':
                        final_parameters_iterable['iterators'].append(value)
                    else:
                        final_parameters_iterable[key] = value
                return final_parameters_iterable

            for key, value in parameters.items():
                print(key, value)
                if key in ['start', 'end', 'step']:
                    value = self.__map_values(value)
                    final_parameters_range[key] = value
                else:
                    final_parameters_range['iterators'].append(value)
            return final_parameters_range

        if parameters['END'] is not None:
            # check if the end is a number
            if self.__is_number(parameters['END'][0]):
                final_parameters_range['end'] = self.__map_values(
                    parameters['END'][0])

                if parameters['VAR'] is not None:
                    final_parameters_range['iterators'].append(
                        parameters['VAR'][0])
                else:
                    final_parameters_range['iterators'].append('i')

                if parameters['START'] is not None and self.__is_number(parameters['START'][0]):
                    # the start and the step are optional
                    # check if the start is a number
                    final_parameters_range['start'] = self.__map_values(
                        parameters['START'][0])

                # check if the step is a number
                if parameters['STEP'] is not None and self.__is_number(parameters['STEP'][0]):
                    final_parameters_range['step'] = self.__map_values(
                        parameters['STEP'][0])

                return final_parameters_range

            else:
                if parameters['START'] is None or parameters['STEP'] is None:
                    parameters['COLLECTION'] = parameters['END']
                    loop_type = False

        if loop_type:  # check if start or step are not none
            if parameters['STEP'] is not None and self.__is_number(parameters['STEP'][0]):
                final_parameters_range['end'] = self.__map_values(
                    parameters['STEP'][0])

            elif parameters['START'] is not None and self.__is_number(parameters['START'][0]):
                final_parameters_range['end'] = self.__map_values(
                    parameters['START'][0])

            if parameters['VAR'] is not None:
                final_parameters_range['iterators'].append(
                    parameters['VAR'][0])
            else:
                final_parameters_range['iterators'].append('i')

        if parameters['COLLECTION'] is not None:
            loop_type = False
            final_parameters_iterable['type'] = 'iterable'
            if parameters['VAR'] is not None:
                final_parameters_iterable['iterators'].append(
                    parameters['VAR'][0])
            else:
                final_parameters_iterable['iterators'].append('i')

            final_parameters_iterable['iterable'] = parameters['COLLECTION'][0]

        if loop_type:
            return final_parameters_range

        return final_parameters_iterable

    # DONE: while loop intent
    def post_process_while_loop(self, parameters, fallback=False):
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
        # check if the LHS and RHS are detected
        # sometimes the LHS get detected as RHS and vice versa
        final_parameters = {
            'condition': []
        }

        condition = {
            'left': None,
            'operator': None,
            'right': None
        }

        if fallback:
            if 'lhs' in parameters.keys():
                condition['left'] = self.__map_values(parameters['lhs'])
            if 'condition' in parameters.keys():
                condition['operator'] = self.__map_condition(
                    parameters['condition'])

            condition['right'] = self.__map_values(parameters['rhs'])

            final_parameters['condition'].append(condition)
            return final_parameters

        if parameters['LHS'] is not None and parameters['RHS'] is not None:
            condition['left'] = parameters['LHS'][0]
            condition['right'] = parameters['RHS'][0]

        elif parameters['LHS'] is not None:
            condition['left'] = parameters['LHS'][0]
            if len(parameters['LHS']) > 1:
                condition['right'] = parameters['LHS'][1]

        elif parameters['RHS'] is not None:
            if len(parameters['RHS']) > 1:
                condition['right'] = parameters['RHS'][1]
                condition['left'] = parameters['RHS'][0]
            else:
                condition['left'] = parameters['RHS'][0]

        if parameters['CONDITION'] is not None:
            condition['operator'] = self.__map_condition(
                parameters['CONDITION'][0])

        final_parameters['condition'].append(condition)

        return final_parameters

    # DONE: casting intent
    def post_process_casting(self, parameters, fallback=False):
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
        final_parameters = {
            'variable': None,
            'type': None
        }

        if fallback:
            final_parameters['type'] = self.__map_type(parameters['type'])
            final_parameters['variable'] = parameters['variable']
            return final_parameters

        if parameters['VAR'] is not None:
            final_parameters['variable'] = parameters['VAR'][0]

        if parameters['TYPE'] is not None:
            final_parameters['type'] = self.__map_type(parameters['TYPE'][0])

        return final_parameters

    # DONE: assignment operation intent
    def post_process_assignment_operation(self, parameters, fallback=False):
        '''
            The tags for the assignment intent:
            - LHS -> always has to be a variable
            - RHS -> could be a variable or value (need to map to the correct type)

            final format:
            {
                "name": LHS,
                "type": "=", (is always =)
                "value": RHS
            }
        '''

        final_parameters = {
            'name': None,
            'type': '=',
            'value': None
        }

        if fallback:
            final_parameters['name'] = parameters['lhs']
            final_parameters['value'] = self.__map_values(parameters['rhs'])

            return final_parameters

        if parameters['LHS'] is not None and parameters['RHS'] is not None:
            final_parameters['name'] = parameters['LHS'][0]
            final_parameters['value'] = parameters['RHS'][0]

        elif parameters['LHS'] is not None:
            final_parameters['name'] = parameters['LHS'][0]
            if len(parameters['LHS']) > 1:
                final_parameters['value'] = parameters['LHS'][1]

        elif parameters['RHS'] is not None:
            final_parameters['name'] = parameters['RHS'][0]
            if len(parameters['RHS']) > 1:
                final_parameters['value'] = parameters['RHS'][1]

        return final_parameters

    # DONE: assertion intent
    def post_process_assertion(self, parameters, fallback=False):
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
        # TODO: process variables and values => make it LHS and RHS
        # check if VAR is None -> then check if VAl detected more than once

        final_parameters = {
            'variable': None,
            'type': None,
            'value': None
        }

        if fallback:
            final_parameters['variable'] = parameters['lhs']
            final_parameters['type'] = self.__map_condition(
                parameters['condition'])
            final_parameters['value'] = self.__map_values(parameters['rhs'])
            return final_parameters

        if parameters['CONDITION'] is not None:
            final_parameters['type'] = self.__map_condition(
                parameters['CONDITION'][0])

        if parameters['VAR'] is not None:
            final_parameters['variable'] = parameters['VAR'][0]
        else:
            # check the length of values
            if len(parameters['VAL']) > 1:
                # parameters['VAR'] = parameters['VAL'][0]
                final_parameters['variable'] = parameters['VAL'][0]

                parameters['VAL'] = parameters['VAL'][1:]

        if parameters['VAL'] is not None:
            final_parameters['value'] = self.__map_values(
                parameters['VAL'][0], None)
        else:
            # check the length of VAR
            if len(parameters['VAR']) > 1:
                parameters['VAL'] = parameters['VAR'][1:]
                parameters['VAR'] = parameters['VAR'][0]
                final_parameters['value'] = self.__map_values(
                    parameters['VAL'][0], None)

        return final_parameters

    # DONE: libraries intent
    def post_process_libraries(self, parameters, fallback=False):
        '''
            the tags for libraries intent:
            - LIB_NAME : the name of the library

            final format:
            {
                "library": "sklearn"
            }
        '''
        final_parameters = {
            'library': None
        }

        if fallback:
            final_parameters = parameters
            return final_parameters

        if parameters['LIB_NAME'] is not None:
            final_parameters['library'] = parameters['LIB_NAME'][0]

        return final_parameters

    # DONE: operations intent
    def post_process_operation(self, parameters, intent, fallback=False):
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
        final_parameters = {
            'variable': None,
            'right': None,
            'operator': None,
            'left': None
        }

        if fallback:
            if 'variable' in parameters.keys():
                final_parameters['variable'] = parameters['variable']
                pass
            final_parameters['left'] = self.__map_values(parameters['lhs'])
            final_parameters['right'] = self.__map_values(parameters['rhs'])
            final_parameters['operator'] = self.__map_operator(
                parameters['operation'], intent.split()[0])

            return final_parameters

        if parameters['OPERAND'] is not None and parameters['OPERATOR'] is not None:
            # map the correct operator
            final_parameters['operator'] = self.__map_operator(
                parameters['OPERATOR'][0], intent.split()[0])

            # check if the operator is not
            if final_parameters['operator'] == 'Not':
                final_parameters['right'] = parameters['OPERAND'][0]
                final_parameters['left'] = None
            else:
                # should be the case if the operator is not not
                if len(parameters['OPERAND']) == 2:
                    final_parameters['left'] = parameters['OPERAND'][0]
                    final_parameters['right'] = parameters['OPERAND'][1]
                else:
                    final_parameters['left'] = parameters['OPERAND'][0]
                    final_parameters['right'] = parameters['OPERAND'][0]

            if intent == 'mathematical operation':
                if parameters['VAR'] is not None:
                    final_parameters['variable'] = parameters['VAR'][0]
                else:
                    final_parameters['variable'] = final_parameters['left']

        print(final_parameters)
        return final_parameters

    # DONE: comment intent
    def post_process_comment(self, parameters, fallback=False):
        '''
            This will be the line comment 
                tags:
                COMMENT

            final format:
            {
                "content": "This is a one line comment"
            }
        '''
        # process the data
        # join the strings of the comment array together
        final_parameters = {
            'content': None
        }

        if fallback:
            final_parameters['content'] = parameters['comment']
            return final_parameters

        if parameters['COMMENT'] is not None:
            final_parameters['content'] = ' '.join(parameters['COMMENT'])

        return final_parameters

    # DONE: input intent
    def post_process_input(self, parameters, fallback=False):
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

        final_parameters = {
            'variable': None,
            'message': None
        }

        if fallback:
            if 'variable' in parameters.keys():
                final_parameters['variable'] = parameters['variable']
            if 'message' in parameters.keys():
                final_parameters['message'] = parameters['message']

            return final_parameters

        if parameters['VAR'] is not None:
            final_parameters['variable'] = parameters['VAR'][0]

        if parameters['MESSAGE'] is not None:
            final_parameters['message'] = parameters['MESSAGE'][0]

        return final_parameters

    # DONE: output intent
    def post_process_output(self, parameters, fallback=False):
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
        final_parameters = {
            'variable': None,
            'type': None
        }

        if fallback:
            for key, value in parameters.items():
                final_parameters['type'] = parameters['type']
                if key in ['variable', 'message']:
                    final_parameters['variable'] = value

                elif key in 'value':
                    final_parameters['variable'] = self.__map_values(value)

            return final_parameters

        if parameters['VAR'] is not None:
            final_parameters['variable'] = parameters['VAR'][0]
            final_parameters['type'] = 'variable'

        # TODO: idea (we can check the word before each message)
        elif parameters['MESSAGE'] is not None:
            final_parameters['variable'] = ' '.join(parameters['MESSAGE']) if len(
                parameters['MESSAGE']) > 1 else parameters['MESSAGE'][0]
            final_parameters['type'] = 'message'

        elif parameters['VAL'] is not None:
            final_parameters['variable'] = self.__map_values(
                parameters['VAL'][0], type=None)
            final_parameters['type'] = 'value'

        return final_parameters

    # TODO: needs revising gamed
    def post_process_conditional_operation(self, parameters, intent, fallback=False):
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

        final_parameters = {
            'keyword': 'if',
            'condition': []
        }

        if fallback:
            # keys = parameters.keys()
            lhss = [value for key, value in parameters.items() if 'lhs' in key]
            rhss = [value for key, value in parameters.items() if 'rhs' in key]
            conditions = [value for key,
                          value in parameters.items() if 'condition' in key]
            operators = [value for key, value in parameters.items()
                         if 'operator' in key]
            print(lhss, rhss, conditions, operators)
            for index, (lhs, rhs, condition) in enumerate(list(zip(lhss, rhss, conditions))):
                if index == 0:
                    final_parameters['condition'].append(
                        {
                            'left': lhs,
                            'operator': self.__map_condition(condition),
                            'right': rhs
                        }
                    )
                else:
                    final_parameters['condition'].append(
                        {
                            "logicalOperator": self.__map_operator(operators[index-1], intent.split()[0]),
                            'left': lhs,
                            'operator': self.__map_condition(condition),
                            'right': rhs
                        }
                    )
                index += 1

            return final_parameters

        # TODO change it to OPERAND?
        if parameters['LHS'] is None:
            pass
        if parameters['RHS'] is None:
            pass
        if parameters['CONDITION'] is None:
            pass
        # foe now they should be equal
        if len(parameters['LHS']) != len(parameters['RHS']) != len(parameters['CONDITION']):
            # do something
            # they all should be the same or one of the conditions is not
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
    def post_process_git_operation(self, parameters, fallback=False):
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
        final_parameters = {
            'action': None,
            'message': None
        }

        if fallback:
            final_parameters['action'] = self.__map_actions(
                parameters['action'], 'git')

            try:
                if final_parameters['action'] == 'push':
                    final_parameters['message'] = parameters['message']
            except:
                print("Message not provided")
            return final_parameters

        # TODO: map actions to the correct git commands
        if parameters['ACTION'] is not None:
            final_parameters['action'] = self.__map_actions(
                parameters['ACTION'][0], 'git')

        if final_parameters['action'] == 'push':
            final_parameters['message'] = parameters['MESSAGE'][0]

        # TODO: api call to git function not implemented yet

        print(final_parameters)
        return final_parameters

    # TODO file system intent ish
    def post_process_file_system(self, parameters, fallback=False):
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

        if fallback:
            final_parameters = parameters
            return final_parameters

        if parameters['ACTION'] is not None:
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
    def post_process_interactive_commands(self, parameters, fallback=False):
        '''
            tags for interactive commands:
            - action
            - type: file | folders | code | functions | classes
        '''
        # map to single type
        final_parameters = {
            'type': None
        }

        if fallback:
            final_parameters['type'] = parameters['type']
            return final_parameters

        if parameters['TYPE'] is not None:
            final_parameters['type'] = parameters['TYPE'][0]

            print(final_parameters)

        return final_parameters

    # DONE: ide operations
    def post_process_ide_operation(self, parameters, fallback=False):
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
        # final_parameters = {
        #     'type': None,
        #     'line': None,
        #     'file': None
        # }

        final_parameters = {}

        if fallback:
            return parameters

        if parameters['ACTION'] is not None:
            action = self.__map_actions(parameters['ACTION'][0], 'ide')
            final_parameters['action'] = action

        if parameters['TYPE'] is not None:
            operation_type = self.__map_ide_type(parameters['TYPE'][0])

            if operation_type == 'terminal':
                final_parameters['type'] = 'terminal'

            elif operation_type == 'file':
                # possible actions
                # goto
                # {
                #  "path" : "aa.x"
                # }
                final_parameters['type'] = 'file'

                if parameters['FILE'] is not None:
                    final_parameters['path'] = parameters['FILE'][0]

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
                if len(parameters['LINE']) > 1:
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
                    final_parameters['line'] = parameters['LINE'][0]
                    final_parameters['character'] = None

        print(final_parameters)
        return final_parameters

    # DONE but not done in the command execution module
    def post_process_array_operation(self, parameters):
        '''
            tags:
            OPERATION
            ARRAY
            ELEMENT (optional)
        '''
        final_parameters = {
            'operation': None,
            'array': None,
            'element': None
        }

        if parameters['OPERATION'] is not None:
            final_parameters['function'] = self.__map_array_function(
                parameters['OPERATION'][0])

        if parameters['ELEMENT'] is not None:
            final_parameters['element'] = self.__map_values(
                parameters['ELEMENT'][0])

        if parameters['ARRAY'] is not None:
            final_parameters['array'] = parameters['ARRAY'][0]

        print(final_parameters)
        return final_parameters

    def post_process_fallback(self, parameters, intent):
        '''
            the fallback ner  
        '''
        final_parameters = {}
        print(parameters)
        print(intent)
        action = None
        match intent:
            case 'activate interactive':
                action = parameters['action']
                actions = command_constants.ACTIONS['activation']
                for key, values in actions.items():
                    if action in values:
                        action = key
                        break
            case 'activate mouse':
                action = parameters['action']
                actions = command_constants.ACTIONS['activation']
                for key, values in actions.items():
                    if action in values:
                        action = key
                        break
            case 'mouse click':
                final_parameters = parameters
                try:
                    action = parameters['action']
                except:
                    action = None

        final_parameters['action'] = action

        print("Final parameters from fallback:", final_parameters)
        return final_parameters
