import utils
import re

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

    def post_process(self, sentence, intent, tags):
        '''
            post_process function is used to post process the output of the two models.
            It takes the detected intent along with the entities and maps them to the final output.
            The final output is then sent to the command execution module using the API.
            The main purpose is to send th ecorrect values with the correct types to the command execution module.
        '''
        sentence = sentence.split()

        intent_tags = self.intent_to_tags[intent]

        parameters = self.__get_parameters(intent, sentence, tags, intent_tags)

        final_parameters = None

        match intent:
            case 'variable declaration':
                final_parameters = self.post_process_declaration(parameters)

            case 'constant declaration':
                final_parameters = self.post_process_declaration(parameters)

            case 'function declaration':
                final_parameters = self.post_process_function_declaration(
                    parameters)

            case 'class declaration':
                final_parameters = self.post_process_class_declaration(
                    parameters)

            case 'for loop':
                final_parameters = self.post_process_for_loop(parameters)

            case 'while loop':
                final_parameters = self.post_process_while_loop(parameters)

            case 'assignment operation':
                final_parameters = self.post_process_assignment_operation(
                    parameters)
                
            case 'bitwise operation':
                final_parameters = self.post_process_bitwise_operation(
                    parameters)
                
            case 'casting':
                final_parameters = self.post_process_casting(parameters)

            case 'input':
                final_parameters = self.post_process_input(parameters)

            case 'output':
                final_parameters = self.post_process_output(parameters)

            case 'assertion':
                final_parameters = self.post_process_assertion(parameters)

            case 'libraries':
                final_parameters = self.post_process_libraries(parameters)

            case 'comment':
                final_parameters = self.post_process_comment(parameters)

            case 'conditional operation':
                pass

            case 'file system':
                pass

            case 'ide operation':
                pass

            case 'interactive commands':
                pass
            case 'git operation':
                pass

            case 'array operation':
                pass

            case 'mathematical operation':
                pass

            case 'membership operation':
                pass
            

        # print(f"Final parameters: {final_parameters}")

        return final_parameters

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
            # if not the previosu then the type is probably string
            # if the value is empty and the type is not specified -> leave it as None
            check = re.match(integer_regex, value)
            if re.match(integer_regex, value) is not None:
                final_value = int(value)
            elif re.match(float_regex, value) is not None:
                final_value = float(value)
            elif re.match(boolean_regex, value) is not None:
                final_value = bool(value.capitalize())
            else:
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

    def __map_bitwise_operator(self, operator):

        final_operator = None

        for key, value in utils.BITWISE_OPERATORS.items():
            if key in operator:
                final_operator = value
                break

        return final_operator

    # DONE: variable and constant declaration
    def post_process_declaration(self, parameters):
        '''
            This function is used to post process the variable declaration intent to send to the execution
            we need to map the tags to the correct format.
            and check the type to get the corrct value.
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
            This function is used to post process the funciotn declaration intent.
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
            final_parameters['iterators'] = [parameters['VAR']
                                             if parameters['VAR'] is not None else 'i']
            final_parameters['iterable'] = parameters['COLLECTION']
            final_parameters['body'] = None

        else:
            final_parameters['type'] = 'range'
            final_parameters['iterators'] = [parameters['VAR']
                                             if parameters['VAR'] is not None else 'i']
            final_parameters['start'] = parameters['START'] if parameters['START'] is not None else '0'
            # must be provided by the user
            final_parameters['end'] = parameters['END']
            final_parameters['step'] = parameters['STEP'] if parameters['STEP'] is not None else '1'

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
            The tags for the assignemtn intent:
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
            the tags for librares intent:
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
    def post_process_bitwise_operation(self, parameters):
        '''
            intent tags are:
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
                final_parameters['operator'] = self.__map_bitwise_operator(
                    parameters['OPERATOR'])
        else:
            final_parameters['left'] = parameters['OPERAND'][0]
            final_parameters['right'] = parameters['OPERAND'][1]
            final_parameters['operator'] = self.__map_bitwise_operator(
                parameters['OPERATOR'])

        return final_parameters

    # DONE: comment intetn
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

            should be atmost 1

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

    def post_process_file_system(self, parameters):
        pass

    def post_process_ide_operation(self, parameters):
        pass

    def post_process_interactive_commands(self, parameters):
        pass

    def post_process_git_operation(self, parameters):
        pass

    def post_process_mathematical_operation(self, parameters):
        pass

    def post_process_membership_operation(self, parameters):
        pass

    # not done in the command execution module
    def post_process_array_operation(self, parameters):
        pass

