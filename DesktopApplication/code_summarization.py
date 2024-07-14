import json

'''

1- imports (numpy)

    We have [libraries] imported


2- constants (name, value)

    The constants declared are


3- classes (parameters, methods)

    We have [classes] defined


4- functions (name, params)

    We have [functions] defined


5- assignments

6- loops

7- conditionals

8- I/O

9- main function
'''


class ASTProcessor:
    def __init__(self, ast):
        self.ast = ast

    def process_ast(self):
        import_nodes, constant_nodes, code_nodes = self.extract_import_nodes()
        summary = []
        summary.append(import_nodes)
        summary.append(constant_nodes)
        for node in code_nodes:
            if node['type'] == "class_definition":
                info = self.process_class_node(node, class_info={})
                info['type'] = 'class_definition'
                summary.append(dict(info))

            elif node['type'] == "function_definition":
                method_info = self.process_function_node(node)
                method_info['type'] = 'function_definition'
                summary.append(dict(method_info))

            elif node['type'] == "expression_statement":
                assignment_info = self.process_expression_statement(node)
                assignment_info['type'] = 'expression_statement'
                summary.append(dict(assignment_info))

            elif node['type'] == "for_statement" or node['type'] == "while_statement":
                loop = self.process_loops(node)
                for l in loop:
                    summary.append(dict(l))

        return summary

    def extract_import_nodes(self, ):

        import_nodes = []
        constant_nodes = []
        code_nodes = []
        error_nodes = []

        for node in self.ast['children']:

            # if the type contains the word import
            if 'type' in node:
                node_type = node['type']
                # print(node)
                # break
                if node_type == 'ERROR':
                    # print(f"Syntax error in code at {node.start_point}: {node.text.decode('utf-8')}")
                    error_nodes.append(node)

                if 'import_statement' in node_type:
                    lib_name = None
                    alias_name = None
                    for child in node['children']:
                        if child['type'] == 'dotted_name':
                            lib_name = '.'.join([c['name'] for c in child['children'] if c['type'] == 'identifier'])
                        elif child['type'] == 'aliased_import':
                            for sub_child in child['children']:
                                if sub_child['type'] == 'dotted_name':
                                    lib_name = '.'.join([c['name'] for c in sub_child['children'] if c['type'] == 'identifier'])
                                elif sub_child['type'] == 'identifier':
                                    alias_name = sub_child['name']
                    if lib_name:
                        if alias_name:
                            import_nodes.append({'type': 'import', 'library': lib_name, 'alias': alias_name})
                        else:
                            import_nodes.append({'type': 'import', 'library': lib_name})

                # elif 'import_from_statement' in node_type:
                #     lib_name = None
                #     modules = []
                #     alias_name = None
                #     for child in node['children']:
                #         if child['type'] == 'dotted_name':
                #             lib_name = [c['name'] for c in child['children'] if c['type'] == 'identifier']
                #         if child['type'] == 'wildcard_import':
                #             modules.append('all')
                #         elif child['type'] == 'dotted_name' and 'children' in child:
                #             modules.extend([c['name'] for c in child['children'] if c['type'] == 'identifier'])
                #         elif child['type'] == 'identifier' and 'name' in child:
                #             alias_name = child['name']
                #     if lib_name:
                #         import_nodes.append({'type': 'import_from', 'library': lib_name, 'modules': modules})
                elif 'import_from_statement' in node_type:
                    lib_name = None
                    modules = []
                    alias_name = None
                    for child in node['children']:
                        if child['type'] == 'from':
                            continue
                        if child['type'] == 'dotted_name' and not lib_name:
                            lib_name = [c['name'] for c in child['children'] if c['type'] == 'identifier'][0]
                        elif child['type'] == 'wildcard_import':
                            modules.append('all')
                        elif child['type'] == 'dotted_name' and lib_name:
                            modules.extend([c['name'] for c in child['children'] if c['type'] == 'identifier'])
                        elif child['type'] == 'identifier' and 'name' in child:
                            alias_name = child['name']
                    if lib_name:
                        import_nodes.append({'type': 'import_from', 'library': lib_name, 'modules': modules})


                elif "expression_statement" in node_type:
                    # check for constants
                    inner_node = node['children'][0]

                    if 'type' in inner_node and 'assignment' in inner_node['type']:
                        for c in inner_node['children']:
                            if 'type' in c and "identifier" in c['type']:
                                # check if the name is all uppercase
                                if c['name'].isupper():
                                    constant_nodes.append({
                                                            'name': c['name'],
                                                            'value': self.process_expression(inner_node['children'][2])
                                    })

                                else:
                                    code_nodes.append(node)

                else:
                    code_nodes.append(node)

        return import_nodes, constant_nodes, code_nodes
        # return import_nodes, constant_nodes, code_nodesو error_nodes

    '''
    Recursively, extract the name of the imported library or module.
    '''

    def get_node_name(self, node):

        if isinstance(node, dict):
            # Check if the current node has a 'name' key
            if 'name' in node:
                return node['name']

            # Recursively search in the children
            for _, value in node.items():
                if isinstance(value, (dict, list)):
                    result = self.get_node_name(value)
                    if result:
                        return result

        elif isinstance(node, list):

            # Iterate over each element in the list and recursively search
            for item in node:
                result = self.get_node_name(item)
                if result:
                    return result
        return None

    def summarize_imports(self, import_nodes):
        return [
            self.get_node_name(node)
            for node in import_nodes
        ]

    def summarize_constants(self, constant_nodes):
        return [
            self.get_node_name(node)
            for node in constant_nodes
        ]

    
    def process_class_node(self, node, indent=0, class_info=None
                        ):
        if node['type'] == 'class_definition':
            for child in node['children']:
                self.process_class_node(child, indent, class_info)
        elif node['type'] == 'class':
            pass  # Class keyword, no action needed
        elif node['type'] == 'identifier' and 'name' in node:
            class_info["class_name"] = node['name']
        elif node['type'] == 'function_definition':
            # class_info["class_methods"].append(process_function_node(node))
            class_info.setdefault("class_methods", []).append(
                self.process_function_node(node))
        elif node['type'] == 'block':
            for child in node['children']:
                self.process_class_node(child, indent, class_info)

        return class_info


    def process_function_node(self, node):
        method_info = {
            "method_name": "",
            "parameters": []
        }
        for child in node['children']:
            if child['type'] == 'def':
                pass  # Def keyword, no action needed

            elif child['type'] == 'identifier' and 'name' in child:
                method_info["method_name"] = child['name']

            elif child['type'] == 'parameters':
                for param in child['children']:
                    if param['type'] == 'identifier':
                        if param['name'] != 'self':
                            method_info["parameters"].append(param['name'])
                    elif param['type'] == 'default_parameter':
                        method_info["parameters"].append(
                            param['children'][0]['name'])

            elif child['type'] == 'block':
                for grandchild in child['children']:
                    if grandchild['type'] == 'expression_statement':
                        self.process_class_node(grandchild)
                    elif grandchild['type'] == 'assignment':
                        self.process_class_node(grandchild)
        return method_info


    def process_expression_statement(self, node):
        for child in node['children']:
            if child['type'] == 'assignment':
                return self.process_assignment(child)
            elif child['type'] == 'augmented_assignment':
                return self.process_augmented_assignment(child)


    def process_assignment(self, node):
        left_side = node['children'][0]
        right_side = node['children'][2]

        left_value = self.process_identifier(left_side)
        right_value = self.process_expression(right_side)

        result = {}
        if left_value:
            result["left_side"] = left_value

        if right_value:
            result["right_side"] = right_value
        result["operation"] = "="
        return result


    def process_augmented_assignment(self, node):
        left_side = node['children'][0]
        operation = node['children'][1]['type']
        right_side = node['children'][2]

        left_value = self.process_identifier(left_side)
        right_value = self.process_expression(right_side)

        return {
            "left_side": left_value,
            "operation": operation,
            "right_side": right_value
        }


    def process_expression(self, node):
        if node['type'] in ['identifier', 'integer', 'string']:
            return node['name']
        if node['type'] in ['true', 'false']:
            return node['type']
        if node['type'] == 'binary_operator':
            left_side = node['children'][0]
            operator = node['children'][1]['type']
            right_side = node['children'][2]

            left_value = self.process_expression(left_side)
            right_value = self.process_expression(right_side)

            return {
                "left_side": left_value,
                "operator": operator,
                "right_side": right_value
            }
        if node['type'] == 'boolean_operator':
            left_side = node['children'][0]
            operator = node['children'][1]['type']
            right_side = node['children'][2]
            left_value = self.process_expression(left_side)
            right_value = self.process_expression(right_side)

            return {
                "left_side": left_value,
                "operator": operator,
                "right_side": right_value
            }
        
        if node['type'] == 'call':
            function_name = node['children'][0]['name']
            arguments = [self.process_expression(
                arg) for arg in node['children'][1]['children'] if arg['type'] != '(' and arg['type'] != ')']
            arguments_str = ", ".join(arguments)

            return {
                "function_name": function_name,
                "arguments": arguments_str
            }

        if node['type'] == 'list':
            return [child['name'] for child in node['children']if 'name' in child]


    def process_identifier(self, node):
        return node['name']

    def process_boolean_operator(self, node):
        left_side = self.process_expression(node['children'][0])
        operator = node['children'][1]['type']
        right_side = self.process_expression(node['children'][2])

        return {
            "left_side": left_side,
            "operator": operator,
            "right_side": right_side
        }

    def process_loops(self, ast_node):
        loops_info = []

        def process_node(node):
            if node['type'] == 'for_statement':
                loop_info = process_for_loop(node)
                loops_info.append(loop_info)
            elif node['type'] == 'while_statement':
                loop_info = process_while_loop(node)
                loops_info.append(loop_info)
            elif 'children' in node:
                for child in node['children']:
                    process_node(child)

        def process_for_loop(node):
            loop_info = {
                "type": "loop",
                "keyword": "for",
                "iterator": "",
                "iterable": "",
                "body": []
            }
            for child in node['children']:
                if child['type'] == 'identifier' and 'name' in child:
                    loop_info["iterator"] = child['name']
                elif child['type'] == 'call':
                    loop_info["iterable"] = process_call_node(child)
                elif child['type'] == 'block':
                    loop_info["body"] = process_block(child)
                elif child['type'] == 'pattern_list':
                    loop_info["iterator"] = process_argument_list(child, pattern=True)
                
                elif child['type'] == 'argument_list':
                    loop_info["iterator"] = process_argument_list(child, pattern=True)
                
                
            return loop_info

        def process_while_loop(node):
            loop_info = {
                "type": "loop",
                "keyword": "while",
                "condition": "",
                "body": []
            }
            for child in node['children']:
                if child['type'] == 'comparison_operator':
                    loop_info["condition"] = process_comparison_operator(child)
                elif child['type'] == 'block':
                    loop_info["body"] = process_block(child)
            return loop_info

        def process_call_node(node):
            call_info = []
            for child in node['children']:
                if child['type'] == 'identifier' and 'name' in child:
                    call_info.append(child['name'])
                elif child['type'] == 'argument_list':
                    call_info.append(process_argument_list(child))
            return ' '.join(call_info)

        def process_argument_list(node, pattern=False):
            arguments = []
            for child in node['children']:
                if 'name' in child and child['name']:
                    arguments.append(child['name'])

            print("arguments")
            print(arguments)
            if pattern:
                return f"({', '.join(arguments)})"
            return f"{' '.join(arguments)}"

        def process_comparison_operator(node):
            comparison = []
            for child in node['children']:
                if 'name' in child:
                    comparison.append(child['name'])
            return ' '.join(comparison)

        def process_block(node):
            statements = []
            for child in node['children']:
                if child['type'] == 'expression_statement':
                    statements.append(process_loop_expression_statement(child))
                elif child['type'] == 'call':
                    statements.append(process_call_node(child))

                elif child['type'] == 'argument_list':
                    statements.append(process_argument_list(child))

            return statements

        def process_loop_expression_statement(node):
            if 'children' in node and node['children']:
                assignment_node = node['children'][0]
                if assignment_node['type'] == 'assignment':
                    left = assignment_node['children'][0]['name']
                    right = assignment_node['children'][2]['name'] if 'name' in assignment_node['children'][2] else "unknown"
                    return f"{left} = {right}"
                elif assignment_node['type'] == 'augmented_assignment':
                    left = assignment_node['children'][0]['name']
                    operator = assignment_node['children'][1]['type']
                    right = assignment_node['children'][2]['name'] if 'name' in assignment_node['children'][2] else "unknown"
                    return f"{left} {operator} {right}"
                elif assignment_node['type'] == 'call':
                    function_name = assignment_node['children'][0]['name']
                    arguments = process_argument_list(
                        assignment_node['children'][1])
                    return f"{function_name}{arguments}"
            return "unknown_statement"

        process_node(ast_node)
        return loops_info
        # return json.dumps(loops_info, indent=2)


    def format_nested_expression(self, exp):
        if isinstance(exp, dict):
            left_side = self.format_nested_expression(exp.get('left_side', ''))
            operator = exp.get('operator', '')
            right_side = self.format_nested_expression(exp.get('right_side', ''))

            return f"({left_side} {operator} {right_side})"
        return str(exp)
    
    # function to transform summary to be human readable
    def get_summary(self, ast_summary):
        summary_text_to_speech = "Summary of the code:\n\n"
        # Imports
        import_list = ast_summary[0]
        if import_list:
            summary_text_to_speech+= "Imported libraries: "
            for item in import_list:
                if item['type'] == 'import':
                    summary_text_to_speech+= f" {item['library']} "
                
            summary_text_to_speech+= "\n"

            for item in import_list:
                if item['type'] == 'import_from':
                    summary_text_to_speech+= f"Imported {', '.join(item['modules'])} modules from {item['library']} \n"
            summary_text_to_speech+= "\n"

        # Constants
        constant_list = ast_summary[1]
        if constant_list:
            summary_text_to_speech+= "The constants declared are: \n"
            for item in constant_list:
                summary_text_to_speech+= f"{item['name']} equals to {item['value']}\n"
            summary_text_to_speech+= "\n"

        # Code
        for item in ast_summary[2:]:
            if item['type'] == 'class_definition':
                summary_text_to_speech += f"Class Named: {item['class_name']}\n"

                for method in item['class_methods']:
                    summary_text_to_speech+= f"\tMethod Named: {method['method_name']} with parameters ({', '.join(method['parameters'])})\n\n"

            elif item['type'] == 'function_definition':
                summary_text_to_speech += f"Function Named: {item['method_name']} with parameters ({', '.join(item['parameters'])})\n\n"

            elif item['type'] == 'expression_statement':
                #  _ = _
                if isinstance(item['right_side'], dict):
                    summary_text_to_speech+=f"A nested expression:\n"
                    left_side = item['left_side']
                    operation = item['operation']
                    right_side = self.format_nested_expression(item['right_side'])

                    summary_text_to_speech+= f"\t{left_side} {operation} {right_side}\n\n"
                else:
                # if 'operation' in item:
                    match item['operation']:
                        case '=':
                            summary_text_to_speech+= f"Assignment Operation: {item['left_side']} = {item['right_side']}\n\n"
                        case '+':
                            summary_text_to_speech+= f"Addition Operation: {item['left_side']} + {item['right_side']}\n\n"
                        case '-':
                            summary_text_to_speech+= f"Subtraction Operation: {item['left_side']} - {item['right_side']}\n\n"
                        case '*':
                            summary_text_to_speech+= f"Multiplication Operation: {item['left_side']} * {item['right_side']}\n\n"
                        case '/':
                            summary_text_to_speech+= f"Division Operation: {item['left_side']} / {item['right_side']}\n\n"
                        case '%':
                            summary_text_to_speech+= f"Modulus Operation: {item['left_side']} % {item['right_side']}\n\n"
                        case '==':
                            summary_text_to_speech+= f"Equality Operation: {item['left_side']} == {item['right_side']}\n\n"
                        case '!=':
                            summary_text_to_speech+= f"Inequality Operation: {item['left_side']} != {item['right_side']}\n\n"
                        case '>':
                            summary_text_to_speech+= f"Greater Than Operation: {item['left_side']} > {item['right_side']}\n\n"
                        case '<':
                            summary_text_to_speech+= f"Less Than Operation: {item['left_side']} < {item['right_side']}\n\n"
                        case '>=':
                            summary_text_to_speech+= f"Greater Than or Equal Operation: {item['left_side']} >= {item['right_side']}\n\n"
                        case '<=':
                            summary_text_to_speech+= f"Less Than or Equal Operation: {item['left_side']} <= {item['right_side']}\n\n"
                        case '+=':
                            summary_text_to_speech+= f"Addition Operation: {item['left_side']} += {item['right_side']}\n\n"
                        case '-=':
                            summary_text_to_speech+= f"Subtraction Operation: {item['left_side']} -= {item['right_side']}\n\n"
                        case '*=':
                            summary_text_to_speech+= f"Multiplication Operation: {item['left_side']} *= {item['right_side']}\n\n"
                        case '/=':
                            summary_text_to_speech+= f"Division Operation: {item['left_side']} /= {item['right_side']}\n\n"
                        case '%=':
                            summary_text_to_speech+= f"Modulus Operation: {item['left_side']} %= {item['right_side']}\n\n"
                        case 'and':
                            summary_text_to_speech+= f"And Operation: {item['left_side']} and {item['right_side']} \n\n"
                        case 'or':
                            summary_text_to_speech+= f"Or Operation: {item['left_side']} or {item['right_side']} \n\n"
                        case 'not':
                            summary_text_to_speech+= f"Not Operation: not {item['right_side']} \n\n"
                        case 'in':
                            summary_text_to_speech+= f"In Operation: {item['left_side']} in {item['right_side']} \n\n"
                        case 'not in':
                            summary_text_to_speech+= f"Not In Operation: {item['left_side']} not in {item['right_side']} \n\n"
                        case 'is':
                            summary_text_to_speech+= f"Is Operation: {item['left_side']} is {item['right_side']} \n\n"
                        case 'is not':
                            summary_text_to_speech+= f"Is Not Operation: {item['left_side']} is not {item['right_side']} \n\n"
                        

            elif item['type'] == 'loop':
                summary_text_to_speech+= f"A {item['keyword']} loop:\n"
                if (item['keyword'] == 'for'):
                    if isinstance(item['iterable'], dict):
                        summary_text_to_speech+= f"\t{item['keyword']} {item['iterator']} in {item['iterable']['function_name']}({item['iterable']['arguments']}) {item['condition']}:\n\n"
                else:
                    # while loop
                    summary_text_to_speech+= f"\t{item['keyword']} {item['condition']}:\n\n"

                for statement in item['body']:
                    summary_text_to_speech+= f"\t{statement}\n\n"

        summary_text_to_speech+= "End of code\n"
        return summary_text_to_speech


# Load the AST from the JSON file
with open('./ast_2.json', 'r') as file:
    ast = json.load(file)

ast_processor = ASTProcessor(ast['ast'])
summary = ast_processor.process_ast()
with open('summary.json', 'w') as file:
    json.dump(summary, file, indent=4)


final = ast_processor.get_summary(summary)
with open('summary.txt', 'w') as file:
    file.write(final)
# print(summary)

