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

            # elif node['type'] == "for_statement":
            #     loop = self.process_for_loop(node)
            #     summary.append(dict(loop))

            # elif node['type'] == "while_statement":
            #     loop = self.process_while_loop(node)
            #     # print(type(loop_info))
            #     summary.append(dict(loop))
            elif node['type'] == "for_statement" or node['type'] == "while_statement":
                loop = self.process_loops(node)
        # print(type(loop_info))
                for l in loop:
                    summary.append(dict(l))

        return summary

    def extract_import_nodes(self, ):

        import_nodes = []
        constant_nodes = []
        code_nodes = []

        for node in self.ast['children']:

            # if the type contains the word import

            if 'type' in node:
                node_type = node['type']

                if 'import' in node_type:
                    import_nodes.append(node)

                elif "expression_statement" in node_type:
                    # check for constants
                    inner_node = node['children'][0]

                    if 'type' in inner_node and 'assignment' in inner_node['type']:
                        for c in inner_node['children']:
                            if 'type' in c and "identifier" in c['type']:
                                # check if the name is all uppercase
                                if c['name'].isupper():
                                    constant_nodes.append(node)

                                else:
                                    code_nodes.append(node)

                else:
                    code_nodes.append(node)

        return import_nodes, constant_nodes, code_nodes

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
                    loop_info["iterator"] = process_argument_list(child)
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

        def process_argument_list(node):
            arguments = []
            for child in node['children']:
                if 'name' in child and child['name']:
                    arguments.append(child['name'])

            print("arguments")
            print(arguments)
            return f"{', '.join(arguments)}"

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



    # function to transform summary to be human readable

    def write_summary_to_file(self, summary, file_name='summary.txt'):
        with open(file_name, 'w') as file:
            file.write("Summary of the code:\n\n")
            for item in summary:
                if item['type'] == 'import':
                    file.write(f"Imported: {item['library']}\n\n")
                elif item['type'] == 'class_definition':
                    file.write(f"Class Named: {item['class_name']}\n")
                    for method in item['class_methods']:
                        file.write(
                            f"\tMethod Named: {method['method_name']} with parameters ({', '.join(method['parameters'])})\n\n")

                elif item['type'] == 'function_definition':
                    file.write(
                        f"Function Named: {item['method_name']} with parameters ({', '.join(item['parameters'])})\n\n")

                elif item['type'] == 'expression_statement':
                    file.write("An expression statement: \n")
                    if 'operation' in item:
                        file.write(
                            f"\t{item['left_side']} {item['operation']} {item['right_side']}\n\n")
                    else:
                        file.write(
                            f"\t{item['left_side']} = {item['right_side']}\n\n")

                elif item['type'] == 'loop':
                    file.write(f"A {item['keyword']} loop:\n")
                    if (item['keyword'] == 'for'):
                        # broblem here
                        print(item['iterable'])
                        if isinstance(item['iterable'], dict):
                            file.write(
                                f"\t{item['keyword']} {item['iterator']} in {item['iterable']['function_name']}({item['iterable']['arguments']}) {item['condition']}:\n\n")
                        # print(f"item['iterable'] is of type {type(item['iterable'])} and has value {item['iterable']}")
                    else:
                        # while loop
                        file.write(
                            f"\t{item['keyword']} {item['condition']}:\n\n")

                    for statement in item['body']:
                        file.write(f"\t{statement}\n\n")

            file.write("End of code\n")


# Load the AST from the JSON file
with open('./ast_3.json', 'r') as file:

    ast = json.load(file)

ast_processor = ASTProcessor(ast['ast'])
summary = ast_processor.process_ast()
ast_processor.write_summary_to_file(summary)
# write into file
with open('summary.json', 'w') as file:
    json.dump(summary, file, indent=4)
# print(summary)
