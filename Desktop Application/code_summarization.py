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

            elif node['type'] == "for_statement" or node['type'] == "while_statement":
                loop = self.process_loops(node)
                for l in loop:
                    summary.append(dict(l))
        return summary

    def extract_import_nodes(self):
        import_nodes = []
        constant_nodes = []
        code_nodes = []

        for node in self.ast['children']:
            if 'type' in node:
                node_type = node['type']
                if 'import' in node_type:
                    import_nodes.append(node)
                elif "expression_statement" in node_type:
                    inner_node = node['children'][0]
                    if 'type' in inner_node and 'assignment' in inner_node['type']:
                        for c in inner_node['children']:
                            if 'type' in c and "identifier" in c['type']:
                                if c['name'].isupper():
                                    constant_nodes.append(node)
                                else:
                                    code_nodes.append(node)
                else:
                    code_nodes.append(node)
        return import_nodes, constant_nodes, code_nodes

    def process_class_node(self, node, indent=0, class_info=None):
        if class_info is None:
            class_info = {
                "class_name": "",
                "class_parameters": [],
                "class_methods": []
            }

        if node['type'] == 'class_definition':
            for child in node['children']:
                self.process_class_node(child, indent, class_info)
        elif node['type'] == 'class':
            pass  # Class keyword, no action needed
        elif node['type'] == 'identifier' and 'name' in node:
            class_info["class_name"] = node['name']
        elif node['type'] == 'function_definition':
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
            if child['type'] == 'identifier' and 'name' in child:
                method_info["method_name"] = child['name']
            elif child['type'] == 'parameters':
                method_info["parameters"] = self.process_parameters(child)
        return method_info

    def process_parameters(self, node):
        params = []
        for param in node['children']:
            if param['type'] == 'identifier' and param['name'] != 'self':
                params.append(param['name'])
            elif param['type'] == 'default_parameter':
                params.append(param['children'][0]['name'])
        return params

    def process_expression_statement(self, node):
        for child in node['children']:
            if child['type'] == 'assignment':
                return self.process_assignment(child)
            elif child['type'] == 'augmented_assignment':
                return self.process_augmented_assignment(child)

    def process_assignment(self, node):
        left_value = self.process_identifier(node['children'][0])
        right_value = self.process_expression(node['children'][2])

        result = {}
        if left_value:
            result["left_side"] = left_value
        if right_value:
            result["right_side"] = right_value

        return result

    def process_augmented_assignment(self, node):
        left_value = self.process_identifier(node['children'][0])
        operation = node['children'][1]['type']
        right_value = self.process_expression(node['children'][2])

        return {
            "left_side": left_value,
            "operation": operation,
            "right_side": right_value
        }

    def process_expression(self, node):
        if node['type'] in ['identifier', 'integer', 'string']:
            return node['name']
        if node['type'] == 'binary_operator':
            return self.process_binary_operator(node)
        if node['type'] == 'call':
            return self.process_call_node(node)
        if node['type'] == 'list':
            return [child['name'] for child in node['children'] if 'name' in child]

    def process_binary_operator(self, node):
        left_value = self.process_expression(node['children'][0])
        operator = node['children'][1]['type']
        right_value = self.process_expression(node['children'][2])

        return {
            "left_side": left_value,
            "operator": operator,
            "right_side": right_value
        }

    def process_identifier(self, node):
        return node['name']

    def process_call_node(self, node):
        function_name = node['children'][0]['name']
        arguments = [self.process_expression(
            arg) for arg in node['children'][1]['children'] if arg['type'] != '(' and arg['type'] != ')']
        arguments_str = ", ".join(arguments)

        return {
            "function_name": function_name,
            "arguments": arguments_str
        }

    def process_loops(self, loop_node):
        loops_info = []

        def process_node(node):
            if node['type'] in ['for_statement', 'while_statement']:
                loop_info = self.process_loop(node)
                loops_info.append(loop_info)
            elif 'children' in node:
                for child in node['children']:
                    process_node(child)

        process_node(loop_node)
        return loops_info

    def process_loop(self, node):
        loop_info = {
            "type": "loop",
            "keyword": node['type'].split('_')[0],
            "condition": "",
            "body": []
        }

        if node['type'] == 'for_statement':
            loop_info["iterator"] = ""
            loop_info["iterable"] = ""

        for child in node['children']:
            if child['type'] == 'identifier' and 'name' in child:
                if node['type'] == 'for_statement':
                    loop_info["iterator"] = child['name']
                else:
                    loop_info["condition"] = self.process_expression(child)
            elif child['type'] == 'call':
                loop_info["iterable"] = self.process_call_node(child)
            elif child['type'] == 'comparison_operator':
                loop_info["condition"] = self.process_comparison_operator(
                    child)
            elif child['type'] == 'block':
                loop_info["body"] = self.process_block(child)

        return loop_info

    def process_comparison_operator(self, node):
        return ' '.join(child['name'] for child in node['children'] if 'name' in child)

    def process_block(self, node):
        statements = []
        for child in node['children']:
            if child['type'] == 'expression_statement':
                statements.append(self.process_expression_statement(child))
        return statements

    # function to transform summary to be human readable 
    def write_summary_to_file(self,summary,file_name='summary.txt'):
        with open(file_name, 'w') as file:
            file.write("Summary of the code:\n\n")
            for item in summary:
                if item['type'] == 'import':
                    file.write(f"Imported: {item['library']}\n\n")
                elif item['type'] == 'class_definition':
                    file.write(f"Class Named: {item['class_name']}\n")
                    for method in item['class_methods']:
                        file.write(f"\tMethod Named: {method['method_name']} with parameters ({', '.join(method['parameters'])})\n\n")

                elif item['type'] == 'function_definition':
                    file.write(f"Function Named: {item['method_name']} with parameters ({', '.join(item['parameters'])})\n\n")

                elif item['type'] == 'expression_statement':
                    file.write("An expression statement: \n")
                    if 'operation' in item:
                        file.write(f"\t{item['left_side']} {item['operation']} {item['right_side']}\n\n")
                    else:
                        file.write(f"\t{item['left_side']} = {item['right_side']}\n\n")

                elif item['type'] == 'loop':
                    file.write(f"A {item['keyword']} loop:\n")
                    if(item['keyword'] == 'for'):
                        #broblem here
                        print(item['iterable'])
                        if isinstance(item['iterable'], dict):
                            file.write(f"\t{item['keyword']} {item['iterator']} in {item['iterable']['function_name']}({item['iterable']['arguments']}) {item['condition']}:\n\n")
                        # print(f"item['iterable'] is of type {type(item['iterable'])} and has value {item['iterable']}")
                    else:
                        # while loop
                        file.write(f"\t{item['keyword']} {item['condition']}:\n\n")

                    for statement in item['body']:
                        file.write(f"\t{statement}\n\n")
            
            file.write("End of code\n")

# Load the AST from the JSON file
with open('./ast_3.json', 'r') as file:

    ast = json.load(file)


# print(ast['ast'])


# Generate narration

# narration = narrate_ast(ast)

# print(narration)


# # Output the narration to file

# with open('narration.txt', 'w') as file:

#     file.write(narration)


# Extract nodes

# import_nodes, constant_nodes, code_nodes = extract_import_nodes(ast['ast'])


# write to file

# with open('import_nodes.json', 'w') as file:

#     json.dump(import_nodes, file, indent=4)


# with open('constant_nodes.json', 'w') as file:

#     json.dump(constant_nodes, file, indent=4)


# with open('code_nodes.json', 'w') as file:

#     json.dump(code_nodes, file, indent=4)


# summary = []
# for node in code_nodes:
#     if node['type'] == "class_definition":
#         info = process_class_node(node, class_info={})
#         info['type'] = 'class_definition'
#         summary.append(dict(info))  # for deepcopy

#     elif node['type'] == "function_definition":
#         method_info = process_function_node(node)
#         method_info['type'] = 'function_definition'
#         summary.append(dict(method_info))

#     elif node['type'] == "expression_statement":
#         assignment_info = process_expression_statement(node)
#         assignment_info['type'] = 'expression_statement'
#         summary.append(dict(assignment_info))

#     elif node['type'] == "for_statement" or node['type'] == "while_statement":
#         loop = process_loops(node)
#         # print(type(loop_info))
#         for l in loop:
#             summary.append(dict(l))

#     # elif node['type'] == "while_statement":
#     #     loop_info = process_loops(node)
#     #     summary.append(dict(loop_info))


# print(summary)


ast_processor = ASTProcessor(ast['ast'])
summary = ast_processor.process_ast()
ast_processor.write_summary_to_file(summary)
# write into file
with open('summary.json', 'w') as file:
    json.dump(summary, file, indent=4)
# print(summary)


