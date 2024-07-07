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


def extract_import_nodes(ast):

    import_nodes = []
    constant_nodes = []
    code_nodes = []

    for node in ast['children']:

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


def get_node_name(node):

    if isinstance(node, dict):

        # Check if the current node has a 'name' key

        if 'name' in node:

            return node['name']

        # Recursively search in the children

        for key, value in node.items():

            if isinstance(value, (dict, list)):

                result = get_node_name(value)

                if result:
                    return result

    elif isinstance(node, list):

        # Iterate over each element in the list and recursively search

        for item in node:

            result = get_node_name(item)

            if result:
                return result

    return None


'''
Extract imported libraries and modules names,
and summarize them in a list.
'''


def summarize_imports(import_nodes):
    return [
        get_node_name(node)
        for node in import_nodes
    ]


def summarize_constants(constant_nodes):

    return [
        get_node_name(node)
        for node in constant_nodes
    ]


def process_class_node(node, indent=0, class_info={
    "class_name": "",
    "class_parameters": [],
    "class_methods": []
}
):
    if node['type'] == 'class_definition':
        for child in node['children']:
            process_class_node(child, indent, class_info)
    elif node['type'] == 'class':
        pass  # Class keyword, no action needed
    elif node['type'] == 'identifier' and 'name' in node:
        class_info["class_name"] = node['name']
    elif node['type'] == 'function_definition':
        class_info["class_methods"].append(process_function_node(node))
    elif node['type'] == 'block':
        for child in node['children']:
            process_class_node(child, indent, class_info)

    return class_info


def process_function_node(node):
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
                    process_class_node(grandchild)
                elif grandchild['type'] == 'assignment':
                    process_class_node(grandchild)
    return method_info


def process_expression_statement(node):
    for child in node['children']:
        if child['type'] == 'assignment':
            return process_assignment(child)
        elif child['type'] == 'augmented_assignment':
            return process_augmented_assignment(child)


def process_assignment(node):
    left_side = node['children'][0]
    right_side = node['children'][2]

    left_value = process_identifier(left_side)
    right_value = process_expression(right_side)

    return {
        "left_side": left_value,
        "right_side": right_value
    }


def process_augmented_assignment(node):
    left_side = node['children'][0]
    operation = node['children'][1]['type']
    right_side = node['children'][2]

    left_value = process_identifier(left_side)
    right_value = process_expression(right_side)

    return {
        "left_side": left_value,
        "operation": operation,
        "right_side": right_value
    }


def process_expression(node):
    if node['type'] == 'identifier':
        return node['name']
    elif node['type'] == 'integer':
        return "integer value"
    elif node['type'] == 'string':
        return node['name']
    elif node['type'] == 'binary_operator':
        left_side = node['children'][0]
        operator = node['children'][1]['type']
        right_side = node['children'][2]

        left_value = process_expression(left_side)
        right_value = process_expression(right_side)

        return {
            "left_side": left_value,
            "operator": operator,
            "right_side": right_value
        }
    elif node['type'] == 'call':
        function_name = node['children'][0]['name']
        arguments = [process_expression(
            arg) for arg in node['children'][1]['children'] if arg['type'] != '(' and arg['type'] != ')']
        arguments_str = ", ".join(arguments)

        return {
            "function_name": function_name,
            "arguments": arguments_str

        }


def process_identifier(node):
    return node['name']


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

import_nodes, constant_nodes, code_nodes = extract_import_nodes(ast['ast'])


# write to file

with open('import_nodes.json', 'w') as file:

    json.dump(import_nodes, file, indent=4)


with open('constant_nodes.json', 'w') as file:

    json.dump(constant_nodes, file, indent=4)


with open('code_nodes.json', 'w') as file:

    json.dump(code_nodes, file, indent=4)


summary = []
for node in code_nodes:
    if node['type'] == "class_definition":
        class_info = process_class_node(node)

        summary.append(dict(class_info))  # for deepcopy
    elif node['type'] == "function_definition":
        method_info = process_function_node(node)
        # print(method_info)
        summary.append(dict(method_info))

    elif node['type'] == "expression_statement":
        assignment_info = process_expression_statement(node)
        summary.append(dict(assignment_info))


print(summary)
