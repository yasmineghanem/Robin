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


def narrate_ast(ast, indent=0):

    indentation = '    ' * indent

    if isinstance(ast, dict):

        if 'type' in ast:

            if 'name' in ast:

                return f'{indentation}{ast["type"]} named "{ast["name"]}".'

            elif 'value' in ast:

                return f'{indentation}{ast["type"]} with value "{ast["value"]}".'

            else:

                narration = f'{indentation}{ast["type"]} containing:\n'

                for key, value in ast.items():

                    if key != 'type':

                        narration += narrate_ast(value, indent + 1) + '\n'

                return narration.strip()

        else:

            return '\n'.join(narrate_ast(value, indent) for key, value in ast.items())

    elif isinstance(ast, list):

        return '\n'.join(narrate_ast(item, indent) for item in ast)

    else:

        return f'{indentation}{str(ast)}'


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


# Load the AST from the JSON file
with open('./ast.json', 'r') as file:

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


# Summarize imports

imports = summarize_imports(import_nodes)
print(imports)

constants = summarize_constants(constant_nodes)
print(constants)

