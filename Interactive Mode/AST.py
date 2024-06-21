#IMPORTS
import ast
import astor 
import codegen 
import json
import ast
from graphviz import Digraph


##################### Define a function to recursively add nodes to the Digraph #####################
dot = Digraph()
def add_node(node, parent=None):
    node_name = str(node.__class__.__name__)
    dot.node(str(id(node)), node_name)
    if parent:
        dot.edge(str(id(parent)), str(id(node)))
    for child in ast.iter_child_nodes(node):
        add_node(child, node)
        
# Render the Digraph as a PNG file
dot.format = 'png'
dot.render('my_ast', view=True)