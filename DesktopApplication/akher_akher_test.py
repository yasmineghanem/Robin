import importlib.util
import sys
import os
import numpy as np
import tensorflow as tf
import torch

np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(1)

# Add the DesktopApplication directory to the Python path
desktop_application_path = os.path.abspath('.')
sys.path.append(desktop_application_path)

# Add the CommandIntent/final_files directory to the Python path
command_intent_path = os.path.abspath('../CommandIntent/final_files')
sys.path.append(command_intent_path)

# Path to the command_intent module
module_name = "command_intent"
module_file_path = os.path.join(command_intent_path, "command_intent.py")
spec = importlib.util.spec_from_file_location(module_name, module_file_path)
command_intent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(command_intent)

# Import the CommandIntent class
CommandIntent = command_intent.CommandIntent

# Correctly resolve paths relative to the script's location
intent_model_path = os.path.abspath(
    '../CommandIntent/models/intent_detection_model_2.h5')
ner_model_path = os.path.abspath('../CommandIntent/models/constrained_ner_model_2.pth')

# Create an instance of CommandIntent with correct paths
command_test = CommandIntent(intent_model_path, ner_model_path)

# Now you can use the CommandIntent instance
print(command_test)

test_sentences = [
    'declare a new integer variable x and assign it the value 5', # 0 # tamam
    'declare a new string constant pi and assign it the value 3.14', # 1 #TODO: fix this
    'define a new function called add that takes two parameters x and y', # 2 # much better
    'create a new class called person', # 3 # tamam
    'create a for loop from 1 to 10 using the variable i', # 4 # mashy
    'for item in items', # 5 #TODO handel collection and end
    'create a while loop that runs until x is greater than 5',  # 6 # better # mashy
    'check if number is equal to 5 and not flag', # 7  #TODO: fix this
    'current is equal to previous', # 8 # tamam
    'bitwise and x and y', # 9 #TODO: handel operands
    'cast x to integer', # 10 # tamam
    'ask the user to input their name', # 11 # mashy 
    'print the variable x', # 12 #tamam
    'assert that x is equal to y', # 13 # TODO: handel in post processing
    'import the math library', # 14 # tamam
    'add a new comment this is a test comment', # 15 $ tamam
    'create new file test.txt', # 16 # 3alaya ana dy  # TODO: fix this + send type
    'push the current changes with the message changed something', # 17 # tamam
    'list all the files in the program', # 18 # correct
    'highlight lines 9 to 15', # 19 # wrong intent
]

test_sentences_2 = [
    "declare a variable number and set it to 5",
    "create a new constant epochs and put 100 in it", # medaye2 menha leh msh 3arfa
    "write a function named binary search that takes values and target",
    "define a class called vehicle",
    "create a for loop that runs 10 times",
    "create a for loop for node in nodes",
    "create a while loop with condition flag is true",
    "whether type is not false",
    "max value equals temp",
    "bitwise value or 10",
    "cast x to float",
    "ask the user for input with message please enter your age and assign it to user age",
    "output the message hello world",
    "verify that length is greater than 1000",
    "import the library numpy",
    "comment we love you",
    "create a file test", # TODO: doesnt detect files # separate
    "stash the changes",
    "list the functions in the program",
    "go to line 100" #TODO: separate
]

while True:
    i = input()
    try:
        print(command_test.process_command(test_sentences_2[int(i)]))

    except Exception as e:
        print(e)

print("ESHTAGHALLLLL")
