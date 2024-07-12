import importlib.util
import sys
import os

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
intent_model_path = os.path.abspath('../CommandIntent/models/intent_detection_model.h5')
ner_model_path = os.path.abspath('../CommandIntent/models/full_ner_model.pth')

# Create an instance of CommandIntent with correct paths
command_test = CommandIntent(intent_model_path, ner_model_path)

# Now you can use the CommandIntent instance
print(command_test)

test_sentences = [
    'declare a new integer variable x and assign it the value 5',
    'declare a new string constant pi and assign it the value 3.14',
    'define a new function called add that takes two parameters x and y', # feha haga ghalat lesa (DONT TRY)
    'create a new class called person',
    'create a for loop from 1 to 10 using the variable i',
    'create a while loop that runs until x is greater than 5',
    'check if number is equal to 5',
    'current is equal to previous',
    'perform bitwise and operation on x and y',
    'cast the variable x to integer',
    'ask the user to input their name',
    'print the variable x',
    'assert that x is equal to y',
    'import the math module',
    'add a new comment this is a test comment',
    'create new file test.txt', # 3alaya ana dy
    'push the current changes with the message changed something',
    'list all the files in the program',
    'select line 10',
]

print(command_test.process_command(test_sentences[2]))

print("ESHTAGHALLLLL")
