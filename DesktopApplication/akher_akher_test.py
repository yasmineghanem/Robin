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
intent_model_path = os.path.abspath('../CommandIntent/models/intent_detection_model.keras')
ner_model_path = os.path.abspath('../CommandIntent/models/ner_model_2.pth')

# Create an instance of CommandIntent with correct paths
command_test = CommandIntent(intent_model_path, ner_model_path)

# Now you can use the CommandIntent instance
print(command_test)

print(command_test.process_command("cast the variable x to integer"))

print("ESHTAGHALLLLL")
