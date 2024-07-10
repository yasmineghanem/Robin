import json
import random
import itertools
from string import Formatter

class DataAugmenter:
    def __init__(self, patterns):
        self.patterns = patterns

    
    def augment_intent(file_path, target_intent, number_of_augmentations=10):
        '''
        This function augments the data in the dataset

        It take the intent keywords and augments them using the synonym augmenter and inserts spelling mistakes in the keywords and saves the augmented data to a new file

        Args:
            - file_path (str) : path of the dataset
        '''

        with open(file_path, 'r') as intents_file:
            data = json.load(intents_file)

            intents = data["intents"]

            for intent in intents:    
                if intent["intent"] == target_intent:
                    print(intent["intent"])
                        
                    # intent_data = intent[target_intent]

                    default_parameters = intent["default parameters"]

                    templates = intent["patterns"]

                    formatted_templates = []

                    print(len(templates))

                    for template in templates:

                        number = number_of_augmentations

                        for _ in range(number):

                            filtered_parameters = {
                                field_name for _, field_name, _, _ in Formatter().parse(template) if field_name}

                            final_parameters = {
                                key: default_parameters[key] for key in filtered_parameters}

                            if intent["intent"] == "Variable Declaration":
                                if "datatype" in final_parameters:

                                    # we check if the type is in the template then we check the type to choose the appropriate synonyms
                                    final_parameters["datatype"] = random.choice(
                                        intent["synonyms"]["datatype"])

                                    match final_parameters["datatype"]:
                                        case "int" | "integer":
                                            final_parameters["value"] = random.choice(
                                                intent["synonyms"]["value"]["integer"])
                                        case "float" | "double":
                                            final_parameters["value"] = random.choice(
                                                intent["synonyms"]["value"]["float"])
                                        case "string":
                                            final_parameters["value"] = random.choice(
                                                intent["synonyms"]["value"]["string"])
                                        case "char" | "character":
                                            final_parameters["value"] = chr(
                                                random.randint(32, 123))
                                        case "bool" | "boolean":
                                            final_parameters["value"] = random.choice(
                                                intent["synonyms"]["value"]["boolean"])
                                else:
                                    final_parameters["value"] = random.choice(
                                        list(itertools.chain(*intent["synonyms"]["value"].values())))

                            for parameter in final_parameters:

                                if intent["intent"] == "Variable Declaration":
                                    if parameter == "value" or parameter == "datatype":
                                        continue
                                if parameter == "value":
                                    if type(intent["synonyms"]["value"]) == dict:
                                        final_parameters["value"] = random.choice(
                                            list(itertools.chain(*intent["synonyms"]["value"].values())))
                                        continue
                                
                                if parameter == "variable_1" or parameter == "variable_2" or parameter == "variable_3":
                                    final_parameters[parameter] = random.choice(
                                        intent["synonyms"]["variable"])
                                    continue
                                
                                if parameter == "start" or parameter == "end" or parameter == "step":
                                    final_parameters[parameter] = random.choice(intent["synonyms"]["number"])
                                    continue

                                synonyms = intent["synonyms"].get(parameter, [])
                                final_parameters[parameter] = random.choice(synonyms)

                            formatted_string = template.format(**final_parameters)

                            formatted_templates.append(formatted_string)

                    intent["formatted patterns"] = formatted_templates

                    print(len(formatted_templates))

                    with open(file_path, 'w') as file:
                        json.dump(data, file, indent=4)


    def augment_data(self, file_path, intents=['all'], number_of_augmentations=10):
        if intents == ['all']:
            with open(file_path, 'r') as intents_file:
                data = json.load(intents_file)

                intents = data["intents"]
                for intent in intents:
                    self.augment_intent(file_path, intent, number_of_augmentations)
        else:
            for intent in intents:
                self.augment_intent(file_path, intent, number_of_augmentations)