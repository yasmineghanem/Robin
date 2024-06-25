import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import numpy as np
import random
import json
from string import Formatter
import itertools


def clean(line):
    '''
    This function cleans the line by removing all the special characters and punctuations

    Args:
        - line (str) : a sentence from the dataset that needs to be cleaned

    Returns:
        - cleaned_line (str) : cleaned sentence
    '''
    cleaned_line = ''
    for char in line:
        if char.isalpha():
            cleaned_line += char
        else:
            cleaned_line += ' '
    cleaned_line = ' '.join(cleaned_line.split())
    return cleaned_line


def load_data(file_path):
    '''
    This function loads the data from the file_path

    Given the path of the dataset this function reads and returns the intents and the corpus of the dataset

    Args:
        - file_path (str) : path of the dataset

    Returns:
        - unique_intents (list[str]) : list of unique intents in the dataset
        - corpus (list[str]) : list of all the sentences in the dataset
        - corpus_intents (list[str]) : list of intents for each sentence in the dataset
        - responses (list[str]) : list of responses for each intent in the dataset
    '''
    unique_intents = []
    corpus = []
    corpus_intents = []
    responses = []

    with open(file_path, 'r') as f:
        dataset = json.load(f)

        intents = dataset['intents']

        for intent in intents:
            if intent['intent'] not in unique_intents:
                unique_intents.append(intent['intent'])
            for keyword in intent['keywords']:
                corpus.append(clean(keyword))
                corpus_intents.append(intent['intent'])

            responses.append(intent['responses'][0])

    return unique_intents, corpus, corpus_intents, responses


def reformat_json(file_path):
    '''
    This function reformats the json file to make it more readable

    Args:
        - file_path (str) : path of the dataset

    Returns:
        - None
    '''
    with open(file_path, 'r') as f:
        data = json.load(f)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def prepare_ner_dataset(file_path):
    '''
    This function prepares the NER dataset from the intents

    It reads the text for each intent and write it to a file to be passed to the annotator

    Args:
        - file_path (str) : path of the dataset
    '''
    with open(file_path, 'r') as f:
        data = json.load(f)

        intents = data["intents"]

        for intent in intents:
            # create a list to store the sentences for each intent
            corpus = []

            for keyword in intent["keywords"]:
                corpus.append(keyword)

            # add a period at the end of each sentence
            corpus = [sentence +
                      ".\n" for sentence in corpus if sentence[-1] != "."]

            # join the sentences with a space
            corpus = " ".join(corpus)

            # rename the file to the intent name
            file_name = intent["intent"].lower().replace(" ", "_") + ".txt"

            # write the corpus to the file
            with open(f'./ner_dataset/{file_name}', 'w') as f:
                f.write(corpus)


def augment_data(file_path):
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
            current_intent = intent["intent"]
            print(current_intent)
            default_parameters = intent["default parameters"]
        
            templates = intent["patterns"]

            formatted_templates = []

            print(len(templates))
            
            for template in templates:

                if len(templates) < 15:
                    random_number = random.randint(10, 15)
                else:
                    random_number = random.randint(5, 10)

                for _ in range(random_number):

                    filtered_parameters = {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name}
                    
                    final_parameters = {key: default_parameters[key] for key in filtered_parameters}

                    if intent["intent"] == "Variable Declaration":
                        if "datatype" in final_parameters:

                            # we check if the type is in the template then we check the type to choose the appropriate synonyms
                            final_parameters["datatype"] = random.choice(intent["synonyms"]["datatype"])

                            match final_parameters["datatype"]:
                                case "int" | "integer":
                                    final_parameters["value"] = random.choice(intent["synonyms"]["value"]["integer"])
                                case "float" | "double":
                                    final_parameters["value"] = random.choice(intent["synonyms"]["value"]["float"])
                                case "string":
                                    final_parameters["value"] = random.choice(intent["synonyms"]["value"]["string"])
                                case "char" | "character":
                                    final_parameters["value"] = chr(random.randint(32, 123))
                                case "bool" | "boolean":
                                    final_parameters["value"] = random.choice(intent["synonyms"]["value"]["boolean"])
                                case "array" | "list":
                                    final_parameters["value"] = random.choice(intent["synonyms"]["value"]["list"])
                                case "dictionary":
                                    final_parameters["value"] = random.choice(intent["synonyms"]["value"]["dictionary"])
                        else:
                            final_parameters["value"] = random.choice(list(itertools.chain(*intent["synonyms"]["value"].values())))
                        
                    for parameter in final_parameters:

                        if intent["intent"] == "Variable Declaration":
                            if parameter == "value" or parameter == "datatype":
                                continue
                        if parameter == "value":
                            if type(intent["synonyms"]["value"]) == dict:
                                final_parameters["value"] = random.choice(list(itertools.chain(*intent["synonyms"]["value"].values())))
                                continue
                        
                        if parameter == "variable_1" or parameter == "variable_2":
                            final_parameters[parameter] = random.choice(intent["synonyms"]["variable"])
                            continue

                        synonyms = intent["synonyms"].get(parameter, [])
                        final_parameters[parameter] = random.choice(synonyms)

                    formatted_string = template.format(**final_parameters)

                    formatted_templates.append(formatted_string)

            intent["formatted patterns"] = formatted_templates

            print(len(formatted_templates))

            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)

def ner_dataset_pre_annotations(file_path):
    
    with open(file_path, 'r') as f:
        data = json.load(f)

        intents = data["intents"]

        for intent in intents:

            corpus = []

            for example in intent["formatted patterns"]:
                corpus.append(example)


            # add a period at the end of each sentence
            corpus = [sentence + ".\n" for sentence in corpus if sentence[-1] != "."]

            # join the sentences with a space
            corpus = " ".join(corpus)

            file_name = intent["intent"].lower().replace(" ", "_") + ".txt"
            with open(f'./ner_dataset/{file_name}', 'w') as f:
                f.write(corpus)
