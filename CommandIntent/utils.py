import re
import numpy as np
import random
import json
from string import Formatter
import itertools
import os
import pandas as pd
import torch


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
    # responses = []

    with open(file_path, 'r') as f:
        dataset = json.load(f)

        intents = dataset['intents']

        for intent in intents:
            if intent['intent'] not in unique_intents:
                unique_intents.append(intent['intent'])
            for keyword in intent['formatted patterns']:
                corpus.append(clean(keyword))
                corpus_intents.append(intent['intent'])

    return unique_intents, corpus, corpus_intents


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


def augment_data(file_path, target_intent):
    '''
    This function augments the data in the dataset

    It take the intent keywords and augments them using the synonym augmenter and inserts spelling mistakes in the keywords and saves the augmented data to a new file

    Args:
        - file_path (str) : path of the dataset
    '''

    with open(file_path, 'r') as intents_file:
        data = json.load(intents_file)

        intents = data["intents"]

        intent = intents[target_intent]

        # for intent in intents:
        # current_intent = intent["intent"]
        current_intent = target_intent
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
                            case "array" | "list":
                                final_parameters["value"] = random.choice(
                                    intent["synonyms"]["value"]["list"])
                            case "dictionary":
                                final_parameters["value"] = random.choice(
                                    intent["synonyms"]["value"]["dictionary"])
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

                    if parameter == "variable_1" or parameter == "variable_2":
                        final_parameters[parameter] = random.choice(
                            intent["synonyms"]["variable"])
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
            corpus = [sentence +
                      ".\n" for sentence in corpus if sentence[-1] != "."]

            # join the sentences with a space
            corpus = " ".join(corpus)

            file_name = intent["intent"].lower().replace(" ", "_") + ".txt"
            with open(f'./ner_dataset/{file_name}', 'w') as f:
                f.write(corpus)


def remove_punctuation(input_string):
    # Define a regex pattern to match punctuation (excluding underscore and digits)
    pattern = r'(?<!\d)\.(?![\d\s])|[^\w\s.\-]|_'

    # Use re.sub to substitute all matches of the pattern with an empty string
    return re.sub(pattern, '', input_string)

def find_subset_indices(subset, larger_list):
    indices = []
    for item in subset:
        try:
            # Find the index of the current item in the larger list
            index = larger_list.index(item)
            indices.append(index)
        except ValueError:
            # If an item is not found, return None indicating subset is not fully present
            return None
    return indices

def convert_annotations_to_csv(file_path):

    # delete the csv file if it exists
    if os.path.exists("./ner_dataset/ner_dataset.csv"):
        os.remove("./ner_dataset/ner_dataset.csv")

    with open(file_path, 'r') as file:
        data = json.load(file)

        annotations = data['annotations']

        intents = annotations.keys()
        print(intents)

        with open("./ner_dataset/ner_dataset.csv", 'w') as csv_file:
            csv_file.write("Sentence #, Word, Tag, Intent\n")

            sentence_index = 0

            for intent in intents: 
                print(intent)
                examples = annotations[intent]
                if len(examples) == 0:
                    continue
                
                for example in examples:
                    sentence = example[0]
                    entities = example[1]["entities"]
                    # print(sentence_index)
                    # print(sentence)
                    # print(entities)

                    sentence = sentence[:-1]

                    if sentence[-1] == ".":
                        sentence = sentence[:-1]

                    # print(sentence)
                    words = sentence.split(" ")

                    tags = ["O"] * len(words)

                    words = [remove_punctuation(word) for word in words]

                    # print(words)
                    # print(tags)

                    for entity in entities:
                        start = entity[0]
                        end = entity[1]
                        tag = entity[2]
                        split_entity = sentence[start:end].split(" ")
                        # print(split_entity)

                        indices = find_subset_indices(split_entity, words)

                        # print(indices)

                        if indices is not None:
                            for i, index in enumerate(indices):
                                if i == 0:
                                    tags[index] = f"B-{tag}"
                                else:
                                    tags[index] = f"I-{tag}"
                        
                        # print(tags)
                    
                    for word, tag in zip(words, tags):
                        csv_file.write(f"{sentence_index}, {word}, {tag}, variable_declaration\n")

                    sentence_index += 1

def read_dataset():
    data = pd.read_csv('./ner_dataset/ner_dataset.csv', encoding='latin1')

    # remove white spaces from column names
    data.columns = data.columns.str.strip()

    print(data.columns)
    # print(data.columns)
    # Group by 'Sentence #' and aggregate
    grouped_data = data.groupby('Sentence #').agg({
        'Word': lambda x: ' '.join(x),  # Join words into a single sentence
        'Tag': lambda x: list(x),       # Collect tags into a list
        'Intent': lambda x: x     # Collect intents into a list
    }).reset_index()  # Reset index to make 'Sentence #' a regular column

    return data, grouped_data


def prepare_data(dataframe):
    dataset = []
    for _, row in dataframe.iterrows():
        sentence = row['Word']
        tags = row['Tag']
        intents = row['Intent'][0]
        dataset.append((sentence, tags, intents))

    return dataset


def argmax(vec, axis=0):
    '''
    returns the argmax as a python int
    '''
    _, idx = torch.max(vec, axis) # axis=0: column-wise (2D tensor), axis=1: row-wise (2D tensor)
    return idx.item()

def prepare_sequence(sequence, to_index):
    '''
    converts a sequence of words to a tensor of indices
    sequence: list of words
    to_index: dictionary mapping words to indices
    
    example:
    sequence = ['The', 'dog', 'barked']
    word_to_index = {'The': 0, 'dog': 1, 'barked': 2}
    output = tensor([0, 1, 2])
    '''
    indices = [to_index[w] for w in sequence]
    return torch.tensor(indices, dtype=torch.long)