import nlpaug.augmenter.word as naw
import numpy as np
import random
import nltk
import json


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
    pass
