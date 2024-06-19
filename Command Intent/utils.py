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
    
def augment_data(file_path):
    pass